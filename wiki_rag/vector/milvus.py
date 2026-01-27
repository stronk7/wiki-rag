#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Milvus-specific implementation of the generic vector interface.

Worth commenting that we are using pymilvus native SDK and not LangChain's own one, because
the former is incomplete and does not support all the features used here. In general, always
use the complete SDKs is a good recommendation.
"""

import logging
import os
import sys

from typing import Any

from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
)

from wiki_rag.config import config
from wiki_rag.vector import BaseVector

logger = logging.getLogger(__name__)


class MilvusVector(BaseVector):
    """Milvus backend vector.

    Requires MILVUS_URL to be defined in environment or database.milvus_url in config.yaml.
    Milvus connection string, e.g. 'http://localhost:19530'
    or 'https://user:password@localhost:19530'.  # pragma: allowlist secret
    """

    def __init__(self) -> None:
        """Initialize the Milvus backend."""
        # TODO: We'll need to change this to use config when we have it (vs env).
        self.uri: str = config.get("database.milvus_url", "")
        if not self.uri:
            logger.error("Milvus URL not found in config or environment. Exiting.")
            sys.exit(1)

    # BaseVector interface.

    def create_collection(self, collection_name: str, embedding_dimension: int) -> None:
        """Create (or recreate) a Milvus collection with the required schema.

        A pre-existing collection with the same name is dropped first.
        Both HNSW (dense) and BM25 (sparse) indexes are created for hybrid search.
        """
        client = MilvusClient(self.uri)
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)

        schema = self._build_schema(embedding_dimension)
        index_params = self._build_index_params(client)
        client.create_collection(collection_name, schema=schema, index_params=index_params)
        client.close()

    def collection_exists(self, name: str) -> bool:
        """Return True if the Milvus collection exists."""
        client = MilvusClient(self.uri)
        exists = client.has_collection(name)
        client.close()
        return True if exists else False

    def drop_collection(self, name: str) -> None:
        """Delete the Milvus collection."""
        client = MilvusClient(self.uri)
        client.drop_collection(name)
        client.close()

    def rename_collection(self, old: str, new: str) -> None:
        """Rename a Milvus collection atomically."""
        client = MilvusClient(self.uri)
        client.rename_collection(old, new)
        client.close()

    def compact_collection(self, name: str) -> None:
        """Trigger Milvus compaction to reclaim disk space and optimise performance."""
        client = MilvusClient(self.uri)
        client.compact(name)
        client.close()

    def insert_batch(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        """Insert a batch of records into the Milvus collection.

        Args:
            collection_name: Target Milvus collection.
            records: List of dictionaries that contain the fields defined
                in the schema created by `create_collection`.

        """
        client = MilvusClient(self.uri)
        try:
            client.insert(collection_name, records)
        except Exception:
            raise
        finally:
            client.close()

    def get_documents_contents_by_id(
        self,
        collection_name: str,
        ids: list[str],
    ) -> dict[str, str]:
        """Retrieve documents (output columns) matching the given ids.

        Args:
            collection_name: Target collection / index.
            ids: List of ids to retrieve.

        Returns:
            dictionary of document ids as keys and document contents as values.

        """
        output_columns = ["id", "title", "text"]

        milvus = MilvusClient(self.uri)

        # Let's find in the collection, the missing elements and get their titles and texts.
        missing_docs_db = milvus.query(
            collection_name, ids=ids, output_fields=output_columns
        )

        # Return a simple dictionary of
        missing_docs = {
            doc["id"]: f"{doc['title']}\n\n{doc['text']}" for doc in missing_docs_db
        }
        milvus.close()
        return missing_docs

    def retrieve(self,
            collection_name: str,
            embedding_model: str,
            embedding_dimensions: int,
            query: str,
    ) -> list[dict]:
        """Retrieve the best matches for a question from the vector store.

        Here we'll be using Milvus hybrid search that performs a vector search (dense, embeddings)
        and a BM25 search (sparse, full text). And then will rerank results with the weighted
        reranker.

        Worth commenting that we are using pymilvus native SDK and not LangChain's own one, because
        the former is incomplete and does not support all the features used here. In general, always
        use the complete SDKs is a good recommendation.
        """
        # Get the embeddings to look for in the dense search.
        embeddings = self._get_query_embeddings(
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            query=query,
        )

        client = MilvusClient(self.uri)

        # TODO: Make a bunch of the defaults used here configurable.
        dense_search_limit = 20
        sparse_search_limit = 20
        sparse_search_drop_ratio = 0.2
        hybrid_rerank_limit = 30
        rerank_weights = (0.7, 0.3)

        # Define the dense search and its parameters.
        dense_search_params = {
            "metric_type": "IP",
            "params": {
                "ef": dense_search_limit,
            },
        }
        dense_search = AnnSearchRequest(
            [embeddings],
            "dense_vector",
            dense_search_params,
            limit=dense_search_limit,
        )

        # Define the sparse search and its parameters.
        sparse_search_params = {
            "metric_type": "BM25",
            "drop_ratio_search": sparse_search_drop_ratio,
        }
        sparse_search = AnnSearchRequest(
            [query],
            "sparse_vector",
            sparse_search_params,
            limit=sparse_search_limit,
        )

        # Perform the hybrid search.
        retrieved_docs = client.hybrid_search(
            collection_name,
            [dense_search, sparse_search],
            WeightedRanker(*rerank_weights),
            limit=hybrid_rerank_limit,
            output_fields=[
                "id",
                "title",
                "text",
                "source",
                "doc_id",
                "doc_title",
                "doc_hash",
                "parent",
                "children",
                "previous",
                "next",
                "relations",
                "page_id",
            ],
        )
        client.close()

        # Need this: Langfuse has problems with Milvus Hit objects, that are UserDict, hence not JSON serializable.
        # Reported @ https://github.com/langfuse/langfuse/issues/9294 , we'll need to keep the workaround, it seems.
        results = [
            dict(doc) for doc in retrieved_docs[0]
        ]
        return results

    # Internal helpers.

    def _build_schema(self, embedding_dimensions: int) -> CollectionSchema:
        """Build the Milvus schema expected by the wiki_rag ingestion pipe.

        Args:
            embedding_dimensions: Dimensionality of the dense vector field.

        Returns:
            A `CollectionSchema` object ready to be used with
            `MilvusClient.create_collection`.

        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000, enable_analyzer=True,
                        analyzer_params={"type": "english"}, enable_match=True, ),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimensions),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="parent", dtype=DataType.VARCHAR, max_length=100, nullable=True),
            FieldSchema(name="children", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                        max_capacity=100, is_array=True),
            FieldSchema(name="previous", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                        max_capacity=100, is_array=True),
            FieldSchema(name="next", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                        max_capacity=100, is_array=True),
            FieldSchema(name="relations", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                        max_capacity=100, is_array=True),
            FieldSchema(name="page_id", dtype=DataType.INT32),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="doc_hash", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],  # Input text field
            output_field_names=["sparse_vector"],  # Internal mapping sparse vector field
            function_type=FunctionType.BM25,  # Model for processing mapping relationship
        )
        schema.add_function(bm25_function)

        return schema

    def _build_index_params(self, client: MilvusClient):  # type: ignore
        """Prepare index parameters for both dense (HNSW/IP) and sparse (BM25) vectors."""
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="IP",
                               params={"M": 64, "efConstruction": 100})
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                               params={"inverted_index_algo": "DAAT_WAND", "drop_ratio_build": 0.2})

        return index_params
