#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Base vector interface that every vector-store backend must implement.

All vector implementations must inherit from :class:`GenericVector` and
override every abstract method.  The rest of the application only
interacts with the vector store through this narrow contract, making it
trivial to swap backends by changing the `INDEX_VENDOR` environment
variable. And a few more, specific for each vector store (url, key, port, ...).
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import OpenAIEmbeddings


class BaseVector(ABC):
    """Minimal contract that every vector-store backend must implement."""

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_dimension: int) -> None:
        """Create (or recreate) a collection / index with the required schema.

        If a collection with the same name already exists it must be
        dropped first.

        Args:
            collection_name: Name of the target collection / index.
            embedding_dimension: Dimensionality of the dense vector that
                will be stored (e.g. 778, 1024, ...).

        """

    @abstractmethod
    def insert_batch(self, collection_name: str, records: list[dict[str, Any]]) -> None:
        """Insert a batch of records into the collection.

        Args:
            collection_name: Target collection / index.
            records: List of dictionaries that contain at least the keys
                defined in the schema built by `create_collection`.

        """

    @abstractmethod
    def get_documents_contents_by_id(self,
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

    @abstractmethod
    def retrieve(self,
        collection_name: str,
        embedding_model: str,
        embedding_dimensions: int,
        query: str,
    ) -> list[dict]:
        """Retrieve the best matches for a question from the vector store.

        The function is in charge of converting the query to the specified embeddings,
        using the specified embedding model and dimensions. Note that there is the
        'self._get_query_embeddings()' function that supports any OpenAI compatible
        embeddings entry point available to get that done for any vector store.

        Args:
            collection_name: Target collection / index.
            embedding_model: Embedding model to use.
            embedding_dimensions: Embedding dimensions to use.
            query: Query to calculate the embeddings.

        Returns:
            list of matching results

        """

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Return True if the collection / index exists.

        Args:
            name: Collection / index name.

        Returns:
            Existence flag.

        """

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete the collection / index.

        Args:
            name: Collection / index name.

        """

    @abstractmethod
    def rename_collection(self, old: str, new: str) -> None:
        """Rename a collection / index (atomic operation provided by most stores).

        Args:
            old: Current name.
            new: Desired name.

        """

    @abstractmethod
    def compact_collection(self, name: str) -> None:
        """Trigger maintenance / compaction on the collection / index.

        For stores that do not support compaction this method must be a
        no-op.

        Args:
            name: Collection / index name.

        """

    def _get_query_embeddings(self,
        embedding_model: str,
        embedding_dimensions: int,
        query: str,
    ) -> list[float]:
        """Return embeddings for a given a query.

        Using an available OpenAI compatible embedding entry point, an embedding model and the
        desired embedding dimensions, return the query embeddings. The entry point is defined
        automatically with the OPENAI_API_BASE and OPEN_API_KEY environment variables.

        Note that, at the moment, we are using LangChain OpenAI embeddings, because we are
        already using it for other dependencies (LangGraph, LangSmith, ...) but this could be
        replaced by upstream OpenAI SDK or any other embeddings API.

        Args:
            embedding_model: Embedding model to use.
            embedding_dimensions: Embedding dimensions to use.
            query: Query to calculate the embeddings.

        Returns:
            list of (float) embeddings.

        """
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            dimensions=embedding_dimensions,
            check_embedding_ctx_length=False,
        )
        return embeddings.embed_query(query.strip())


def load_vector_store(name: str) -> BaseVector:
    """Instantiate a vector store by its name.

    For example "milvus" will instantiate a wiki_rag.vector.milvus.MilvusVector class.

    Args:
        name: Name of the vector store

    """
    module_name = f"wiki_rag.vector.{name}"
    class_name = f"{name.capitalize()}Vector"

    try:
        module = __import__(module_name, fromlist=[class_name])
        vector_class = getattr(module, class_name)
        vector_instance = vector_class()
    except Exception as exc:
        msg = f"Cannot load vector backend {class_name!r} from {module_name!r}"
        raise RuntimeError(msg) from exc

    return vector_instance
