#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

""" Util functions to proceed to index the information to Milvus collection"""
import json
import logging

from langchain_openai import OpenAIEmbeddings

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Function, FunctionType

from tqdm import tqdm


logger = logging.getLogger(__name__)

def load_parsed_information(input_file: str) -> list[dict]:
    """Load the parsed information from the file."""
    pages = []
    try:
        with open(input_file, "r") as f:
            pages = json.load(f)
    except Exception as e:
        logger.error(f"Error loading the parsed information from {input_file}: {e}")
    
    return pages

def create_temp_collection_schema(collection_name: str, embedding_dimension: int) -> None:
    """Create a temporary schema for the collection."""

    milvus = MilvusClient("http://localhost:19530")
    if milvus.has_collection(collection_name):
        milvus.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000, enable_analyzer=True,
                    analyzer_params={"type": "english"}, enable_match=True, ),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimension),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="parent", dtype=DataType.VARCHAR, max_length=100, nullable=True),
        FieldSchema(name="children", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                    max_capacity=100, is_array=True),
        FieldSchema(name="previous", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                    max_capacity=100, is_array=True),
        FieldSchema(name="next", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000, max_capacity=100,
                    is_array=True),
        FieldSchema(name="relations", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=4000,
                    max_capacity=100, is_array=True),
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

    index_params = milvus.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="IP",
                           params={"M": 64, "efConstruction": 100})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                           params={"inverted_index_algo": "DAAT_WAND", "drop_ratio_build": 0.2})

    milvus.create_collection(collection_name, schema=schema, index_params=index_params)

    milvus.close()
    
def index_pages(pages: list[dict], collection_name: str, embedding_model: str, embedding_dimension: int) -> [int, int]:
    """Index the pages to the collection."""
    milvus = MilvusClient("http://localhost:19530")

    logging.getLogger("httpx").setLevel(logging.WARNING)

    embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dimension)
    
    num_pages = 0
    num_sections = 0

    for page in tqdm(pages, desc=f"Processing pages", unit="pages"):
        for section in page["sections"]:
            if len(section["text"]) > 5000:
                # TODO: We need to split the text in smaller chunks here, say 2500 max or so. For now, just trim.
                section["text"] = section["text"][:5000]
                logger.warning(f"Text too long for section \"{section["doc_title"]} / {section["title"]}\", trimmed to 5000 characters.")
            
            dense_embedding = embeddings.embed_documents(
                [f"{section["doc_title"]} / {section["title"]}\n\n{section["text"]}"])
            data = [
                {
                    "id": str(section["id"]),
                    "title": section["title"],
                    "text": section["text"],
                    "source": section["source"],
                    "dense_vector": dense_embedding[0],
                    "parent": str(section["parent"]) if section["parent"] else None,
                    "children": [str(child) for child in section["children"]],
                    "previous": [str(prv) for prv in section["previous"]],
                    "next": [str(nxt) for nxt in section["next"]],
                    "relations": [str(rel) for rel in section["relations"]],
                    "doc_id": str(section["doc_id"]),
                    "doc_title": section["doc_title"],
                    "doc_hash": str(section["doc_hash"]),
                }
            ]
            try:
                milvus.insert(collection_name, data)
                num_sections += 1
            except Exception as e:
                print(e)
        num_pages += 1

    milvus.close()
    return [num_pages, num_sections]

def replace_previous_collection(collection_name: str, temp_collection_name: str) -> None:
    """Replace the previous collection with the new one."""
    
    milvus = MilvusClient("http://localhost:19530")
    
    if not milvus.has_collection(temp_collection_name):
        raise ValueError(f"Collection {temp_collection_name} does not exist.")
    
    if milvus.has_collection(collection_name):
        milvus.drop_collection(collection_name)
    milvus.rename_collection(temp_collection_name, collection_name)
    
    # We have inserted lots of date to the collection, let's compact it.
    logger.info(f"Compacting collection {collection_name}")
    milvus.compact(collection_name)
    
    milvus.close()