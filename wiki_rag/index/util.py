#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to proceed to index to some collection is a vector store / index."""

import json
import logging

from datetime import UTC, datetime, timedelta
from pathlib import Path

from jsonschema import ValidationError, validate
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

import wiki_rag.vector as vector

from wiki_rag import ROOT_DIR

logger = logging.getLogger(__name__)


def load_parsed_information(input_file: Path) -> dict:
    """Load the parsed information from the file."""
    information = []
    try:
        with open(input_file) as f:
            information = json.load(f)
    except Exception as e:
        logger.error(f"Error loading the parsed information from {input_file}: {e}")

    # If the old format (array of pages) is detected, let's convert it to the new format,
    # (basic information in "meta" and pages in "sites").
    if isinstance(information, list):
        logger.warning(f"Old format detected in {input_file}, converting to new format.")
        file_mod_time = datetime.fromtimestamp(input_file.stat().st_mtime, UTC)
        two_days_ago = file_mod_time - timedelta(days=2)  # ftime -48h so we don't miss anything on incremental index.
        information = {
            "meta": {
                "timestamp": two_days_ago.isoformat(),
                "num_sites": 1,
            },
            "sites": [
                {
                    "site_url": "unknown",
                    "num_pages": len(information),
                    "pages": information,
                }
            ]
        }

    # Let's validate the schema as much as we can.
    schema = json.load(open(ROOT_DIR / "wiki_rag/schema.json"))
    try:
        validate(information, schema)
        logger.debug("Successfully parsed the JSON information")
    except ValidationError as e:
        msg = f"Error validating the JSON information from {input_file}: {e}"
        logger.error(msg)
        exit(1)

    return information


def create_temp_collection_schema(collection_name: str, embedding_dimension: int) -> None:
    """Create a temporary schema for the collection."""
    vector.store.create_collection(collection_name, embedding_dimension)


def index_pages(
        pages: list[dict],
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int
) -> list[int]:
    """Index the pages to the collection."""
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Don't log (INFO) all http requests.

    embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dimension)

    num_pages = 0
    num_sections = 0

    for page in tqdm(pages, desc="Processing pages", unit="pages"):
        for section in page["sections"]:
            # Calculate the preamble text (doc + section title).
            text_preamble = section["doc_title"]
            if section["title"] != section["doc_title"]:
                text_preamble = text_preamble + f" / {section['title']}"
            text_preamble = text_preamble.strip()

            # Calculate the complete text (preamble + text, if existing).
            text_content = section["text"] if section["text"] else ""
            if len(text_content) > 5000:
                # TODO: We need to split the text in smaller chunks here, say 2500 max or so. For now, just trim.
                text_content = text_content[:5000].strip()
                logger.warning(f'Text too long for section "{text_preamble}", trimmed to 5000 characters.')
            complete_text = text_preamble + "\n\n" + text_content
            logger.debug(f"Embedding {text_preamble}, text len {len(text_content)}")

            dense_embedding = embeddings.embed_documents([complete_text])
            logger.debug(f"Embedding for {text_preamble}, dim len {len(dense_embedding[0])}")
            record = {
                "id": str(section["id"]),
                "title": section["title"],
                "text": text_content,
                "source": section["source"],
                "dense_vector": dense_embedding[0],
                "parent": str(section["parent"]) if section["parent"] else None,
                "children": [str(child) for child in section["children"]],
                "previous": [str(prv) for prv in section["previous"]],
                "next": [str(nxt) for nxt in section["next"]],
                "relations": [str(rel) for rel in section["relations"]],
                "page_id": int(section["page_id"]),
                "doc_id": str(section["doc_id"]),
                "doc_title": section["doc_title"],
                "doc_hash": str(section["doc_hash"]),
            }
            try:
                vector.store.insert_batch(collection_name, [record])
                num_sections += 1
            except Exception as e:
                logger.error(f"Failed to insert data: {e}")
        num_pages += 1

    return [num_pages, num_sections]


def replace_previous_collection(collection_name: str, temp_collection_name: str) -> None:
    """Replace the previous collection with the new one."""
    if not vector.store.collection_exists(temp_collection_name):
        msg = f"Collection {temp_collection_name} does not exist."
        raise ValueError(msg)

    if vector.store.collection_exists(collection_name):
        vector.store.drop_collection(collection_name)
    vector.store.rename_collection(temp_collection_name, collection_name)

    # We have inserted lots of date to the collection, let's compact it.
    logger.info(f"Compacting collection {collection_name}")
    vector.store.compact_collection(collection_name)
