#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the document indexer."""

import logging
import sys

from pathlib import Path

import wiki_rag.vector as vector

from wiki_rag import ROOT_DIR, __version__
from wiki_rag import LOG_LEVEL
from wiki_rag.config import get_config
from wiki_rag.index.util import (
    create_temp_collection_schema,
    index_pages,
    load_parsed_information,
    replace_previous_collection,
)
from wiki_rag.util import setup_logging
from wiki_rag.vector import load_vector_store


def main():
    """Make an index from the json information present in the specified file."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-index starting up...")

    logger.warning(f"Version: {__version__}")

    config = get_config()

    mediawiki_url = config.get("mediawiki.url")
    mediawiki_namespaces = config.get_int_list("mediawiki.namespaces")

    loader_dump_path_str = config.get("collection.dump_path")
    if loader_dump_path_str:
        loader_dump_path = Path(loader_dump_path_str)
    else:
        loader_dump_path = ROOT_DIR / "data"
    if not loader_dump_path.exists():
        logger.warning(f"Data directory {loader_dump_path} not found. Creating it.")
        try:
            loader_dump_path.mkdir()
        except Exception:
            logger.error(f"Could not create data directory {loader_dump_path}. Exiting.")
            sys.exit(1)

    collection_name = config.get("collection.name")
    index_vendor = config.get("index_vendor", "milvus")

    user_agent = config.get("crawler.user_agent", f"Moodle Research Crawler/{{version}} (https://git.in.moodle.com/research)")
    user_agent = user_agent.format(version=__version__)

    embedding_model = config.get("models.embedding")
    embedding_dimensions = config.get_int("models.embedding_dimensions")

    vector.store = load_vector_store(index_vendor)

    input_candidate = ""
    for file in sorted(loader_dump_path.iterdir()):
        if file.is_file() and file.name.startswith(f"{collection_name}-") and file.name.endswith(".json"):
            input_candidate = file
    if not input_candidate:
        logger.error(f"No input file found in {loader_dump_path} with collection name {collection_name}. Exiting.")
        sys.exit(1)

    input_file = loader_dump_path / input_candidate

    logger.info(f"Loading parsed pages from JSON: {input_file}, namespaces: {mediawiki_namespaces}")
    information = load_parsed_information(input_file)
    pages = information["sites"][0]["pages"]
    logger.info(f"Loaded {len(pages)} pages from JSON file")

    temp_collection_name = f"{collection_name}_temp"
    logger.info(f'Preparing new temp collection "{temp_collection_name}" schema')
    create_temp_collection_schema(temp_collection_name, embedding_dimensions)
    logger.info(f'Collection "{temp_collection_name}" created.')

    logger.info(f'Indexing pages into temp collection "{temp_collection_name}"')
    [total_pages, total_sections] = index_pages(pages, temp_collection_name, embedding_model, embedding_dimensions)
    logger.info(f"Indexed {total_pages} pages ({total_sections} sections/chunks).")

    logger.info(f'Replacing previous collection "{collection_name}" with new collection "{temp_collection_name}"')
    replace_previous_collection(collection_name, temp_collection_name)
    logger.info(f"Collection {collection_name} replaced with {temp_collection_name}.")

    logger.info("wiki_rag-index finished.")


if __name__ == "__main__":
    main()