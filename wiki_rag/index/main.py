#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the document indexer."""

import logging
import os
import sys

from pathlib import Path

from dotenv import load_dotenv

import wiki_rag.index as index

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__
from wiki_rag.index.util import (
    create_temp_collection_schema,
    index_pages,
    load_parsed_information,
    replace_previous_collection,
)
from wiki_rag.util import setup_logging


def main():
    """Make an index from the json information present in the specified file."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-index starting up...")

    # Print the version of the bot.
    logger.warning(f"Version: {__version__}")

    dotenv_file = ROOT_DIR / ".env"
    if dotenv_file.exists():
        logger.warning("Loading environment variables from %s", dotenv_file)
        logger.warning("Note: .env files are not supposed to be used in production. Use env secrets instead.")
        load_dotenv(dotenv_file)

    mediawiki_url = os.getenv("MEDIAWIKI_URL")
    if not mediawiki_url:
        logger.error("Mediawiki URL not found in environment. Exiting.")
        sys.exit(1)

    mediawiki_namespaces = os.getenv("MEDIAWIKI_NAMESPACES")
    if not mediawiki_namespaces:
        logger.error("Mediawiki namespaces not found in environment. Exiting.")
        sys.exit(1)
    mediawiki_namespaces = mediawiki_namespaces.split(",")
    mediawiki_namespaces = [int(ns.strip()) for ns in mediawiki_namespaces]  # no whitespace and int.
    mediawiki_namespaces = list(set(mediawiki_namespaces))  # unique

    loader_dump_path = os.getenv("LOADER_DUMP_PATH")
    if loader_dump_path:
        loader_dump_path = Path(loader_dump_path)
    else:
        loader_dump_path = ROOT_DIR / "data"
    # If the directory does not exist, create it.
    if not loader_dump_path.exists():
        logger.error(f"Data directory {loader_dump_path} not found. Please ensure it exists. Exiting.")
        sys.exit(1)

    collection_name = os.getenv("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in environment. Exiting.")
        sys.exit(1)

    index.milvus_url = os.getenv("MILVUS_URL")
    if not index.milvus_url:
        logger.error("Milvus URL not found in environment. Exiting.")
        sys.exit(1)

    user_agent = os.getenv("USER_AGENT")
    if not user_agent:
        logger.info("User agent not found in environment. Using default.")
        user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
    user_agent = f"{user_agent.format(version=__version__)}"

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        logger.error("Embedding model not found in environment. Exiting.")
        sys.exit(1)

    embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    if not embedding_dimensions:
        logger.error("Embedding dimensions not found in environment. Exiting.")
        sys.exit(1)
    embedding_dimensions = int(embedding_dimensions)

    input_candidate = ""
    # TODO: Implement CLI argument to accept the input file here.

    # No candidate yet, let's find the last file in the directory (with collection_name
    # as prefix to filter out other files).
    for file in sorted(loader_dump_path.iterdir()):
        if file.is_file() and file.name.startswith(collection_name) and file.name.endswith(".json"):
            input_candidate = file
    if not input_candidate:
        logger.error(f"No input file found in {loader_dump_path} with collection name {collection_name}. Exiting.")
        sys.exit(1)

    # TODO: Make this to accept CLI argument or, by default, use the last file in the directory.
    input_file = loader_dump_path / input_candidate

    logger.info(f"Loading parsed pages from json: {input_file}, namespaces: {mediawiki_namespaces}")
    pages = load_parsed_information(input_file)
    logger.info(f"Loaded {len(pages)} pages from json file")

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
