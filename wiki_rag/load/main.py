#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the document loader."""

import logging
import os
import sys

from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__
from wiki_rag.load.util import (
    get_mediawiki_pages_list,
    get_mediawiki_parsed_pages,
    save_parsed_pages,
)
from wiki_rag.util import setup_logging


def main():
    """Load and parse all the files, storing the information in a file with date."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-load starting up...")

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

    # The format for now is a semicolon separated list of "type", colon and comma
    # separated list of values, for example:
    # MEDIAWIKI_EXCLUDED="categories:Plugin, Contributed code;wikitext:Hi world, ho world
    # TODO: Move this also to the config YAML file.
    excluded = os.getenv("MEDIAWIKI_EXCLUDED")
    exclusions = {}
    # Let's process the exclusions and return them in a nice dict.
    if excluded:
        for exclusion in excluded.split(";"):
            exclusion_type, exclusion_values = exclusion.split(":")
            exclusion_values = [value.strip() for value in exclusion_values.split(",")]
            exclusions[exclusion_type] = exclusion_values
    logger.info(f"Applying exclusions: {exclusions}")

    collection_name = os.getenv("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in environment. Exiting.")
        sys.exit(1)
    # File name is the collection name + toady's date and time (hours and minutes) + .json
    dump_filename = loader_dump_path / f"{collection_name}-{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M')}.json"

    user_agent = os.getenv("USER_AGENT")
    if not user_agent:
        logger.info("User agent not found in environment. Using default.")
        user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
    user_agent = f"{user_agent.format(version=__version__)}"

    logger.info(f"Pre-loading page list for mediawiki: {mediawiki_url}, namespaces: {mediawiki_namespaces}")
    pages = get_mediawiki_pages_list(mediawiki_url, mediawiki_namespaces, user_agent)
    logger.info(f"Loaded {len(pages)} pages.")

    logger.info("Parsing and splitting pages")
    parsed_pages = get_mediawiki_parsed_pages(mediawiki_url, pages, user_agent, exclusions)
    logger.info(f"Parsed {len(parsed_pages)} pages.")

    logger.info(f"Saving parsed pages to {dump_filename}")
    save_parsed_pages(parsed_pages, dump_filename)

    logger.info(f"wiki_rag-load finished.")  # noqa: F541


if __name__ == "__main__":
    main()
