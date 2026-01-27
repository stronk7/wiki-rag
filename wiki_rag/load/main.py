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
from wiki_rag.config import settings
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

    mediawiki_url = settings.get_str("MEDIAWIKI_URL")
    if not mediawiki_url:
        logger.error("Mediawiki URL not found in configuration. Exiting.")
        sys.exit(1)

    mediawiki_namespaces = settings.get_list("MEDIAWIKI_NAMESPACES")
    if not mediawiki_namespaces:
        logger.error("Mediawiki namespaces not found in configuration. Exiting.")
        sys.exit(1)
    mediawiki_namespaces = [int(ns) for ns in mediawiki_namespaces]  # no whitespace and int.
    mediawiki_namespaces = list(set(mediawiki_namespaces))  # unique

    loader_dump_path = settings.get_str("LOADER_DUMP_PATH")
    if loader_dump_path:
        loader_dump_path = Path(loader_dump_path)
    else:
        loader_dump_path = ROOT_DIR / "data"
    # If the directory does not exist, create it.
    if not loader_dump_path.exists():
        logger.warning(f"Data directory {loader_dump_path} not found. Creating it.")
        try:
            loader_dump_path.mkdir()
        except Exception:
            logger.error(f"Could not create data directory {loader_dump_path}. Exiting.")
            sys.exit(1)

    # The format for now is a semicolon separated list of "type", colon and comma
    # separated list of values, for example:
    # MEDIAWIKI_EXCLUDED="categories:Plugin, Contributed code;wikitext:Hi world, ho world
    # TODO: Move this also to the config YAML file.
    excluded = settings.get_str("MEDIAWIKI_EXCLUDED")
    exclusions = {}
    # Let's process the exclusions and return them in a nice dict.
    if excluded:
        for exclusion in excluded.split(";"):
            exclusion_type, exclusion_values = exclusion.split(":")
            exclusion_values = [value.strip() for value in exclusion_values.split(",")]
            exclusions[exclusion_type] = exclusion_values
    logger.info(f"Applying exclusions: {exclusions}")

    # The list of templates, comma separated, that we want to keep in the wiki text. Others will be removed.
    keep_templates = settings.get_list("MEDIAWIKI_KEEP_TEMPLATES")
    logger.info(f"Keeping templates: {keep_templates}")

    collection_name = settings.get_str("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in configuration. Exiting.")
        sys.exit(1)
    # The dump datetime is now, before starting the loading. We use also for the filename.
    dump_datetime = datetime.now(UTC).replace(microsecond=0)
    dump_filename = loader_dump_path / f"{collection_name}-{dump_datetime.strftime('%Y-%m-%d-%H-%M')}.json"

    user_agent = settings.get_str("USER_AGENT")
    if not user_agent:
        logger.info("User agent not found in configuration. Using default.")
        user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
    user_agent = f"{user_agent.format(version=__version__)}"

    # Default to True unless explicitly set to false
    enable_rate_limiting = settings.get_bool("ENABLE_RATE_LIMITING", True)

    if enable_rate_limiting:
        logger.info("Rate limiting is enabled.")
    else:
        logger.info("Rate limiting is disabled.")

    logger.info(f"Pre-loading page list for mediawiki: {mediawiki_url}, namespaces: {mediawiki_namespaces}")
    pages = get_mediawiki_pages_list(
        mediawiki_url, mediawiki_namespaces, user_agent, 500, enable_rate_limiting
    )
    logger.info(f"Loaded {len(pages)} pages.")

    logger.info("Fetching, parsing and splitting pages")
    parsed_pages = get_mediawiki_parsed_pages(
        mediawiki_url, pages, user_agent, exclusions, keep_templates, enable_rate_limiting
    )
    logger.info(f"Parsed {len(parsed_pages)} pages.")

    logger.info(f"Saving parsed pages to {dump_filename}")
    save_parsed_pages(parsed_pages, dump_filename, dump_datetime, mediawiki_url)

    logger.info("wiki_rag-load finished.")


if __name__ == "__main__":
    main()
