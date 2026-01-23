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
from wiki_rag.config import get_config
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

    logger.warning(f"Version: {__version__}")

    config = get_config()

    mediawiki_url = config.get("mediawiki.url")
    if not mediawiki_url:
        logger.error("Mediawiki URL not found in configuration. Exiting.")
        sys.exit(1)

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

    exclusions = config.get("mediawiki.excluded", {})
    logger.info(f"Applying exclusions: {exclusions}")

    keep_templates = config.get_list("mediawiki.keep_templates", [])
    logger.info(f"Keeping templates: {keep_templates}")

    collection_name = config.get("collection.name")
    if not collection_name:
        logger.error("Collection name not found in configuration. Exiting.")
        sys.exit(1)
    dump_datetime = datetime.now(UTC).replace(microsecond=0)
    dump_filename = loader_dump_path / f"{collection_name}-{dump_datetime.strftime('%Y-%m-%d-%H-%M')}.json"

    user_agent = config.get("crawler.user_agent", f"Moodle Research Crawler/{{version}} (https://git.in.moodle.com/research)")
    user_agent = user_agent.format(version=__version__)

    enable_rate_limiting = config.get_bool("crawler.rate_limiting", True)
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
