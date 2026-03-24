#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the document loader."""

import argparse
import logging
import os
import sys

from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__
from wiki_rag.index.util import load_parsed_information
from wiki_rag.load.util import (
    get_incremental_changes,
    get_mediawiki_pages_list,
    get_mediawiki_parsed_pages,
    merge_incremental_pages,
    save_parsed_pages,
)
from wiki_rag.util import instance_lock, setup_logging


def main():
    """Load and parse all the files, storing the information in a file with date."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-load starting up...")

    # Print the version of the bot.
    logger.warning(f"Version: {__version__}")

    parser = argparse.ArgumentParser(description="Load and parse MediaWiki pages into a dump file.")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental mode: only fetch changed pages since the base dump.",
    )
    parser.add_argument(
        "--base-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the base dump JSON for incremental mode. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--force-save",
        action="store_true",
        help=(
            "Always generate a new dump file even when no changes are detected "
            "(incremental mode only; ignored for full loads)."
        ),
    )
    args = parser.parse_args()

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
    excluded = os.getenv("MEDIAWIKI_EXCLUDED")
    exclusions = {}
    # Let's process the exclusions and return them in a nice dict.
    if excluded:
        for exclusion in excluded.split(";"):
            exclusion_type, exclusion_values = exclusion.split(":")
            exclusion_values = [value.strip() for value in exclusion_values.split(",")]
            exclusions[exclusion_type] = exclusion_values
    logger.info(f"Applying exclusions: {exclusions}")

    # The list of templates, comma separated, that we want to keep in the wiki text. Others will be removed.
    keep_templates = os.getenv("MEDIAWIKI_KEEP_TEMPLATES")
    if keep_templates:
        keep_templates = [template.strip() for template in keep_templates.split(",")]
    else:
        keep_templates = []
    logger.info(f"Keeping templates: {keep_templates}")

    collection_name = os.getenv("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in environment. Exiting.")
        sys.exit(1)

    with instance_lock(collection_name, loader_dump_path):
        # The dump datetime is now, before starting the loading. We use also for the filename.
        dump_datetime = datetime.now(UTC).replace(microsecond=0)
        dump_filename = loader_dump_path / f"{collection_name}-{dump_datetime.strftime('%Y-%m-%d-%H-%M')}.json"

        user_agent = os.getenv("USER_AGENT")
        if not user_agent:
            logger.info("User agent not found in environment. Using default.")
            user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
        user_agent = f"{user_agent.format(version=__version__)}"

        enable_rate_limiting_setting: str | None = os.getenv("ENABLE_RATE_LIMITING")
        if not enable_rate_limiting_setting:
            logger.info("ENABLE_RATE_LIMITING setting not found in environment. Using the default value 'true'")
            enable_rate_limiting_setting = "true"

        enable_rate_limiting_setting = enable_rate_limiting_setting.lower()

        if enable_rate_limiting_setting == "true":
            enable_rate_limiting = True
            logger.info("Rate limiting is enabled.")
        elif enable_rate_limiting_setting == "false":
            enable_rate_limiting = False
            logger.info("Rate limiting is disabled.")
        else:
            logger.error("ENABLE_RATE_LIMITING environment variable can only get values 'true' or 'false'. Exiting.")
            sys.exit(1)

        if args.incremental:
            # --- Incremental load mode ---
            logger.info("Incremental mode enabled.")

            # Determine the base JSON file.
            base_json_file: Path | None = args.base_json
            if not base_json_file:
                # Auto-detect the most recent matching dump in the dump directory.
                for f in sorted(loader_dump_path.iterdir()):
                    if f.is_file() and f.name.startswith(f"{collection_name}-") and f.name.endswith(".json"):
                        base_json_file = f
                if not base_json_file:
                    logger.error(
                        f"No base dump found in {loader_dump_path} for collection '{collection_name}'. Exiting."
                    )
                    sys.exit(1)
            logger.info(f"Using base dump: {base_json_file}")

            # Load and validate the base dump.
            base_information = load_parsed_information(base_json_file)
            base_site = base_information["sites"][0]
            since = base_site.get("timestamp") or base_information["meta"]["timestamp"]
            logger.info(f"Fetching changes since: {since}")

            # Fetch changed page IDs from the MediaWiki API.
            revised_page_ids, final_log_states, pages_to_fetch = get_incremental_changes(
                mediawiki_url, mediawiki_namespaces, user_agent, since, enable_rate_limiting
            )
            logger.info(
                f"Found {len(revised_page_ids)} revised page(s), "
                f"{len(final_log_states)} page(s) with delete/restore events, "
                f"{len(pages_to_fetch)} page(s) to re-fetch."
            )

            # If nothing changed, skip file generation unless --force-save was requested.
            if not revised_page_ids and not final_log_states:
                if args.force_save:
                    logger.info("No changes detected, but --force-save is set. Generating dump anyway.")
                else:
                    logger.info("No changes detected since the base dump. Skipping file generation.")
                    return

            # Parse the pages that need re-fetching using the existing pipeline.
            logger.info("Fetching, parsing and splitting changed pages")
            parsed_pages = get_mediawiki_parsed_pages(
                mediawiki_url, pages_to_fetch, user_agent, exclusions, keep_templates, enable_rate_limiting
            )
            logger.info(f"Parsed {len(parsed_pages)} changed page(s).")

            # Merge with base pages to produce a complete dump.
            base_pages = {page["id"]: page for page in base_information["sites"][0]["pages"]}
            merged_pages = merge_incremental_pages(base_pages, parsed_pages, final_log_states, revised_page_ids)
            logger.info(f"Merged dump contains {len(merged_pages)} page(s).")

            logger.info(f"Saving incremental dump to {dump_filename}")
            save_parsed_pages(
                merged_pages, dump_filename, dump_datetime, mediawiki_url,
                dump_type="incremental", base_dump=base_json_file.name,
            )
        else:
            # --- Full load mode (default) ---
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
