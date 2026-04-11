#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the document loader."""

import argparse
import logging
import sys

from datetime import UTC, datetime
from pathlib import Path

from wiki_rag import __version__
from wiki_rag.config import load_config
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

    cfg = load_config(command="load")
    setup_logging(level=cfg.log_level)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-load starting up...")

    # Print the version of the bot.
    logger.warning(f"Version: {__version__}")

    logger.info(f"Applying exclusions: {cfg.mediawiki.excluded}")
    logger.info(f"Keeping templates: {cfg.mediawiki.keep_templates}")

    with instance_lock(cfg.collection_name, cfg.loader.dump_path):
        # The dump datetime is now, before starting the loading. We use also for the filename.
        dump_datetime = datetime.now(UTC).replace(microsecond=0)
        dump_filename = (
            cfg.loader.dump_path
            / f"{cfg.collection_name}-{dump_datetime.strftime('%Y-%m-%d-%H-%M')}.json"
        )

        if cfg.loader.rate_limiting:
            logger.info("Rate limiting is enabled.")
        else:
            logger.info("Rate limiting is disabled.")

        if args.incremental:
            # --- Incremental load mode ---
            logger.info("Incremental mode enabled.")

            # Determine the base JSON file.
            base_json_file: Path | None = args.base_json
            if not base_json_file:
                # Auto-detect the most recent matching dump in the dump directory.
                for f in sorted(cfg.loader.dump_path.iterdir()):
                    if (
                        f.is_file()
                        and f.name.startswith(f"{cfg.collection_name}-")
                        and f.name.endswith(".json")
                    ):
                        base_json_file = f
                if not base_json_file:
                    logger.error(
                        f"No base dump found in {cfg.loader.dump_path} "
                        f"for collection '{cfg.collection_name}'. Exiting."
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
                cfg.mediawiki.url,
                cfg.mediawiki.namespaces,
                cfg.user_agent,
                since,
                cfg.loader.rate_limiting,
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
                cfg.mediawiki.url,
                pages_to_fetch,
                cfg.user_agent,
                cfg.mediawiki.excluded,
                cfg.mediawiki.keep_templates,
                cfg.loader.rate_limiting,
            )
            logger.info(f"Parsed {len(parsed_pages)} changed page(s).")

            # Merge with base pages to produce a complete dump.
            base_pages = {page["id"]: page for page in base_information["sites"][0]["pages"]}
            merged_pages = merge_incremental_pages(base_pages, parsed_pages, final_log_states, revised_page_ids)
            logger.info(f"Merged dump contains {len(merged_pages)} page(s).")

            logger.info(f"Saving incremental dump to {dump_filename}")
            save_parsed_pages(
                merged_pages, dump_filename, dump_datetime, cfg.mediawiki.url,
                dump_type="incremental", base_dump=base_json_file.name,
            )
        else:
            # --- Full load mode (default) ---
            logger.info(
                f"Pre-loading page list for mediawiki: {cfg.mediawiki.url}, "
                f"namespaces: {cfg.mediawiki.namespaces}"
            )
            pages = get_mediawiki_pages_list(
                cfg.mediawiki.url,
                cfg.mediawiki.namespaces,
                cfg.user_agent,
                500,
                cfg.loader.rate_limiting,
            )
            logger.info(f"Loaded {len(pages)} pages.")

            logger.info("Fetching, parsing and splitting pages")
            parsed_pages = get_mediawiki_parsed_pages(
                cfg.mediawiki.url,
                pages,
                cfg.user_agent,
                cfg.mediawiki.excluded,
                cfg.mediawiki.keep_templates,
                cfg.loader.rate_limiting,
            )
            logger.info(f"Parsed {len(parsed_pages)} pages.")

            logger.info(f"Saving parsed pages to {dump_filename}")
            save_parsed_pages(parsed_pages, dump_filename, dump_datetime, cfg.mediawiki.url)

        logger.info("wiki_rag-load finished.")


if __name__ == "__main__":
    main()
