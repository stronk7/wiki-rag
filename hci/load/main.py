#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause
import json
import pprint
from datetime import datetime, timezone
from uuid import UUID

from hci import ROOT_DIR, __version__, LOG_LEVEL
from hci.load.util import get_mediawiki_pages_list, get_mediawiki_parsed_pages
from hci.util import setup_logging

from dotenv import load_dotenv
import sys
import os

import logging


def main():
    """ Load and parse all the files, storing the information in a file with date"""

    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("hci_load starting up...")

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

    mediawiki_namespaces = os.getenv("MEDIAWIKI_NAMESPACES").split(',')
    if not mediawiki_namespaces:
        logger.error("Mediawiki namespaces not found in environment. Exiting.")
        sys.exit(1)
    mediawiki_namespaces = [int(ns.strip()) for ns in mediawiki_namespaces] # no whitespace and int.
    mediawiki_namespaces = list(set(mediawiki_namespaces)) # unique
    
    collection_name = os.getenv("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in environment. Exiting.")
        sys.exit(1)
    # File name is the collection name + toady's date and time (hours and minutes) + .json
    dump_filename = f"{collection_name}-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M')}.json"

    user_agent = os.getenv("USER_AGENT")
    if not user_agent:
        logger.info("User agent not found in environment. Using default.")
        user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
    user_agent = f"{user_agent.format(version=__version__)}"

    logger.info(f"Pre-loading page list for mediawiki: {mediawiki_url}, namespaces: {mediawiki_namespaces}")
    pages = get_mediawiki_pages_list(mediawiki_url, mediawiki_namespaces, user_agent)
    logger.info(f"Loaded {len(pages)} pages.")

    logger.info(f"Parsing and splitting pages")
    parsed_pages = get_mediawiki_parsed_pages(mediawiki_url, pages, user_agent)
    logger.info(f"Parsed {len(parsed_pages)} pages.")


    logger.info(f"Saving parsed pages to {dump_filename}")

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, UUID):
                return str(obj)
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)

    with open(dump_filename, "w") as f:
        json.dump(pages, f, cls=CustomEncoder)
    


if __name__ == "__main__":
    main()
