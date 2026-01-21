#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Generate `config.yaml` from a legacy `.env` file."""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__
from wiki_rag.config import CONFIG_FILE_NAME, dump_config_yaml, generate_config_yaml_from_dotenv
from wiki_rag.util import setup_logging


def main() -> None:
    """Convert `.env` settings into `config.yaml` (no secrets)."""

    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)

    logger.info("wiki_rag-config-update starting up...")
    logger.warning(f"Version: {__version__}")

    parser = argparse.ArgumentParser(
        prog="wr-config-update",
        description="Convert legacy .env configuration into config.yaml (secrets omitted)",
    )
    parser.add_argument(
        "--dotenv",
        default=str(ROOT_DIR / ".env"),
        help="Path to the source .env file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / CONFIG_FILE_NAME),
        help="Path to write config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config.yaml if present",
    )

    args = parser.parse_args()

    dotenv_path = Path(args.dotenv)
    output_path = Path(args.output)

    if not dotenv_path.exists():
        logger.error(f"Dotenv file not found: {dotenv_path}")
        sys.exit(1)

    if output_path.exists() and not args.force:
        logger.error(f"Output file already exists: {output_path}. Use --force to overwrite.")
        sys.exit(2)

    config = generate_config_yaml_from_dotenv(dotenv_path)
    output_path.write_text(dump_config_yaml(config), encoding="utf-8")

    logger.info(f"Wrote {output_path}")
    logger.warning("Secrets are not written to YAML; set them via OS environment variables")


if __name__ == "__main__":
    main()
