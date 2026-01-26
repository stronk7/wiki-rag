"""CLI commands for configuration management.

This module provides the wr-config-update command that handles
migration from .env to config.yaml format.
"""

import argparse
import logging
import sys
from pathlib import Path

from wiki_rag import ROOT_DIR
from wiki_rag.config import ConfigUpdater
from wiki_rag.config.loader import ConfigManager
from wiki_rag.util import setup_logging


logger = logging.getLogger(__name__)


def config_update() -> None:
    """CLI command for updating configuration."""
    parser = argparse.ArgumentParser(
        description="Manage wiki-rag configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # migrate command
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate .env to config.yaml format"
    )
    migrate_parser.add_argument(
        "--from-env",
        type=Path,
        help="Path to .env file (default: ./.env)"
    )
    migrate_parser.add_argument(
        "--to",
        type=Path,
        help="Path to output config.yaml file (default: ./config.yaml)"
    )
    migrate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing config.yaml"
    )
    
    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration file"
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yaml file"
    )
    
    # show-diff command
    show_diff_parser = subparsers.add_parser(
        "show-diff",
        help="Show differences between .env and config.yaml"
    )
    show_diff_parser.add_argument(
        "--from-env",
        type=Path,
        help="Path to .env file"
    )
    
    # generate-template command
    template_parser = subparsers.add_parser(
        "generate-template",
        help="Generate a template config.yaml file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    if not args.command:
        parser.print_help()
        logger.error("No command specified")
        sys.exit(1)
    
    # Process commands
    if args.command == "migrate":
        dotenv_path = args.from_env or ROOT_DIR / ".env"
        config_path = args.to or ROOT_DIR / "config.yaml"
        
        updater = ConfigUpdater(dotenv_path=dotenv_path, config_path=config_path)
        
        try:
            updater.migrate(overwrite=args.overwrite)
            logger.info("Configuration migration completed successfully")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            sys.exit(1)
    
    elif args.command == "validate":
        config_path = args.config or ROOT_DIR / "config.yaml"
        updater = ConfigUpdater(config_path=config_path)
        
        try:
            if updater.validate_config(config_path):
                logger.info("Configuration is valid")
            else:
                sys.exit(1)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            sys.exit(1)
    
    elif args.command == "show-diff":
        dotenv_path = args.from_env or ROOT_DIR / ".env"
        updater = ConfigUpdater(dotenv_path=dotenv_path)
        
        try:
            diff = updater.show_diff()
            print(diff)
        except Exception as e:
            logger.error(f"Failed to show diff: {e}")
            sys.exit(1)
    
    elif args.command == "generate-template":
        updater = ConfigUpdater()
        
        try:
            updater.generate_template()
            logger.info("Template generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate template: {e}")
            sys.exit(1)
    
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)
