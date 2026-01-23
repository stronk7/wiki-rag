#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration management for wiki-rag."""

import logging
import os
import sys

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from wiki_rag import ROOT_DIR, __version__

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager supporting both YAML and dotenv files."""

    def __init__(self, config_path: Path | None = None):
        """Initialize configuration.

        Args:
            config_path: Optional path to config.yaml. If not provided, will look for
                config.yaml in ROOT_DIR, or fall back to .env for backward compatibility.
        """
        self._config: dict[str, Any] = {}
        self._from_yaml = False

        if config_path or (ROOT_DIR / "config.yaml").exists():
            self._load_yaml(config_path or (ROOT_DIR / "config.yaml"))
        elif (ROOT_DIR / ".env").exists():
            self._load_env()
        else:
            logger.error("No config.yaml or .env file found. Exiting.")
            sys.exit(1)

    def _load_yaml(self, path: Path) -> None:
        """Load configuration from YAML file."""
        import yaml

        logger.warning(f"Loading configuration from {path}")
        try:
            with open(path) as f:
                self._config = yaml.safe_load(f) or {}
            self._from_yaml = True

            if not self._config.get("openai_api_key", ""):
                logger.error("OPENAI_API_KEY is required. Exiting.")
                sys.exit(1)

            self._validate_config()
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}. Exiting.")
            sys.exit(1)

    def _load_env(self) -> None:
        """Load configuration from .env file."""
        dotenv_file = ROOT_DIR / ".env"
        logger.warning("Loading environment variables from %s", dotenv_file)
        logger.warning("Note: .env files are deprecated. Consider using config.yaml instead.")
        load_dotenv(dotenv_file)

        self._from_yaml = False

    def _validate_config(self) -> None:
        """Validate required configuration fields."""
        required = ["openai_api_key", "mediawiki_url", "collection_name",
                    "embedding_model", "embedding_dimensions", "llm_model"]
        for field in required:
            if not self._config.get(field):
                logger.error(f"Required field '{field}' is missing from config.yaml. Exiting.")
                sys.exit(1)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._from_yaml:
            keys = key.split(".")
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        else:
            return os.getenv(key.upper(), default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        if value is None:
            return default
        return int(value)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        if value is None:
            return default
        return float(value)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, str(default))
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"

    def get_list(self, key: str, default: list[str] | None = None) -> list[str]:
        """Get list configuration value (comma-separated)."""
        value = self.get(key)
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",")]
        return default or []

    def get_int_list(self, key: str, default: list[int] | None = None) -> list[int]:
        """Get integer list configuration value."""
        value = self.get(key)
        if value is None:
            return default or []
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return default or []

    @property
    def is_yaml(self) -> bool:
        """Check if configuration was loaded from YAML."""
        return self._from_yaml


_global_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config