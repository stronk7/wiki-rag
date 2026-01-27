#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration management for Wiki-RAG.

This module handles loading configuration from environment variables (.env)
and a YAML configuration file (config.yaml).

Priority: Environment Variables > config.yaml > Defaults
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file immediately upon import to ensure os.environ is populated
# for backward compatibility and mixed usage.
load_dotenv()


class Config:
    """Configuration singleton."""

    _instance = None
    _config_data: dict[str, Any] = {}
    _yaml_path = Path("config.yaml")

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_yaml()
        return cls._instance

    def _load_yaml(self):
        """Load configuration from config.yaml if it exists."""
        if self._yaml_path.exists():
            try:
                with open(self._yaml_path, "r") as f:
                    self._config_data = yaml.safe_load(f) or {}
            except Exception as e:
                # We log to console here as logging might not be set up yet
                print(f"Warning: Failed to load config.yaml: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Priority:
        1. Environment Variable (os.getenv)
        2. YAML Configuration
        3. Default value
        """
        # 1. Environment Variable
        env_val = os.getenv(key)
        if env_val is not None:
            return env_val

        # 2. YAML Configuration
        if key in self._config_data:
            return self._config_data[key]

        # 3. Default
        return default

    # Type-safe getters for common types
    def get_str(self, key: str, default: str = "") -> str:
        """Get a string value."""
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer value."""
        val = self.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean value.

        Handles strings like "true", "True", "1" as True.
        """
        val = self.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes", "on")
        return bool(val)

    def get_list(self, key: str, default: list | None = None) -> list:
        """Get a list value.

        Handles comma-separated strings from env vars.
        """
        val = self.get(key)
        if val is None:
            return default or []
        
        if isinstance(val, list):
            return val
        
        if isinstance(val, str):
            # Parse comma-separated string, stripping whitespace
            return [item.strip() for item in val.split(",") if item.strip()]
            
        return [val]


# Global instance
settings = Config()
