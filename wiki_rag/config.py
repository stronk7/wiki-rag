#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration management for wiki-rag."""

import os

from typing import Any

import yaml


class Config:
    """Configuration manager with fallback to environment variables."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize configuration from file and environment."""
        self.config_path = config_path
        self.config_data: dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config_data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from file, then environment, then default.

        Supports nested keys using dot notation for the YAML file.
        Environment variables are always uppercase and use underscores.
        """
        # Try YAML first
        value = self._get_nested(key)
        if value is not None:
            return value

        # Try Environment variable
        env_key = key.replace(".", "_").upper()
        # Handle cases where the nesting doesn't match the old env var exactly if needed,
        # but here we follow the standard transformation.
        env_value = os.getenv(env_key)
        
        # Backward Compatibility: if "database.milvus_url" fails, try "MILVUS_URL"
        if env_value is None and "_" in env_key:
            # Try the last part if it matches old simple env vars
            env_value = os.getenv(env_key.split("_")[-1].upper())
            if env_value is None:
                # Try common patterns
                alt_keys = {
                    "DATABASE_MILVUS_URL": "MILVUS_URL",
                    "DATABASE_COLLECTION_NAME": "COLLECTION_NAME",
                    "MODELS_EMBEDDING_NAME": "EMBEDDING_MODEL",
                    "MODELS_EMBEDDING_DIMENSIONS": "EMBEDDING_DIMENSIONS",
                    "MODELS_LLM_NAME": "LLM_MODEL",
                }
                if env_key in alt_keys:
                    env_value = os.getenv(alt_keys[env_key])

        if env_value is not None:
            # Basic type conversion for env vars
            if env_value.lower() in ("true", "false"):
                return env_value.lower() == "true"
            try:
                if "." in env_value:
                    return float(env_value)
                return int(env_value)
            except ValueError:
                return env_value

        return default

    def _get_nested(self, key: str) -> Any | None:
        """Traverse nested dictionary for dot-notated key."""
        parts = key.split(".")
        data = self.config_data
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return None
        return data


# Global config instance
config = Config()
