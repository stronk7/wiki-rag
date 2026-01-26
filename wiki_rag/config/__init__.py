"""Configuration module for wiki-rag.

This module provides configuration management including:
- YAML-based configuration
- Environment variable support
- .env file backward compatibility
- Configuration validation
- Migration utilities
"""

from wiki_rag.config.loader import ConfigManager
from wiki_rag.config.schema import ConfigSchema
from wiki_rag.config.updater import ConfigUpdater


__all__ = ["ConfigManager", "ConfigSchema", "ConfigUpdater"]
