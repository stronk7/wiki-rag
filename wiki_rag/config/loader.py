
"""Configuration loader and manager.

This module provides utilities for loading configuration from various sources
including YAML files, environment variables, and .env files, with proper
merging and validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import dotenv_values

from wiki_rag import ROOT_DIR
from wiki_rag.config.schema import ConfigSchema


logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager for wiki-rag.
    
    Handles loading configuration from multiple sources with proper
    precedence and validation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default path.
        """
        self.config_path = config_path or ROOT_DIR / "config.yaml"
        self.config = self._load_config()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary, or empty dict if file doesn't exist
        """
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing {self.config_path}: {e}")
            raise ValueError(f"Invalid YAML in {self.config_path}") from e
        except Exception as e:
            logger.error(f"Error reading {self.config_path}: {e}")
            raise
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from .env file (for backward compatibility).
        
        Returns:
            Configuration dictionary
        """
        dotenv_path = ROOT_DIR / ".env"
        if not dotenv_path.exists():
            return {}
        
        logger.warning(
            "Loading configuration from .env file. "
            "Please consider migrating to config.yaml. "
            "Use 'wr-config-update' to migrate."
        )
        
        env_dict = dotenv_values(dotenv_path)
        config = ConfigSchema.convert_env_to_config(env_dict)
        return config
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback chain.
        
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # 1. Load from config.yaml (if exists)
        yaml_config = self._load_yaml_config()
        config.update(yaml_config)
        
        # 2. Fallback to .env for backward compatibility
        env_config = self._load_env_config()
        config.update(env_config)
        
        # 3. Apply environment variable overrides
        env_overrides = ConfigSchema.convert_env_to_config(dict(os.environ))
        config.update(env_overrides)
        
        # 4. Apply defaults from schema
        schema_config = {}
        for section, fields in ConfigSchema.SCHEMA.items():
            schema_config[section] = {}
            for field, schema in fields.items():
                if "default" in schema:
                    schema_config[section][field] = schema["default"]
        
        # Update with schema defaults only for missing values
        for section, fields in schema_config.items():
            if section not in config:
                config[section] = {}
            for field, default_val in fields.items():
                if field not in config.get(section, {}):
                    config[section][field] = default_val if "schema" in locals() else default_val
        
        # 5. Validate against schema
        validated_config = ConfigSchema.validate_schema(config)
        
        return validated_config
    
    def get(self, path: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value by path.
        
        Args:
            path: Dot-separated path (e.g., "mediawiki.url")
            default: Default value if not found
            
        Returns:
            The configuration value, or default if not found
        """
        return ConfigSchema.get_nested_value(self.config, path) or default
    
    def save(self) -> None:
        """Save current configuration to YAML file.
        
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f, sort_keys=False)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def check_for_dotenv(self) -> bool:
        """Check if .env file exists and needs migration.
        
        Returns:
            True if .env exists without config.yaml
        """
        return (ROOT_DIR / ".env").exists() and not self.config_path.exists()
