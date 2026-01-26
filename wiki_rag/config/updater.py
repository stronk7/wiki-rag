
"""Configuration updater utility.

This module provides utilities for migrating from .env files
to config.yaml format, validating configurations, and generating
templates.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import dotenv_values

from wiki_rag import ROOT_DIR
from wiki_rag.config.loader import ConfigManager
from wiki_rag.config.schema import ConfigSchema


logger = logging.getLogger(__name__)


class ConfigUpdater:
    """Utility for updating and migrating configurations.
    
    Handles conversion from .env format to config.yaml format,
    validation, and configuration generation.
    """
    
    def __init__(self, dotenv_path: Optional[Path] = None, config_path: Optional[Path] = None):
        """Initialize configuration updater.
        
        Args:
            dotenv_path: Path to .env file. If None, uses default path.
            config_path: Path to config.yaml file. If None, uses default path.
        """
        self.dotenv_path = dotenv_path or ROOT_DIR / ".env"
        self.config_path = config_path or ROOT_DIR / "config.yaml"
    
    def convert_env_to_yaml(self) -> Dict[str, Any]:
        """Convert .env file to config.yaml format.
        
        Returns:
            Configuration dictionary in config.yaml format
            
        Raises:
            FileNotFoundError: If .env file doesn't exist
        """
        if not self.dotenv_path.exists():
            raise FileNotFoundError(f".env file not found at {self.dotenv_path}")
        
        # Load .env values
        env_dict = dotenv_values(self.dotenv_path)
        
        # Convert to config format
        config = ConfigSchema.convert_env_to_config(env_dict)
        
        # Apply schema defaults for missing fields
        for section, fields in ConfigSchema.SCHEMA.items():
            if section not in config:
                config[section] = {}
            for field, schema in fields.items():
                if "default" in schema and field not in config[section]:
                    config[section][field] = schema["default"]
        
        return config
    
    def migrate(self, overwrite: bool = False) -> None:
        """Migrate from .env to config.yaml format.
        
        Args:
            overwrite: Whether to overwrite existing config.yaml
            
        Returns:
            None
            
        Raises:
            FileExistsError: If config.yaml exists and overwrite=False
        """
        if self.config_path.exists() and not overwrite:
            raise FileExistsError(
                f"config.yaml already exists at {self.config_path}. "
                "Use --overwrite to replace it."
            )
        
        config = self.convert_env_to_yaml()
        
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)
            logger.info(f"Migrated configuration to {self.config_path}")
            
            # Validate the generated config
            validated = ConfigSchema.validate_schema(config)
            logger.info("Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Error migrating configuration: {e}")
            raise
    
    def show_diff(self) -> str:
        """Show differences between .env and config.yaml formats.
        
        Returns:
            String showing differences
        """
        env_config = ConfigSchema.convert_env_to_config(dotenv_values(self.dotenv_path))
        
        diff_lines = []
        diff_lines.append("Configuration Migration Preview:")
        diff_lines.append("=" * 50)
        
        for section in ConfigSchema.SCHEMA:
            diff_lines.append(f"\nSection: {section}")
            for field, schema in ConfigSchema.SCHEMA[section].items():
                value = ConfigSchema.get_nested_value(env_config, f"{section}.{field}")
                diff_lines.append(f"  {field}: {value}")
        
        diff_lines.append("\n" + "=" * 50)
        diff_lines.append("Note: Some values may be converted or parsed differently")
        
        return "\n".join(diff_lines)
    
    def validate_config(self, config_path: Optional[Path] = None) -> bool:
        """Validate a configuration file.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default.
            
        Returns:
            True if valid, False otherwise
        """
        path = config_path or self.config_path
        
        if not path.exists():
            logger.error(f"Config file not found: {path}")
            return False
        
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
            
            ConfigSchema.validate_schema(config)
            logger.info(f"Configuration at {path} is valid")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def generate_template(self) -> None:
        """Generate a template config.yaml file with example values."""
        try:
            template = {
                "mediawiki": {
                    "url": "https://en.wikipedia.org"
                },
                "index": {
                    "vendor": "milvus",
                    "host": "http://localhost:19530",
                    "collection_name": "wiki_docs"
                },
                "models": {
                    "embedding": "text-embedding-3-small",
                    "completion": "gpt-4o",
                    "rewrite": "gpt-4o"
                },
                "loader": {
                    "dump_path": "data",
                    "chunk_size": 512,
                    "chunk_overlap": 50
                },
                "server": {
                    "port": 8080
                }
            }
            
            template_path = ROOT_DIR / "config.example.yaml"
            with open(template_path, "w") as f:
                yaml.dump(template, f, sort_keys=False)
            
            logger.info(f"Template generated at {template_path}")
            
        except Exception as e:
            logger.error(f"Error generating template: {e}")
            raise
    
    def detect_needs_migration(self) -> bool:
        """Check if migration from .env to config.yaml is needed.
        
        Returns:
            True if .env exists without config.yaml, False otherwise
        """
        return self.dotenv_path.exists() and not self.config_path.exists()
