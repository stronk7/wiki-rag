#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration management for wiki-rag.

This module handles loading configuration from YAML files with backward
compatibility support for dotenv files. It provides a unified interface
for accessing configuration values.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration loading fails."""


class Config:
    """Configuration manager with YAML and dotenv support."""
    
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize configuration from YAML or fall back to dotenv.
        
        Args:
            config_path: Path to config.yml file. If None, searches in:
                1. ./config.yml
                2. ~/.config/wiki-rag/config.yml
                3. Falls back to dotenv if no YAML found
        """
        self.config_path = config_path or self._find_config_file()
        self.config_data: Dict[str, Any] = {}
        self.using_yaml = False
        
        if self.config_path and self.config_path.exists():
            self._load_yaml()
            self.using_yaml = True
        else:
            self._load_dotenv()
            self.using_yaml = False
    
    def _find_config_file(self) -> Optional[Path]:
        """Find config.yml in standard locations."""
        search_paths = [
            Path.cwd() / "config.yml",
            Path.home() / ".config" / "wiki-rag" / "config.yml",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        return None
    
    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path:
            raise ConfigError("No config file path provided")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to load YAML config from {self.config_path}: {e}")
    
    def _load_dotenv(self) -> None:
        """Load configuration from dotenv file for backward compatibility."""
        # Load .env file if it exists
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Try to load from parent directory as well
            parent_env = Path.cwd().parent / ".env"
            if parent_env.exists():
                load_dotenv(parent_env)
        
        # Build config data from environment variables
        self.config_data = self._build_config_from_env()
    
    def _build_config_from_env(self) -> Dict[str, Any]:
        """Build configuration dictionary from environment variables."""
        return {
            'openai': {
                'api_base': os.getenv('OPENAI_API_BASE', 'https://openai.com/v1'),
                'api_key': os.getenv('OPENAI_API_KEY', ''),
            },
            'mediawiki': {
                'url': os.getenv('MEDIAWIKI_URL', ''),
                'namespaces': self._parse_list(os.getenv('MEDIAWIKI_NAMESPACES', '0,4,12')),
                'excluded': os.getenv('MEDIAWIKI_EXCLUDED', ''),
                'keep_templates': self._parse_list(os.getenv('MEDIAWIKI_KEEP_TEMPLATES', '')),
            },
            'collection': {
                'name': os.getenv('COLLECTION_NAME', ''),
                'dump_path': os.getenv('LOADER_DUMP_PATH', ''),
            },
            'milvus': {
                'url': os.getenv('MILVUS_URL', 'http://0.0.0.0:19530'),
            },
            'models': {
                'embedding': os.getenv('EMBEDDING_MODEL', ''),
                'embedding_dimensions': int(os.getenv('EMBEDDING_DIMENSIONS', '768')),
                'llm': os.getenv('LLM_MODEL', ''),
                'contextualisation': os.getenv('CONTEXTUALISATION_MODEL', ''),
            },
            'wrapper': {
                'api_base': os.getenv('WRAPPER_API_BASE', '0.0.0.0:8080'),
                'chat_max_turns': int(os.getenv('WRAPPER_CHAT_MAX_TURNS', '10')),
                'chat_max_tokens': int(os.getenv('WRAPPER_CHAT_MAX_TOKENS', '1536')),
                'model_name': os.getenv('WRAPPER_MODEL_NAME', ''),
            },
            'mcp': {
                'api_base': os.getenv('MCP_API_BASE', '0.0.0.0:8081'),
            },
            'auth': {
                'tokens': self._parse_list(os.getenv('AUTH_TOKENS', '')),
                'url': os.getenv('AUTH_URL', ''),
            },
            'langsmith': {
                'tracing': os.getenv('LANGSMITH_TRACING', 'false').lower() == 'true',
                'prompts': os.getenv('LANGSMITH_PROMPTS', 'false').lower() == 'true',
                'prompt_prefix': os.getenv('LANGSMITH_PROMPT_PREFIX', ''),
                'endpoint': os.getenv('LANGSMITH_ENDPOINT', 'https://eu.api.smith.langchain.com'),
                'api_key': os.getenv('LANGSMITH_API_KEY', ''),
            },
            'langfuse': {
                'tracing': os.getenv('LANGFUSE_TRACING', 'false').lower() == 'true',
                'prompts': os.getenv('LANGFUSE_PROMPTS', 'false').lower() == 'true',
                'prompt_prefix': os.getenv('LANGFUSE_PROMPT_PREFIX', ''),
                'host': os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com'),
                'secret_key': os.getenv('LANGFUSE_SECRET_KEY', ''),
                'public_key': os.getenv('LANGFUSE_PUBLIC_KEY', ''),
            },
            'crawler': {
                'user_agent': os.getenv('USER_AGENT', 'Moodle Research Wiki-RAG Crawler'),
                'enable_rate_limiting': os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            },
        }
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated string into list."""
        if not value:
            return []
        return [item.strip() for item in value.split(',')]
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'openai.api_key')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config_data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def is_using_yaml(self) -> bool:
        """Check if configuration is loaded from YAML."""
        return self.using_yaml
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary."""
        return self.config_data.copy()


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get the global configuration instance.
    
    Args:
        config_path: Optional path to config.yml file (str or Path)
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        if isinstance(config_path, str):
            config_path = Path(config_path)
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> Config:
    """Reload the global configuration.
    
    Args:
        config_path: Optional path to config.yml file
        
    Returns:
        New configuration instance
    """
    global _config
    _config = Config(config_path)
    return _config