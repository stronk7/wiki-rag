"""Configuration schema and validation for wiki-rag.

This module provides validation schemas for all configuration options
and utilities for loading and validating configuration files.
"""

from typing import Any, Dict, List, Mapping


class ConfigSchema:
    """Configuration schema definition and validation.
    
    This class defines the schema for all configuration options
    and provides validation methods.
    """
    
    # Main configuration schema structure
    SCHEMA = {
        "mediawiki": {
            "url": {
                "type": "string",
                "required": True,
                "description": "Base URL of MediaWiki instance"
            },
            "namespaces": {
                "type": "list",
                "schema": {"type": "int"},
                "default": [0],
                "description": "List of namespaces to include"
            },
            "excluded": {
                "type": "dict",
                "default": {},
                "description": "Exclusions by type (categories, wikitext, etc.)"
            },
            "keep_templates": {
                "type": "list",
                "schema": {"type": "string"},
                "default": [],
                "description": "Templates to keep in wiki text"
            }
        },
        "index": {
            "vendor": {
                "type": "string",
                "default": "milvus",
                "description": "Vector database vendor"
            },
            "host": {
                "type": "string",
                "default": "http://localhost:19530",
                "description": "Vector database host URL"
            },
            "collection_name": {
                "type": "string",
                "required": True,
                "description": "Name of the vector collection"
            },
            "embedding_dimension": {
                "type": "int",
                "default": 768,
                "description": "Dimensionality of embeddings"
            }
        },
        "models": {
            "embedding": {
                "type": "string",
                "default": "text-embedding-3-small",
                "description": "Embedding model to use"
            },
            "completion": {
                "type": "string",
                "default": "gpt-4o",
                "description": "Completion model for responses"
            },
            "rewrite": {
                "type": "string",
                "default": "gpt-4o",
                "description": "Model for query rewrite"
            }
        },
        "loader": {
            "dump_path": {
                "type": "string",
                "default": "data",
                "description": "Path to store JSON dump"
            },
            "chunk_size": {
                "type": "int",
                "default": 512,
                "description": "Chunk size for document splitting"
            },
            "chunk_overlap": {
                "type": "int",
                "default": 50,
                "description": "Overlap between chunks"
            }
        },
        "server": {
            "port": {
                "type": "int",
                "default": 8080,
                "description": "Server port"
            },
            "timeout": {
                "type": "int",
                "default": 20,
                "description": "Request timeout in seconds"
            },
            "stream_timeout": {
                "type": "int",
                "default": 60,
                "description": "Streaming response timeout"
            }
        }
    }
    
    # Mapping from environment variables to config paths
    ENV_TO_CONFIG = {
        "MEDIAWIKI_URL": ("mediawiki", "url"),
        "MEDIAWIKI_NAMESPACES": ("mediawiki", "namespaces"),
        "MEDIAWIKI_EXCLUDED": ("mediawiki", "excluded"),
        "MEDIAWIKI_KEEP_TEMPLATES": ("mediawiki", "keep_templates"),
        "INDEX_VENDOR": ("index", "vendor"),
        "MILVUS_URL": ("index", "host"),
        "COLLECTION_NAME": ("index", "collection_name"),
        "EMBEDDING_DIMENSIONS": ("index", "embedding_dimension"),
        "EMBEDDING_MODEL": ("models", "embedding"),
        "GENERATE_MODEL": ("models", "completion"),
        "REWRITE_MODEL": ("models", "rewrite"),
        "LOADER_DUMP_PATH": ("loader", "dump_path"),
        "CHUNK_SIZE": ("loader", "chunk_size"),
        "CHUNK_OVERLAP": ("loader", "chunk_overlap"),
    }
    
    @classmethod
    def get_nested_value(cls, config: Dict[str, Any], path: str) -> Any:
        """Get a nested value from a config dictionary.
        
        Args:
            config: The config dictionary
            path: Dot-separated path (e.g., "mediawiki.url")
            
        Returns:
            The nested value, or None if not found
        """
        keys = path.split(".")
        value = config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    @classmethod
    def set_nested_value(cls, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in a config dictionary.
        
        Args:
            config: The config dictionary
            path: Dot-separated path (e.g., "mediawiki.url")
            value: The value to set
        """
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    @classmethod
    def validate_type(cls, value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected type.
        
        Args:
            value: The value to validate
            expected_type: One of "string", "int", "list", "dict"
            
        Returns:
            True if valid, False otherwise
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "int":
            return isinstance(value, int)
        elif expected_type == "list":
            return isinstance(value, list)
        elif expected_type == "dict":
            return isinstance(value, dict)
        return False
    
    @classmethod
    def validate_schema(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration with defaults applied
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        for section, fields in cls.SCHEMA.items():
            validated[section] = {}
            
            for field, schema in fields.items():
                # Get value from config or use default
                value = config.get(section, {}).get(field)
                
                if value is None:
                    if schema.get("required", False):
                        raise ValueError(f"Missing required field: {section}.{field}")
                    if "default" in schema:
                        value = schema["default"]
                    else:
                        continue
                
                # Validate type
                if "type" in schema and not cls.validate_type(value, schema["type"]):
                    raise ValueError(
                        f"Invalid type for {section}.{field}: "
                        f"expected {schema['type']}, got {type(value).__name__}"
                    )
                
                validated[section][field] = value
        
        return validated
    
    @classmethod
    def convert_env_to_config(cls, env_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert environment variables to config format.
        
        Args:
            env_dict: Dictionary of environment variables
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        for env_var, config_path_tuple in cls.ENV_TO_CONFIG.items():
            if env_var in env_dict:
                value = env_dict[env_var]
                
                # Parse special formats
                if env_var.endswith("_NAMESPACES"):
                    # Convert comma-separated string to list of ints
                    value = [int(ns.strip()) for ns in value.split(",")]
                elif env_var.endswith("_EXCLUDED"):
                    # Parse exclusion format: "categories:val1,val2;wikitext:val3"
                    exclusions = {}
                    for exclusion in value.split(";"):
                        if ":" in exclusion:
                            ex_type, ex_values = exclusion.split(":", 1)
                            ex_values = [v.strip() for v in ex_values.split(",")]
                            exclusions[ex_type.strip()] = ex_values
                    value = exclusions
                elif env_var.endswith("_KEEP_TEMPLATES"):
                    # Convert comma-separated string to list
                    value = [t.strip() for t in value.split(",")]
                
                # Convert tuple to path string
                config_path = ".".join(config_path_tuple)
                cls.set_nested_value(config, config_path, value)
        
        return config
        
        return config
