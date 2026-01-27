#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration update utility for converting dotenv to YAML format."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv


def load_dotenv_config(env_path: Path) -> Dict[str, Any]:
    """Load configuration from dotenv file."""
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    
    # Load the .env file
    load_dotenv(env_path)
    
    # Build configuration dictionary matching YAML structure
    config = {
        'openai': {
            'api_base': os.getenv('OPENAI_API_BASE', 'https://openai.com/v1'),
            'api_key': os.getenv('OPENAI_API_KEY', ''),
        },
        'mediawiki': {
            'url': os.getenv('MEDIAWIKI_URL', ''),
            'namespaces': _parse_comma_list(os.getenv('MEDIAWIKI_NAMESPACES', '0,4,12')),
            'excluded': os.getenv('MEDIAWIKI_EXCLUDED', ''),
            'keep_templates': _parse_comma_list(os.getenv('MEDIAWIKI_KEEP_TEMPLATES', '')),
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
            'tokens': _parse_comma_list(os.getenv('AUTH_TOKENS', '')),
            'url': os.getenv('AUTH_URL', ''),
        },
        'langsmith': {
            'tracing': _parse_bool(os.getenv('LANGSMITH_TRACING', 'false')),
            'prompts': _parse_bool(os.getenv('LANGSMITH_PROMPTS', 'false')),
            'prompt_prefix': os.getenv('LANGSMITH_PROMPT_PREFIX', ''),
            'endpoint': os.getenv('LANGSMITH_ENDPOINT', 'https://eu.api.smith.langchain.com'),
            'api_key': os.getenv('LANGSMITH_API_KEY', ''),
        },
        'langfuse': {
            'tracing': _parse_bool(os.getenv('LANGFUSE_TRACING', 'false')),
            'prompts': _parse_bool(os.getenv('LANGFUSE_PROMPTS', 'false')),
            'prompt_prefix': os.getenv('LANGFUSE_PROMPT_PREFIX', ''),
            'host': os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com'),
            'secret_key': os.getenv('LANGFUSE_SECRET_KEY', ''),
            'public_key': os.getenv('LANGFUSE_PUBLIC_KEY', ''),
        },
        'crawler': {
            'user_agent': os.getenv('USER_AGENT', 'Moodle Research Wiki-RAG Crawler'),
            'enable_rate_limiting': _parse_bool(os.getenv('ENABLE_RATE_LIMITING', 'true')),
        },
    }
    
    # Remove empty values and sections
    return _cleanup_config(config)


def _parse_comma_list(value: str) -> list:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ('true', '1', 'yes', 'on')


def _cleanup_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty sections and values from configuration."""
    cleaned = {}
    
    for section, content in config.items():
        if isinstance(content, dict):
            # Clean nested dictionaries
            cleaned_section = {}
            for key, value in content.items():
                if value or value == 0:  # Keep 0 values but remove empty strings/lists
                    cleaned_section[key] = value
            
            # Only add section if it has content
            if cleaned_section:
                cleaned[section] = cleaned_section
        elif content:  # For non-dict values
            cleaned[section] = content
    
    return cleaned


def save_yaml_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=True, indent=2)
        
        print(f"Configuration successfully saved to: {output_path}")
        
    except Exception as e:
        raise IOError(f"Failed to save YAML config to {output_path}: {e}")


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration for reference."""
    return {
        'openai': {
            'api_base': 'https://openai.com/v1',
            'api_key': '<<<YOUR_OPENAI_API_KEY>>>',
        },
        'mediawiki': {
            'url': 'https://your.mediawiki/base/url',
            'namespaces': [0, 4, 12],
            'excluded': 'categories:Category A, Category B;wikitext:Some regex to exclude',
            'keep_templates': ['Template1', 'Template2'],
        },
        'collection': {
            'name': 'your_collection_name',
            'dump_path': '',
        },
        'milvus': {
            'url': 'http://0.0.0.0:19530',
        },
        'models': {
            'embedding': 'your_embedding_model',
            'embedding_dimensions': 768,
            'llm': 'your_llm_model',
            'contextualisation': 'your_llm_model',
        },
        'wrapper': {
            'api_base': '0.0.0.0:8080',
            'chat_max_turns': 10,
            'chat_max_tokens': 1536,
            'model_name': 'Your Model Name',
        },
        'mcp': {
            'api_base': '0.0.0.0:8081',
        },
        'auth': {
            'tokens': ['11111111', '22222222', '33333333'],
            'url': 'http://0.0.0.0:4000/key/info',
        },
        'langsmith': {
            'tracing': False,
            'prompts': False,
            'prompt_prefix': 'your_prefix/',
            'endpoint': 'https://eu.api.smith.langchain.com',
            'api_key': '<<<YOUR_LANGSMITH_API_KEY_HERE>>>',
        },
        'langfuse': {
            'tracing': False,
            'prompts': False,
            'prompt_prefix': 'your_prefix-',
            'host': 'https://cloud.langfuse.com',
            'secret_key': '<<<YOUR_LANGFUSE_SECRET_KEY_HERE>>>',
            'public_key': '<<<YOUR_LANGFUSE_PUBLIC_KEY_HERE>>>',
        },
        'crawler': {
            'user_agent': 'Moodle Research Wiki-RAG Crawler',
            'enable_rate_limiting': True,
        },
    }


def main() -> None:
    """Main entry point for wr-config-update."""
    parser = argparse.ArgumentParser(
        description='Convert dotenv configuration to YAML format for wiki-rag'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('.env'),
        help='Input dotenv file (default: .env)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('config.yml'),
        help='Output YAML file (default: config.yml)'
    )
    
    parser.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Generate a sample config.yml file'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing output file'
    )
    
    args = parser.parse_args()
    
    try:
        if args.sample:
            config = create_sample_config()
            print("Creating sample configuration...")
        else:
            if not args.input.exists():
                print(f"Error: Input file not found: {args.input}")
                print("Use --sample to create a sample configuration.")
                sys.exit(1)
            
            print(f"Loading configuration from: {args.input}")
            config = load_dotenv_config(args.input)
        
        # Check if output file exists
        if args.output.exists() and not args.force:
            print(f"Error: Output file already exists: {args.output}")
            print("Use --force to overwrite or specify a different output file.")
            sys.exit(1)
        
        # Save to YAML
        save_yaml_config(config, args.output)
        
        # Provide migration guidance
        if not args.sample:
            print("\nMigration completed successfully!")
            print(f"1. Your new configuration is saved to: {args.output}")
            print("2. Update the configuration values as needed")
            print("3. The system will automatically use config.yml if present")
            print("4. You can safely remove the .env file once you've verified everything works")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()