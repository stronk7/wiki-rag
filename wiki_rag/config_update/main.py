#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the config update utility.

Converts .env file to config.yaml format.
"""

import logging
import os
import sys

from pathlib import Path

from dotenv import load_dotenv
import yaml

from wiki_rag import ROOT_DIR, __version__
from wiki_rag.util import setup_logging


def convert_env_to_yaml(env_path: Path, yaml_path: Path) -> None:
    """Convert .env file to config.yaml.

    Args:
        env_path: Path to the .env file
        yaml_path: Path where config.yaml will be created
    """
    load_dotenv(env_path)

    config: dict = {}

    config["openai"] = {
        "api_base": os.getenv("OPENAI_API_BASE", "https://openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    }

    config["mediawiki"] = {
        "url": os.getenv("MEDIAWIKI_URL", ""),
        "namespaces": parse_list(os.getenv("MEDIAWIKI_NAMESPACES", "0"), as_int=True),
        "excluded": {
            "categories": parse_list(os.getenv("MEDIAWIKI_EXCLUDED", "").split(";")[0].split(":")[1] if ":" in os.getenv("MEDIAWIKI_EXCLUDED", "") else ""),
            "wikitext": parse_list(os.getenv("MEDIAWIKI_EXCLUDED", "").split(";")[1].split(":")[1] if ";" in os.getenv("MEDIAWIKI_EXCLUDED", "") and ":" in os.getenv("MEDIAWIKI_EXCLUDED", "").split(";")[1] else ""),
        },
        "keep_templates": parse_list(os.getenv("MEDIAWIKI_KEEP_TEMPLATES", "")),
    }

    config["collection"] = {
        "name": os.getenv("COLLECTION_NAME", ""),
        "dump_path": os.getenv("LOADER_DUMP_PATH", ""),
    }

    config["milvus"] = {
        "url": os.getenv("MILVUS_URL", "http://0.0.0.0:19530"),
    }

    config["models"] = {
        "embedding": os.getenv("EMBEDDING_MODEL", ""),
        "embedding_dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
        "llm": os.getenv("LLM_MODEL", ""),
    }

    contextualisation = os.getenv("CONTEXTUALISATION_MODEL")
    if contextualisation:
        config["models"]["contextualisation"] = contextualisation

    config["wrapper"] = {
        "api_base": os.getenv("WRAPPER_API_BASE", "0.0.0.0:8080"),
        "chat_max_turns": int(os.getenv("WRAPPER_CHAT_MAX_TURNS", "0")),
        "chat_max_tokens": int(os.getenv("WRAPPER_CHAT_MAX_TOKENS", "0")),
        "model_name": os.getenv("WRAPPER_MODEL_NAME", ""),
    }

    config["mcp"] = {
        "api_base": os.getenv("MCP_API_BASE", "0.0.0.0:8081"),
    }

    config["auth"] = {
        "tokens": parse_list(os.getenv("AUTH_TOKENS", "")),
    }

    auth_url = os.getenv("AUTH_URL")
    if auth_url:
        config["auth"]["url"] = auth_url

    if os.getenv("LANGSMITH_TRACING", "false") == "true" or os.getenv("LANGSMITH_ENDPOINT"):
        config["langsmith"] = {
            "tracing": os.getenv("LANGSMITH_TRACING", "false") == "true",
            "prompts": os.getenv("LANGSMITH_PROMPTS", "false") == "true",
            "prompt_prefix": os.getenv("LANGSMITH_PROMPT_PREFIX", ""),
            "endpoint": os.getenv("LANGSMITH_ENDPOINT", ""),
            "api_key": os.getenv("LANGSMITH_API_KEY", ""),
        }

    if os.getenv("LANGFUSE_TRACING", "false") == "true" or os.getenv("LANGFUSE_HOST"):
        config["langfuse"] = {
            "tracing": os.getenv("LANGFUSE_TRACING", "false") == "true",
            "prompts": os.getenv("LANGFUSE_PROMPTS", "false") == "true",
            "prompt_prefix": os.getenv("LANGFUSE_PROMPT_PREFIX", ""),
            "host": os.getenv("LANGFUSE_HOST", ""),
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
        }

    config["crawler"] = {
        "user_agent": os.getenv("USER_AGENT", f"Moodle Research Wiki-RAG Crawler/{__version__} (https://github.com/moodlehq/wiki-rag)"),
        "rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
    }

    config["index_vendor"] = os.getenv("INDEX_VENDOR", "milvus")

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully converted {env_path} to {yaml_path}")
    print(f"\nPlease review {yaml_path} and ensure all values are correct.")
    print(f"After verification, you can remove the .env file if desired.")


def parse_list(value: str, as_int: bool = False) -> list:
    """Parse comma-separated list from string.

    Args:
        value: Comma-separated string
        as_int: Convert values to integers if True

    Returns:
        List of values
    """
    if not value:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    if as_int:
        return [int(item) for item in items]
    return items


def main() -> None:
    """Run the config update utility."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-config-update starting up...")
    logger.warning(f"Version: {__version__}")

    env_file = ROOT_DIR / ".env"
    yaml_file = ROOT_DIR / "config.yaml"

    if not env_file.exists():
        logger.error(f"No .env file found at {env_file}. Exiting.")
        sys.exit(1)

    if yaml_file.exists():
        logger.warning(f"config.yaml already exists at {yaml_file}.")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("Aborted.")
            sys.exit(0)

    try:
        convert_env_to_yaml(env_file, yaml_file)
        logger.info("wiki_rag-config-update finished.")
    except Exception as e:
        logger.error(f"Error converting .env to config.yaml: {e}. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()