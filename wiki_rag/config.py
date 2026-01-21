#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration loader.

The project historically relied on environment variables (optionally loaded from a
`.env` file). We are moving towards a YAML-based configuration file while keeping
backwards compatibility for a transition period.

Precedence order:

1. OS environment variables
2. config.yaml (defaults)
3. .env (lowest priority)

Both config.yaml and .env are applied as *fill-missing-only* sources to preserve
that precedence.

The rest of the codebase continues to read configuration via `os.getenv(...)`.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

from wiki_rag import ROOT_DIR

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_FILES = [
    ROOT_DIR / "config.yaml",
    ROOT_DIR / "config.yml",
]


def _bool_to_env(v: bool) -> str:
    return "true" if v else "false"


def _to_env_str(value: Any) -> str:
    """Convert a Python value to an env-var string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_to_env(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    # Lists/dicts should have been flattened before reaching here.
    return str(value)


def _set_env_if_missing(key: str, value: Any) -> None:
    # Respect anything already present in the environment (highest precedence).
    if key in os.environ:
        return
    os.environ[key] = _to_env_str(value)


def _pick_config_path(config_path: str | Path | None) -> Path | None:
    """Return the config path to use, or None if not found."""
    if config_path is not None:
        return Path(config_path)

    env_path = os.getenv("WIKI_RAG_CONFIG")
    if env_path:
        return Path(env_path)

    for candidate in DEFAULT_CONFIG_FILES:
        if candidate.exists():
            return candidate

    return None


def apply_config_yaml_to_environ(config_path: str | Path | None = None) -> Path | None:
    """Load config.yaml and apply it to `os.environ` (fill-missing-only).

    Args:
        config_path: Optional explicit path. If None, tries WIKI_RAG_CONFIG and
            then defaults to ROOT_DIR/config.yaml or config.yml.

    Returns:
        The path that was attempted/used, or None if no config file was found.
    """
    chosen = _pick_config_path(config_path)
    if chosen is None:
        return None

    if not chosen.exists():
        logger.warning("Config file %s not found. Skipping.", chosen)
        return chosen

    try:
        data = yaml.safe_load(chosen.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read config file %s: %s", chosen, exc)
        raise

    if not isinstance(data, dict):
        raise ValueError(f"Config file {chosen} must contain a YAML mapping at the root")

    # Mapping between YAML structure and the legacy env vars.
    # NOTE: Keep this in sync with dotenv.template.
    mapping: list[tuple[str, str, str | None]] = [
        ("openai", "OPENAI_API_BASE", "api_base"),
        ("openai", "OPENAI_API_KEY", "api_key"),

        ("mediawiki", "MEDIAWIKI_URL", "url"),
        ("mediawiki", "MEDIAWIKI_NAMESPACES", "namespaces"),
        ("mediawiki", "MEDIAWIKI_EXCLUDED", "excluded"),
        ("mediawiki", "MEDIAWIKI_KEEP_TEMPLATES", "keep_templates"),
        ("mediawiki", "USER_AGENT", "user_agent"),
        ("mediawiki", "ENABLE_RATE_LIMITING", "enable_rate_limiting"),

        ("storage", "COLLECTION_NAME", "collection_name"),
        ("storage", "LOADER_DUMP_PATH", "loader_dump_path"),

        ("index", "INDEX_VENDOR", "vendor"),
        ("index", "MILVUS_URL", "milvus_url"),

        ("models", "EMBEDDING_MODEL", "embedding_model"),
        ("models", "EMBEDDING_DIMENSIONS", "embedding_dimensions"),
        ("models", "LLM_MODEL", "llm_model"),
        ("models", "CONTEXTUALISATION_MODEL", "contextualisation_model"),

        ("wrapper", "WRAPPER_API_BASE", "api_base"),
        ("wrapper", "WRAPPER_CHAT_MAX_TURNS", "chat_max_turns"),
        ("wrapper", "WRAPPER_CHAT_MAX_TOKENS", "chat_max_tokens"),
        ("wrapper", "WRAPPER_MODEL_NAME", "model_name"),

        ("mcp", "MCP_API_BASE", "api_base"),

        ("auth", "AUTH_TOKENS", "tokens"),
        ("auth", "AUTH_URL", "url"),

        ("langsmith", "LANGSMITH_TRACING", "tracing"),
        ("langsmith", "LANGSMITH_PROMPTS", "prompts"),
        ("langsmith", "LANGSMITH_PROMPT_PREFIX", "prompt_prefix"),
        ("langsmith", "LANGSMITH_ENDPOINT", "endpoint"),
        ("langsmith", "LANGSMITH_API_KEY", "api_key"),

        ("langfuse", "LANGFUSE_TRACING", "tracing"),
        ("langfuse", "LANGFUSE_PROMPTS", "prompts"),
        ("langfuse", "LANGFUSE_PROMPT_PREFIX", "prompt_prefix"),
        ("langfuse", "LANGFUSE_HOST", "host"),
        ("langfuse", "LANGFUSE_PUBLIC_KEY", "public_key"),
        ("langfuse", "LANGFUSE_SECRET_KEY", "secret_key"),
    ]

    def get_nested(root: dict[str, Any], section: str, key: str | None) -> Any:
        if key is None:
            return root.get(section)
        section_val = root.get(section)
        if not isinstance(section_val, dict):
            return None
        return section_val.get(key)

    for section, env_key, yaml_key in mapping:
        value = get_nested(data, section, yaml_key)
        if value is None:
            continue

        # Special conversions to match existing dotenv expectations.
        if env_key == "MEDIAWIKI_NAMESPACES":
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
        elif env_key == "MEDIAWIKI_KEEP_TEMPLATES":
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
        elif env_key == "AUTH_TOKENS":
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)

        _set_env_if_missing(env_key, value)

    return chosen


def apply_dotenv_to_environ(dotenv_path: str | Path | None = None) -> Path:
    """Load `.env` (fill-missing-only). Lowest precedence."""
    path = Path(dotenv_path) if dotenv_path else (ROOT_DIR / ".env")
    if path.exists():
        logger.warning("Loading environment variables from %s", path)
        logger.warning("Note: .env files are not supposed to be used in production. Use env secrets instead.")
        # Lowest priority: do not override anything already in the environment.
        load_dotenv(path, override=False)
    return path


def load_config(
    *,
    config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
) -> None:
    """Load configuration sources in precedence order.

    OS env is already present.
    Then config.yaml is applied as defaults.
    Then .env is applied as defaults.
    """
    apply_config_yaml_to_environ(config_path=config_path)
    apply_dotenv_to_environ(dotenv_path=dotenv_path)
