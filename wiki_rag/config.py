#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration loading utilities.

BC goals:
- Support loading legacy `.env` files (if present).
- Prefer `config.yaml` when present.
- Always let OS environment variables win.

Important: secrets must be provided via OS environment variables.
During the BC period the codebase will still *read* secrets from `.env` if users
put them there, but `wr-config-update` will never write secrets into YAML.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import dotenv_values, load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    dotenv_values = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from wiki_rag import ROOT_DIR


CONFIG_FILE_NAME = "config.yaml"


@dataclass(frozen=True)
class ConfigLoadResult:
    """Details about how configuration was loaded."""

    config_path: Path | None
    loaded_from_yaml: bool
    loaded_dotenv: bool


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config_to_env(
    *,
    root_dir: Path = ROOT_DIR,
    config_path: Path | None = None,
    dotenv_path: Path | None = None,
) -> ConfigLoadResult:
    """Load configuration into process environment.

    Precedence (BC): OS env > config.yaml > .env.

    This function mutates `os.environ` only for keys that are currently unset.

    Args:
        root_dir: Project root dir.
        config_path: Optional explicit path to YAML config.
        dotenv_path: Optional explicit path to `.env`.

    Returns:
        ConfigLoadResult describing what was loaded.
    """

    resolved_config_path = config_path or (root_dir / CONFIG_FILE_NAME)
    resolved_dotenv_path = dotenv_path or (root_dir / ".env")

    loaded_from_yaml = False
    loaded_dotenv = False

    # 1) YAML -> env (only if env key not set).
    if resolved_config_path.exists():
        if yaml is None:
            raise RuntimeError(
                "Missing dependency 'pyyaml'. Install wiki-rag with dependencies to use config.yaml."
            )
        data = _read_yaml_config(resolved_config_path)
        for key, value in _flatten_to_env(data).items():
            if key not in os.environ and value is not None:
                os.environ[key] = str(value)
        loaded_from_yaml = True

    # 2) .env -> env (only if env key not set).
    if resolved_dotenv_path.exists():
        if load_dotenv is None:
            raise RuntimeError(
                "Missing dependency 'python-dotenv'. Install wiki-rag with dependencies to use .env files."
            )
        # Use python-dotenv loader because it supports common .env format and
        # does not overwrite by default.
        load_dotenv(resolved_dotenv_path, override=False)
        loaded_dotenv = True

    return ConfigLoadResult(
        config_path=resolved_config_path if loaded_from_yaml else None,
        loaded_from_yaml=loaded_from_yaml,
        loaded_dotenv=loaded_dotenv,
    )


def _read_yaml_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        msg = f"Config file {path} must contain a YAML mapping at the root"
        raise ValueError(msg)
    return data


def _flatten_to_env(config: dict[str, Any]) -> dict[str, Any]:
    """Convert structured YAML config into env-style keys."""

    # The schema is intentionally permissive. Unknown keys are ignored.
    env: dict[str, Any] = {}

    openai = config.get("openai", {}) or {}
    if isinstance(openai, dict):
        _maybe_set(env, "OPENAI_API_BASE", openai.get("api_base"))

    mediawiki = config.get("mediawiki", {}) or {}
    if isinstance(mediawiki, dict):
        _maybe_set(env, "MEDIAWIKI_URL", mediawiki.get("url"))
        namespaces = mediawiki.get("namespaces")
        if isinstance(namespaces, list):
            _maybe_set(env, "MEDIAWIKI_NAMESPACES", ",".join(str(v) for v in namespaces))
        elif namespaces is not None:
            _maybe_set(env, "MEDIAWIKI_NAMESPACES", namespaces)
        _maybe_set(env, "MEDIAWIKI_EXCLUDED", mediawiki.get("excluded"))
        keep_templates = mediawiki.get("keep_templates")
        if isinstance(keep_templates, list):
            _maybe_set(env, "MEDIAWIKI_KEEP_TEMPLATES", ",".join(str(v) for v in keep_templates))
        elif keep_templates is not None:
            _maybe_set(env, "MEDIAWIKI_KEEP_TEMPLATES", keep_templates)
        _maybe_set(env, "USER_AGENT", mediawiki.get("user_agent"))
        if "enable_rate_limiting" in mediawiki:
            _maybe_set(env, "ENABLE_RATE_LIMITING", "true" if _env_bool(str(mediawiki.get("enable_rate_limiting")), True) else "false")

    storage = config.get("storage", {}) or {}
    if isinstance(storage, dict):
        _maybe_set(env, "COLLECTION_NAME", storage.get("collection_name"))
        _maybe_set(env, "LOADER_DUMP_PATH", storage.get("loader_dump_path"))
        _maybe_set(env, "INDEX_VENDOR", storage.get("index_vendor"))
        milvus = storage.get("milvus", {}) or {}
        if isinstance(milvus, dict):
            _maybe_set(env, "MILVUS_URL", milvus.get("url"))

    models = config.get("models", {}) or {}
    if isinstance(models, dict):
        _maybe_set(env, "EMBEDDING_MODEL", models.get("embedding_model"))
        _maybe_set(env, "EMBEDDING_DIMENSIONS", models.get("embedding_dimensions"))
        _maybe_set(env, "LLM_MODEL", models.get("llm_model"))
        _maybe_set(env, "CONTEXTUALISATION_MODEL", models.get("contextualisation_model"))

    wrapper = config.get("wrapper", {}) or {}
    if isinstance(wrapper, dict):
        _maybe_set(env, "WRAPPER_API_BASE", wrapper.get("api_base"))
        _maybe_set(env, "WRAPPER_CHAT_MAX_TURNS", wrapper.get("chat_max_turns"))
        _maybe_set(env, "WRAPPER_CHAT_MAX_TOKENS", wrapper.get("chat_max_tokens"))
        _maybe_set(env, "WRAPPER_MODEL_NAME", wrapper.get("model_name"))

    mcp = config.get("mcp", {}) or {}
    if isinstance(mcp, dict):
        _maybe_set(env, "MCP_API_BASE", mcp.get("api_base"))

    auth = config.get("auth", {}) or {}
    if isinstance(auth, dict):
        _maybe_set(env, "AUTH_TOKENS", auth.get("tokens"))
        _maybe_set(env, "AUTH_URL", auth.get("url"))

    prompts = config.get("prompts", {}) or {}
    if isinstance(prompts, dict):
        _maybe_set(env, "LANGSMITH_PROMPTS", _bool_to_env(prompts.get("langsmith_enabled")))
        _maybe_set(env, "LANGSMITH_PROMPT_PREFIX", prompts.get("langsmith_prefix"))
        _maybe_set(env, "LANGFUSE_PROMPTS", _bool_to_env(prompts.get("langfuse_enabled")))
        _maybe_set(env, "LANGFUSE_PROMPT_PREFIX", prompts.get("langfuse_prefix"))

    observability = config.get("observability", {}) or {}
    if isinstance(observability, dict):
        _maybe_set(env, "LANGSMITH_TRACING", _bool_to_env(observability.get("langsmith_tracing")))
        _maybe_set(env, "LANGSMITH_ENDPOINT", observability.get("langsmith_endpoint"))
        _maybe_set(env, "LANGFUSE_TRACING", _bool_to_env(observability.get("langfuse_tracing")))
        _maybe_set(env, "LANGFUSE_HOST", observability.get("langfuse_host"))

    return env


def _bool_to_env(value: Any) -> str | None:
    if value is None:
        return None
    return "true" if bool(value) else "false"


def _maybe_set(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    target[key] = value


_SECRET_ENV_KEYS: set[str] = {
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
}


def generate_config_yaml_from_dotenv(
    dotenv_path: Path,
) -> dict[str, Any]:
    """Convert dotenv values into structured config.yaml content.

    Secrets are not exported.
    """

    if dotenv_values is None:
        raise RuntimeError(
            "Missing dependency 'python-dotenv'. Install wiki-rag with dependencies to read .env files."
        )

    values = dotenv_values(dotenv_path)

    def get(key: str) -> str | None:
        if key in _SECRET_ENV_KEYS:
            return None
        val = values.get(key)
        if val is None:
            return None
        return str(val).strip().strip('"').strip("'")

    mediawiki_namespaces = get("MEDIAWIKI_NAMESPACES")
    namespaces_list: list[int] | None = None
    if mediawiki_namespaces:
        namespaces_list = [int(part.strip()) for part in mediawiki_namespaces.split(",") if part.strip()]

    keep_templates = get("MEDIAWIKI_KEEP_TEMPLATES")
    keep_templates_list: list[str] | None = None
    if keep_templates:
        keep_templates_list = [part.strip() for part in keep_templates.split(",") if part.strip()]

    config: dict[str, Any] = {
        "openai": {
            "api_base": get("OPENAI_API_BASE"),
        },
        "mediawiki": {
            "url": get("MEDIAWIKI_URL"),
            "namespaces": namespaces_list,
            "excluded": get("MEDIAWIKI_EXCLUDED"),
            "keep_templates": keep_templates_list,
            "user_agent": get("USER_AGENT"),
            "enable_rate_limiting": _env_bool(get("ENABLE_RATE_LIMITING"), True),
        },
        "storage": {
            "collection_name": get("COLLECTION_NAME"),
            "loader_dump_path": get("LOADER_DUMP_PATH"),
            "index_vendor": get("INDEX_VENDOR"),
            "milvus": {
                "url": get("MILVUS_URL"),
            },
        },
        "models": {
            "embedding_model": get("EMBEDDING_MODEL"),
            "embedding_dimensions": _try_int(get("EMBEDDING_DIMENSIONS")),
            "llm_model": get("LLM_MODEL"),
            "contextualisation_model": get("CONTEXTUALISATION_MODEL"),
        },
        "wrapper": {
            "api_base": get("WRAPPER_API_BASE"),
            "chat_max_turns": _try_int(get("WRAPPER_CHAT_MAX_TURNS")),
            "chat_max_tokens": _try_int(get("WRAPPER_CHAT_MAX_TOKENS")),
            "model_name": get("WRAPPER_MODEL_NAME") or get("COLLECTION_NAME"),
        },
        "mcp": {
            "api_base": get("MCP_API_BASE"),
        },
        "auth": {
            "tokens": get("AUTH_TOKENS"),
            "url": get("AUTH_URL"),
        },
        "observability": {
            "langsmith_tracing": _env_bool(values.get("LANGSMITH_TRACING"), False),
            "langsmith_endpoint": get("LANGSMITH_ENDPOINT"),
            "langfuse_tracing": _env_bool(values.get("LANGFUSE_TRACING"), False),
            "langfuse_host": get("LANGFUSE_HOST"),
        },
        "prompts": {
            "langsmith_enabled": _env_bool(values.get("LANGSMITH_PROMPTS"), False),
            "langsmith_prefix": get("LANGSMITH_PROMPT_PREFIX"),
            "langfuse_enabled": _env_bool(values.get("LANGFUSE_PROMPTS"), False),
            "langfuse_prefix": get("LANGFUSE_PROMPT_PREFIX"),
        },
    }

    # Remove Nones and empty dicts/lists for cleanliness.
    return _prune_empty(config)


def dump_config_yaml(data: dict[str, Any]) -> str:
    """Serialize config dict into YAML."""

    if yaml is None:
        raise RuntimeError(
            "Missing dependency 'pyyaml'. Install wiki-rag with dependencies to write config.yaml."
        )

    return yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def _try_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {k: _prune_empty(v) for k, v in value.items()}
        pruned = {k: v for k, v in pruned.items() if v is not None}
        # remove empty dicts
        return {k: v for k, v in pruned.items() if v != {} and v != []}
    if isinstance(value, list):
        pruned_list = [_prune_empty(v) for v in value]
        pruned_list = [v for v in pruned_list if v is not None]
        return pruned_list if pruned_list else None
    return value
