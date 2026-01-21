#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Utility to convert a dotenv file into a config.yaml file.

This is meant to help during the transition from `.env`-based configuration to
YAML configuration.

It reads a dotenv file, parses common values into structured YAML, and writes a
config.yaml that can be used by wiki-rag.

NOTE: Secrets may be present in the dotenv file. Use --redact-secrets to avoid
writing credentials to disk.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import yaml

from dotenv import dotenv_values


SECRET_KEYS = {
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
}


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _env_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def env_to_config(env: dict[str, str | None], *, redact_secrets: bool) -> dict:
    def get(key: str):
        v = env.get(key)
        if v is None:
            return None
        if redact_secrets and key in SECRET_KEYS and v.strip() != "":
            return "<<<REDACTED>>>"
        return v

    cfg: dict = {
        "openai": {
            "api_base": get("OPENAI_API_BASE"),
            "api_key": get("OPENAI_API_KEY"),
        },
        "mediawiki": {
            "url": get("MEDIAWIKI_URL"),
            "namespaces": None,
            "excluded": get("MEDIAWIKI_EXCLUDED"),
            "keep_templates": None,
            "user_agent": get("USER_AGENT"),
            "enable_rate_limiting": None,
        },
        "storage": {
            "collection_name": get("COLLECTION_NAME"),
            "loader_dump_path": get("LOADER_DUMP_PATH"),
        },
        "index": {
            "vendor": get("INDEX_VENDOR"),
            "milvus_url": get("MILVUS_URL"),
        },
        "models": {
            "embedding_model": get("EMBEDDING_MODEL"),
            "embedding_dimensions": None,
            "llm_model": get("LLM_MODEL"),
            "contextualisation_model": get("CONTEXTUALISATION_MODEL"),
        },
        "wrapper": {
            "api_base": get("WRAPPER_API_BASE"),
            "chat_max_turns": None,
            "chat_max_tokens": None,
            "model_name": get("WRAPPER_MODEL_NAME"),
        },
        "mcp": {
            "api_base": get("MCP_API_BASE"),
        },
        "auth": {
            "tokens": None,
            "url": get("AUTH_URL"),
        },
        "langsmith": {
            "tracing": None,
            "prompts": None,
            "prompt_prefix": get("LANGSMITH_PROMPT_PREFIX"),
            "endpoint": get("LANGSMITH_ENDPOINT"),
            "api_key": get("LANGSMITH_API_KEY"),
        },
        "langfuse": {
            "tracing": None,
            "prompts": None,
            "prompt_prefix": get("LANGFUSE_PROMPT_PREFIX"),
            "host": get("LANGFUSE_HOST"),
            "public_key": get("LANGFUSE_PUBLIC_KEY"),
            "secret_key": get("LANGFUSE_SECRET_KEY"),
        },
    }

    # Parse / coerce types.
    namespaces = get("MEDIAWIKI_NAMESPACES")
    if namespaces:
        cfg["mediawiki"]["namespaces"] = [int(v) for v in _split_csv(namespaces)]

    keep_templates = get("MEDIAWIKI_KEEP_TEMPLATES")
    if keep_templates:
        cfg["mediawiki"]["keep_templates"] = _split_csv(keep_templates)

    enable_rate_limiting = get("ENABLE_RATE_LIMITING")
    if enable_rate_limiting is not None:
        cfg["mediawiki"]["enable_rate_limiting"] = _env_bool(enable_rate_limiting)

    embed_dims = get("EMBEDDING_DIMENSIONS")
    if embed_dims:
        cfg["models"]["embedding_dimensions"] = int(embed_dims)

    chat_max_turns = get("WRAPPER_CHAT_MAX_TURNS")
    if chat_max_turns:
        cfg["wrapper"]["chat_max_turns"] = int(chat_max_turns)

    chat_max_tokens = get("WRAPPER_CHAT_MAX_TOKENS")
    if chat_max_tokens:
        cfg["wrapper"]["chat_max_tokens"] = int(chat_max_tokens)

    auth_tokens = get("AUTH_TOKENS")
    if auth_tokens:
        cfg["auth"]["tokens"] = _split_csv(auth_tokens)

    langsmith_tracing = get("LANGSMITH_TRACING")
    if langsmith_tracing is not None:
        cfg["langsmith"]["tracing"] = _env_bool(langsmith_tracing)

    langsmith_prompts = get("LANGSMITH_PROMPTS")
    if langsmith_prompts is not None:
        cfg["langsmith"]["prompts"] = _env_bool(langsmith_prompts)

    langfuse_tracing = get("LANGFUSE_TRACING")
    if langfuse_tracing is not None:
        cfg["langfuse"]["tracing"] = _env_bool(langfuse_tracing)

    langfuse_prompts = get("LANGFUSE_PROMPTS")
    if langfuse_prompts is not None:
        cfg["langfuse"]["prompts"] = _env_bool(langfuse_prompts)

    # Drop empty subtrees to keep file tidy.
    def prune(obj):
        if isinstance(obj, dict):
            new = {k: prune(v) for k, v in obj.items()}
            return {k: v for k, v in new.items() if v not in (None, "", [], {})}
        if isinstance(obj, list):
            return [prune(v) for v in obj]
        return obj

    return prune(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a dotenv file into a config.yaml file")
    parser.add_argument("--env", dest="env_path", default=".env", help="Path to dotenv file (default: .env)")
    parser.add_argument("--out", dest="out_path", default="config.yaml", help="Output YAML path")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    parser.add_argument("--redact-secrets", action="store_true", help="Redact known secret keys")
    parser.add_argument("--print", dest="print_only", action="store_true", help="Print YAML to stdout")

    args = parser.parse_args()

    env_path = Path(args.env_path)
    if not env_path.exists():
        print(f"dotenv file not found: {env_path}", file=sys.stderr)
        raise SystemExit(2)

    out_path = Path(args.out_path)
    if out_path.exists() and not args.force and not args.print_only:
        print(f"Refusing to overwrite existing file: {out_path} (use --force)", file=sys.stderr)
        raise SystemExit(2)

    env = dotenv_values(env_path)
    cfg = env_to_config(env, redact_secrets=args.redact_secrets)

    yaml_text = yaml.safe_dump(cfg, sort_keys=False)

    if args.print_only:
        print(yaml_text)
        return

    out_path.write_text(yaml_text)


if __name__ == "__main__":
    main()
