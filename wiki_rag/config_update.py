#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Utility to migrate .env configuration to config.yaml."""

import os

import yaml

from dotenv import dotenv_values


def main() -> None:
    """Migrate .env to config.yaml."""
    env_path = ".env"
    config_path = "config.yaml"

    if not os.path.exists(env_path):
        print(f"No {env_path} file found. Nothing to migrate.")
        return

    env_data = dotenv_values(env_path)
    
    # Define mapping of env vars to nested yaml keys
    mapping = {
        "OPENAI_API_BASE": "openai.api_base",
        "OPENAI_API_KEY": "openai.api_key",
        "MEDIAWIKI_URL": "mediawiki.url",
        "MEDIAWIKI_NAMESPACES": "mediawiki.namespaces",
        "MEDIAWIKI_EXCLUDED": "mediawiki.excluded",
        "MEDIAWIKI_KEEP_TEMPLATES": "mediawiki.keep_templates",
        "COLLECTION_NAME": "database.collection_name",
        "LOADER_DUMP_PATH": "database.loader_dump_path",
        "MILVUS_URL": "database.milvus_url",
        "EMBEDDING_MODEL": "models.embedding.name",
        "EMBEDDING_DIMENSIONS": "models.embedding.dimensions",
        "LLM_MODEL": "models.llm.name",
        "CONTEXTUALISATION_MODEL": "models.contextualisation.name",
        "WRAPPER_API_BASE": "wrapper.api_base",
        "WRAPPER_CHAT_MAX_TURNS": "wrapper.chat.max_turns",
        "WRAPPER_CHAT_MAX_TOKENS": "wrapper.chat.max_tokens",
        "WRAPPER_MODEL_NAME": "wrapper.model_name",
        "MCP_API_BASE": "mcp.api_base",
        "AUTH_TOKENS": "auth.tokens",
        "AUTH_URL": "auth.url",
        "LANGSMITH_TRACING": "tracing.langsmith.enabled",
        "LANGSMITH_PROMPTS": "tracing.langsmith.prompts",
        "LANGSMITH_PROMPT_PREFIX": "tracing.langsmith.prefix",
        "LANGSMITH_ENDPOINT": "tracing.langsmith.endpoint",
        "LANGSMITH_API_KEY": "tracing.langsmith.api_key",
        "LANGFUSE_TRACING": "tracing.langfuse.enabled",
        "LANGFUSE_PROMPTS": "tracing.langfuse.prompts",
        "LANGFUSE_PROMPT_PREFIX": "tracing.langfuse.prefix",
        "LANGFUSE_HOST": "tracing.langfuse.host",
        "LANGFUSE_SECRET_KEY": "tracing.langfuse.secret_key",
        "LANGFUSE_PUBLIC_KEY": "tracing.langfuse.public_key",
        "USER_AGENT": "crawler.user_agent",
        "ENABLE_RATE_LIMITING": "crawler.rate_limiting",
    }

    config_dict = {}

    for env_key, yaml_key in mapping.items():
        value = env_data.get(env_key)
        if value is not None:
            # Type conversion
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            
            # Nesting
            parts = yaml_key.split(".")
            d = config_dict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)

    print(f"Successfully migrated {env_path} to {config_path}")


if __name__ == "__main__":
    main()
