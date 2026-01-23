# Configuration Migration Guide

## Overview
Wiki-RAG has been updated to support YAML configuration files (`config.yaml`) while maintaining backward compatibility with the legacy `.env` file format.

## New Configuration System

### Key Features
- **YAML-first configuration**: Primary method using `config.yaml` for better organization and readability
- **Backward compatibility**: Existing `.env` files continue to work
- **Migration utility**: `wr-config-update` command to convert `.env` to `config.yaml`
- **Type-safe access**: Helper methods for integers, floats, booleans, and lists
- **Nested configuration**: Support for dot-notation access to nested values

### Files Created/Modified

**New Files:**
- `wiki_rag/config/__init__.py` - Core configuration manager (`Config` class and `get_config()` function)
- `wiki_rag/config_update/main.py` - Utility to convert `.env` to `config.yaml`
- `wiki_rag/config_update/__init__.py` - Package init
- `config.yaml.template` - YAML configuration template

**Modified Files:**
- `wiki_rag/load/main.py` - Updated to use Config class
- `wiki_rag/index/main.py` - Updated to use Config class  
- `wiki_rag/search/main.py` - Updated to use Config class
- `wiki_rag/search/util.py` - Updated to use Config class
- `wiki_rag/server/main.py` - Updated to use Config class
- `wiki_rag/server/util.py` - Updated to use Config class
- `wiki_rag/mcp_server/main.py` - Updated to use Config class
- `wiki_rag/vector/milvus.py` - Updated to use Config class
- `pyproject.toml` - Added `wr-config-update` script and `pyyaml` dependency
- `.gitignore` - Added `config*.yaml` to ignore list
- `README.md` - Updated documentation
- `CHANGELOG.md` - Added changelog entry

## Usage

### Using config.yaml (Recommended)

```bash
# Copy the template
cp config.yaml.template config.yaml

# Edit config.yaml with your settings
# Then run any command normally
wr-load
wr-index
wr-search
```

### Using .env (Legacy)

```bash
# Existing .env files continue to work without changes
wr-load
# The system automatically falls back to .env if config.yaml is not found
```

### Migrating from .env to config.yaml

```bash
# Run the migration utility
wr-config-update

# This will:
# 1. Read values from .env
# 2. Convert them to config.yaml format
# 3. Create config.yaml
# 4. Keep .env for backup/removal
```

## Configuration Mappings

| .env Variable | config.yaml Path |
|---------------|------------------|
| `OPENAI_API_BASE` | `openai.api_base` |
| `OPENAI_API_KEY` | `openai.api_key` |
| `MEDIAWIKI_URL` | `mediawiki.url` |
| `MEDIAWIKI_NAMESPACES` | `mediawiki.namespaces` (array) |
| `MEDIAWIKI_EXCLUDED` | `mediawiki.excluded.categories`, `mediawiki.excluded.wikitext` |
| `MEDIAWIKI_KEEP_TEMPLATES` | `mediawiki.keep_templates` |
| `COLLECTION_NAME` | `collection.name` |
| `LOADER_DUMP_PATH` | `collection.dump_path` |
| `MILVUS_URL` | `milvus.url` |
| `EMBEDDING_MODEL` | `models.embedding` |
| `EMBEDDING_DIMENSIONS` | `models.embedding_dimensions` |
| `LLM_MODEL` | `models.llm` |
| `CONTEXTUALISATION_MODEL` | `models.contextualisation` |
| `WRAPPER_API_BASE` | `wrapper.api_base` |
| `WRAPPER_CHAT_MAX_TURNS` | `wrapper.chat_max_turns` |
| `WRAPPER_CHAT_MAX_TOKENS` | `wrapper.chat_max_tokens` |
| `WRAPPER_MODEL_NAME` | `wrapper.model_name` |
| `MCP_API_BASE` | `mcp.api_base` |
| `AUTH_TOKENS` | `auth.tokens` |
| `AUTH_URL` | `auth.url` |
| `LANGSMITH_*` | `langsmith.*` |
| `LANGFUSE_*` | `langfuse.*` |
| `USER_AGENT` | `crawler.user_agent` |
| `ENABLE_RATE_LIMITING` | `crawler.rate_limiting` |
| `INDEX_VENDOR` | `index_vendor` |

## Config Class API

```python
from wiki_rag.config import get_config

config = get_config()

# Get string value
value = config.get("mediawiki.url")

# Get integer value
timeout = config.get_int("models.embedding_dimensions", default=768)

# Get float value
temp = config.get_float(".temperature", default=0.5)

# Get boolean value
enabled = config.get_bool("crawler.rate_limiting", default=True)

# Get list value
namespaces = config.get_list("mediawiki.namespaces")

# Get integer list value
int_list = config.get_int_list("mediawiki.namespaces")
```

## Backward Compatibility

The system automatically:
1. Checks for `config.yaml` first
2. Falls back to `.env` if YAML not found
3. Logs a deprecation warning when using `.env`

This ensures existing installations continue to work without modification.

## Deprecation Timeline

1. **Current**: Both `.env` and `config.yaml` supported with `.env` warnings
2. **Future versions**: `.env` support will be maintained for backward compatibility
3. **Long-term**: Consider removing `.env` after a major version bump

## Troubleshooting

### "No config.yaml or .env file found"
- Ensure at least one configuration file exists
- Files must be in the project root directory

### "OPENAI_API_KEY is required"  
- This field must be set in `config.yaml` under `openai.api_key`
- Cannot be empty or omitted

### Configuration not loading
- Check that `config.yaml` has valid YAML syntax
- Ensure required fields are present
- Check the logs for specific error messages
