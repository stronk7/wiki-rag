#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the knowledge base OpenAI compatible server."""

import logging
import os
import sys

from pathlib import Path

import uvicorn

from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

import wiki_rag.vector as vector

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__, server
from wiki_rag.config import settings
from wiki_rag.search.util import ContextSchema, build_graph
from wiki_rag.util import setup_logging
from wiki_rag.vector import load_vector_store


def main():
    """Run the OpenAI server with all the configuration in place."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-server starting up...")

    # Print the version of the bot.
    logger.warning(f"Version: {__version__}")

    mediawiki_url = settings.get_str("MEDIAWIKI_URL")
    if not mediawiki_url:
        logger.error("Mediawiki URL not found in configuration. Exiting.")
        sys.exit(1)

    mediawiki_namespaces = settings.get_list("MEDIAWIKI_NAMESPACES")
    if not mediawiki_namespaces:
        logger.error("Mediawiki namespaces not found in configuration. Exiting.")
        sys.exit(1)
    mediawiki_namespaces = [int(ns) for ns in mediawiki_namespaces]  # no whitespace and int.
    mediawiki_namespaces = list(set(mediawiki_namespaces))  # unique

    loader_dump_path = settings.get_str("LOADER_DUMP_PATH")
    if loader_dump_path:
        loader_dump_path = Path(loader_dump_path)
    else:
        loader_dump_path = ROOT_DIR / "data"
    # If the directory does not exist, create it.
    if not loader_dump_path.exists():
        logger.warning(f"Data directory {loader_dump_path} not found. Creating it.")
        try:
            loader_dump_path.mkdir()
        except Exception:
            logger.error(f"Could not create data directory {loader_dump_path}. Exiting.")
            sys.exit(1)

    collection_name = settings.get_str("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in configuration. Exiting.")
        sys.exit(1)
        # TODO: Validate that only numbers, letters and underscores are used.

    index_vendor = settings.get_str("INDEX_VENDOR")
    if not index_vendor:
        logger.warning("Index vendor (INDEX_VENDOR) not found in configuration. Defaulting to 'milvus'.")
        index_vendor = "milvus"

    # If LangSmith tracing is enabled, put a name for the project and verify that all required env vars are set.
    if settings.get_bool("LANGSMITH_TRACING", False):
        os.environ["LANGSMITH_PROJECT"] = f"{collection_name}"
        if settings.get("LANGSMITH_ENDPOINT") is None:
            logger.error("LANGSMITH_ENDPOINT (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if settings.get("LANGSMITH_API_KEY") is None:
            logger.error("LANGSMITH_API_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
    # If LangSmith prompts are enabled, put a name for the project and verify that all required env vars are set.
    if settings.get_bool("LANGSMITH_PROMPTS", False):
        os.environ["LANGSMITH_PROJECT"] = f"{collection_name}"
        os.environ["LANGSMITH_PROMPT_PREFIX"] = settings.get_str("LANGSMITH_PROMPT_PREFIX", "")
        if settings.get("LANGSMITH_ENDPOINT") is None:
            logger.error("LANGSMITH_ENDPOINT (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if settings.get("LANGSMITH_API_KEY") is None:
            logger.error("LANGSMITH_API_KEY (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)

    # If Langfuse tracing is enabled, verify that all required env vars are set.
    if settings.get_bool("LANGFUSE_TRACING", False):
        if settings.get("LANGFUSE_HOST") is None:
            logger.error("LANGFUSE_HOST (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if settings.get("LANGFUSE_PUBLIC_KEY") is None:
            logger.error("LANGFUSE_PUBLIC_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if settings.get("LANGFUSE_SECRET_KEY") is None:
            logger.error("LANGFUSE_SECRET_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
    # If Langfuse prompts are enabled, verify that all required env vars are set.
    if settings.get_bool("LANGFUSE_PROMPTS", False):
        if settings.get("LANGFUSE_HOST") is None:
            logger.error("LANGFUSE_HOST (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if settings.get("LANGFUSE_PUBLIC_KEY") is None:
            logger.error("LANGFUSE_PUBLIC_KEY (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if settings.get("LANGFUSE_SECRET_KEY") is None:
            logger.error("LANGFUSE_SECRET_KEY (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)

    user_agent = settings.get_str("USER_AGENT")
    if not user_agent:
        logger.info("User agent not found in configuration. Using default.")
        user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
    user_agent = f"{user_agent.format(version=__version__)}"

    embedding_model = settings.get_str("EMBEDDING_MODEL")
    if not embedding_model:
        logger.error("Embedding model not found in configuration. Exiting.")
        sys.exit(1)

    embedding_dimensions = settings.get_int("EMBEDDING_DIMENSIONS")
    if not embedding_dimensions:
        logger.error("Embedding dimensions not found in configuration. Exiting.")
        sys.exit(1)

    llm_model = settings.get_str("LLM_MODEL")
    if not llm_model:
        logger.error("LLM model not found in configuration. Exiting.")
        sys.exit(1)

    contextualisation_model = settings.get_str("CONTEXTUALISATION_MODEL")

    vector.store = load_vector_store(index_vendor)  # Set up the global wiki_rag.vector.store to be used elsewhere.

    wrapper_api_base = settings.get_str("WRAPPER_API_BASE")
    if not wrapper_api_base:
        logger.error("Wrapper API base not found in configuration. Exiting.")
        sys.exit(1)
    parts = wrapper_api_base.split(":")
    wrapper_server = parts[0]
    if len(parts) > 1:
        wrapper_port = int(parts[1])
    else:
        wrapper_port = 8000

    # These are optional, default to 0 (unlimited).
    wrapper_chat_max_turns = settings.get_int("WRAPPER_CHAT_MAX_TURNS", 0)
    wrapper_chat_max_tokens = settings.get_int("WRAPPER_CHAT_MAX_TOKENS", 0)
    wrapper_model_name = settings.get_str("WRAPPER_MODEL_NAME") or settings.get_str("COLLECTION_NAME")
    if not wrapper_model_name:
        logger.error("Public wrapper name not found in configuration. Exiting.")  # This is unreachable.
        sys.exit(1)

    # Prepare the configuration schema.
    # TODO, make prompt name, task_def, kb_*, cutoff, max tokens, temperature, top_p
    #  configurable. With defaults applied if not configured.
    server.context = ContextSchema(
        prompt_name="wiki-rag",
        product="Moodle",
        task_def="Moodle user documentation",
        kb_name="Moodle Docs",
        kb_url=mediawiki_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimensions,
        llm_model=llm_model,
        contextualisation_model=contextualisation_model,
        search_distance_cutoff=0.6,
        max_completion_tokens=1536,  # TODO: Make these 3 configurable.
        temperature=0.05,
        top_p=0.85,
        stream=False,
        wrapper_chat_max_turns=wrapper_chat_max_turns,
        wrapper_chat_max_tokens=wrapper_chat_max_tokens,
        wrapper_model_name=wrapper_model_name,
        langfuse_callback=None,
    )

    # Prepare the configuration.

    # If we want to use langfuse, let's instantiate the handler here, only once
    # (doing that in the server would create a new handler for each request and has
    # a big impact on threads and performance).
    if settings.get_bool("LANGFUSE_TRACING", False):
        server.context["langfuse_callback"] = CallbackHandler()

    logger.info("Building the graph")
    server.graph = build_graph(server.context)

    from wiki_rag.server.server import app

    # Start the web server
    uvicorn.run(
        app,
        host=wrapper_server,
        port=wrapper_port,
    )

    logger.info("wiki_rag-server finished.")


if __name__ == "__main__":
    main()
