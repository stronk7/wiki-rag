#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the knowledge base MCP compatible server."""

import logging
import os
import sys

from pathlib import Path

from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

import wiki_rag.index as index
import wiki_rag.mcp_server as mcp_global

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__, server
from wiki_rag.search.util import ContextSchema
from wiki_rag.util import setup_logging


def main():
    """Run the MCP server with all the configuration in place."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-server-mcp_server starting up...")

    # Print the version of the bot.
    logger.warning(f"Version: {__version__}")

    dotenv_file = ROOT_DIR / ".env"
    if dotenv_file.exists():
        logger.warning("Loading environment variables from %s", dotenv_file)
        logger.warning("Note: .env files are not supposed to be used in production. Use env secrets instead.")
        load_dotenv(dotenv_file)

    mediawiki_url = os.getenv("MEDIAWIKI_URL")
    if not mediawiki_url:
        logger.error("Mediawiki URL not found in environment. Exiting.")
        sys.exit(1)

    mediawiki_namespaces = os.getenv("MEDIAWIKI_NAMESPACES")
    if not mediawiki_namespaces:
        logger.error("Mediawiki namespaces not found in environment. Exiting.")
        sys.exit(1)
    mediawiki_namespaces = mediawiki_namespaces.split(",")
    mediawiki_namespaces = [int(ns.strip()) for ns in mediawiki_namespaces]  # no whitespace and int.
    mediawiki_namespaces = list(set(mediawiki_namespaces))  # unique

    loader_dump_path = os.getenv("LOADER_DUMP_PATH")
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

    collection_name = os.getenv("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in environment. Exiting.")
        sys.exit(1)
        # TODO: Validate that only numbers, letters and underscores are used.

    index.milvus_url = os.getenv("MILVUS_URL")
    if not index.milvus_url:
        logger.error("Milvus URL not found in environment. Exiting.")
        sys.exit(1)

    # If LangSmith tracing is enabled, put a name for the project and verify that all required env vars are set.
    if os.getenv("LANGSMITH_TRACING", "false") == "true":
        os.environ["LANGSMITH_PROJECT"] = f"{collection_name}"
        if os.getenv("LANGSMITH_ENDPOINT") is None:
            logger.error("LANGSMITH_ENDPOINT (required if tracing is enabled) not found in environment. Exiting.")
            sys.exit(1)
        if os.getenv("LANGSMITH_API_KEY") is None:
            logger.error("LANGSMITH_API_KEY (required if tracing is enabled) not found in environment. Exiting.")
            sys.exit(1)
    # If LangSmith prompts are enabled, put a name for the project and verify that all required env vars are set.
    if os.getenv("LANGSMITH_PROMPTS", "false") == "true":
        os.environ["LANGSMITH_PROJECT"] = f"{collection_name}"
        os.environ["LANGSMITH_PROMPT_PREFIX"] = os.environ["LANGSMITH_PROMPT_PREFIX"] or ""
        if os.getenv("LANGSMITH_ENDPOINT") is None:
            logger.error("LANGSMITH_ENDPOINT (required if prompts are enabled) not found in environment. Exiting.")
            sys.exit(1)
        if os.getenv("LANGSMITH_API_KEY") is None:
            logger.error("LANGSMITH_API_KEY (required if prompts are enabled) not found in environment. Exiting.")
            sys.exit(1)

    # If Langfuse tracing is enabled, verify that all required env vars are set.
    if os.getenv("LANGFUSE_TRACING", "false") == "true":
        if os.getenv("LANGFUSE_HOST") is None:
            logger.error(
                "LANGFUSE_HOST (required if tracing is enabled) not found in environment. Exiting."
            )
            sys.exit(1)
        if os.getenv("LANGFUSE_PUBLIC_KEY") is None:
            logger.error(
                "LANGFUSE_PUBLIC_KEY (required if tracing is enabled) not found in environment. Exiting."
            )
            sys.exit(1)
        if os.getenv("LANGFUSE_SECRET_KEY") is None:
            logger.error(
                "LANGFUSE_SECRET_KEY (required if tracing is enabled) not found in environment. Exiting."
            )
            sys.exit(1)
    # If Langfuse prompts are enabled, verify that all required env vars are set.
    if os.getenv("LANGFUSE_PROMPTS", "false") == "true":
        if os.getenv("LANGFUSE_HOST") is None:
            logger.error(
                "LANGFUSE_HOST (required if prompts are enabled) not found in environment. Exiting."
            )
            sys.exit(1)
        if os.getenv("LANGFUSE_PUBLIC_KEY") is None:
            logger.error(
                "LANGFUSE_PUBLIC_KEY (required if prompts are enabled) not found in environment. Exiting."
            )
            sys.exit(1)
        if os.getenv("LANGFUSE_SECRET_KEY") is None:
            logger.error(
                "LANGFUSE_SECRET_KEY (required if prompts are enabled) not found in environment. Exiting."
            )
            sys.exit(1)

    user_agent = os.getenv("USER_AGENT")
    if not user_agent:
        logger.info("User agent not found in environment. Using default.")
        user_agent = "Moodle Research Crawler/{version} (https://git.in.moodle.com/research)"
    user_agent = f"{user_agent.format(version=__version__)}"

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        logger.error("Embedding model not found in environment. Exiting.")
        sys.exit(1)

    embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    if not embedding_dimensions:
        logger.error("Embedding dimensions not found in environment. Exiting.")
        sys.exit(1)
    embedding_dimensions = int(embedding_dimensions)

    llm_model = os.getenv("LLM_MODEL")
    if not llm_model:
        logger.error("LLM model not found in environment. Exiting.")
        sys.exit(1)

    contextualisation_model = os.getenv("CONTEXTUALISATION_MODEL")

    mcp_api_base = os.getenv("MCP_API_BASE")
    if not mcp_api_base:
        logger.error("MCP API base not found in environment. Exiting.")
        sys.exit(1)
    parts = mcp_api_base.split(":")
    mcp_server = parts[0]
    if len(parts) > 1:
        mcp_port = int(parts[1])
    else:
        mcp_port = 8081

    # Calculate the file that we are going to use as source for the resources.
    input_candidate = ""
    for file in sorted(loader_dump_path.iterdir()):
        if file.is_file() and file.name.startswith(collection_name) and file.name.endswith(".json"):
            input_candidate = file
    if input_candidate:
        mcp_global.res_file = loader_dump_path / input_candidate

    if not mcp_global.res_file:
        logger.warning(f"No input file found in {loader_dump_path} with collection name {collection_name}.")

    # These are optional, default to 0 (unlimited).
    wrapper_chat_max_turns = int(os.getenv("WRAPPER_CHAT_MAX_TURNS", 0))
    wrapper_chat_max_tokens = int(os.getenv("WRAPPER_CHAT_MAX_TOKENS", 0))
    wrapper_model_name = os.getenv("WRAPPER_MODEL_NAME") or os.getenv("COLLECTION_NAME")
    if not wrapper_model_name:
        logger.error("Public wrapper name not found in environment. Exiting.")  # This is unreachable.
        sys.exit(1)

    # Prepare the configuration schema.
    # TODO, make prompt name, task_def, kb_*, cutoff, max tokens, temperature, top_p
    #  configurable. With defaults applied if not configured.
    server.context = ContextSchema(
        prompt_name="wiki-rag",
        task_def="Moodle user documentation",
        kb_name="Moodle Docs",
        kb_url=mediawiki_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimensions,
        llm_model=llm_model,
        contextualisation_model=contextualisation_model,
        search_distance_cutoff=0.6,
        max_completion_tokens=960,
        temperature=0.05,
        top_p=0.85,
        stream=False,
        wrapper_chat_max_turns=wrapper_chat_max_turns,
        wrapper_chat_max_tokens=wrapper_chat_max_tokens,
        wrapper_model_name=wrapper_model_name,
        langfuse_callback=None
    )

    # Prepare the configuration.

    # If we want to use langfuse, let's instantiate the handler here, only once
    # (doing that in the server would create a new handler for each request and has
    # a big impact on threads and performance).
    if os.getenv("LANGFUSE_TRACING", "false") == "true":
        server.context["langfuse_callback"] = CallbackHandler()

    # Start the mcp_server server
    from wiki_rag.mcp_server.server import mcp

    mcp.settings.host = mcp_server
    mcp.settings.port = mcp_port
    mcp.run("sse")
    # import asyncio
    # asyncio.run(mcp_server.run_sse_async())

    logger.info("wiki_rag-server-mcp_server finished.")


if __name__ == "__main__":
    main()
