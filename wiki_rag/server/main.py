#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the knowledge base OpenAI compatible server."""

import logging
import os
import sys

from pathlib import Path

import uvicorn

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

import wiki_rag.index as index

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__, server
from wiki_rag.search.util import ConfigSchema, build_graph
from wiki_rag.util import setup_logging


def main():
    """Make an index from the json information present in the specified file."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-server starting up...")

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
        logger.error(f"Data directory {loader_dump_path} not found. Please ensure it exists. Exiting.")
        sys.exit(1)

    collection_name = os.getenv("COLLECTION_NAME")
    if not collection_name:
        logger.error("Collection name not found in environment. Exiting.")
        sys.exit(1)

    index.milvus_url = os.getenv("MILVUS_URL")
    if not index.milvus_url:
        logger.error("Milvus URL not found in environment. Exiting.")
        sys.exit(1)

    # If tracing is enabled, put a name for the project.
    if os.getenv("LANGSMITH_TRACING", "false") == "true":
        os.environ["LANGSMITH_PROJECT"] = f"{collection_name}"

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

    wrapper_api_base = os.getenv("WRAPPER_API_BASE")
    if not wrapper_api_base:
        logger.error("Wrapper API base not found in environment. Exiting.")
        sys.exit(1)
    parts = wrapper_api_base.split(":")
    wrapper_server = parts[0]
    if len(parts) > 1:
        wrapper_port = int(parts[1])
    else:
        wrapper_port = 8000

    # These are optional, default to 0 (unlimited).
    wrapper_chat_max_turns = int(os.getenv("WRAPPER_CHAT_MAX_TURNS", 0))
    wrapper_chat_max_tokens = int(os.getenv("WRAPPER_CHAT_MAX_TOKENS", 0))

    logger.info("Building the graph")
    server.graph = build_graph()

    # Prepare the configuration schema.
    # TODO, make prompt name, task_def, kb_*, cutoff, max tokens, temperature, top_p
    #  configurable. With defaults applied if not configured.
    config_schema = ConfigSchema(
        prompt_name="moodlehq/wiki-rag",
        task_def="Moodle user documentation",
        kb_name="Moodle Docs",
        kb_url=mediawiki_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimensions,
        llm_model=llm_model,
        search_distance_cutoff=0.6,
        max_completion_tokens=768,
        temperature=0.05,
        top_p=0.85,
        stream=False,
        wrapper_chat_max_turns=wrapper_chat_max_turns,
        wrapper_chat_max_tokens=wrapper_chat_max_tokens,
    ).items()

    # Prepare the configuration.
    server.config = RunnableConfig(configurable=dict(config_schema))

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
