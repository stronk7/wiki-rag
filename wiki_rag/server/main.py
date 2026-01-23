#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the knowledge base OpenAI compatible server."""

import logging
import sys

from pathlib import Path

import uvicorn

from langfuse.langchain import CallbackHandler

import wiki_rag.vector as vector

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__, server
from wiki_rag.config import get_config
from wiki_rag.search.util import ContextSchema, build_graph
from wiki_rag.util import setup_logging
from wiki_rag.vector import load_vector_store


def main():
    """Run the OpenAI server with all the configuration in place."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-server starting up...")

    logger.warning(f"Version: {__version__}")

    config = get_config()

    mediawiki_url = config.get("mediawiki.url")
    mediawiki_namespaces = config.get_int_list("mediawiki.namespaces")

    loader_dump_path_str = config.get("collection.dump_path")
    if loader_dump_path_str:
        loader_dump_path = Path(loader_dump_path_str)
    else:
        loader_dump_path = ROOT_DIR / "data"
    if not loader_dump_path.exists():
        logger.warning(f"Data directory {loader_dump_path} not found. Creating it.")
        try:
            loader_dump_path.mkdir()
        except Exception:
            logger.error(f"Could not create data directory {loader_dump_path}. Exiting.")
            sys.exit(1)

    collection_name = config.get("collection.name")
    index_vendor = config.get("index_vendor", "milvus")

    if config.get_bool("langsmith.tracing", False):
        if not config.get("langsmith.endpoint"):
            logger.error("LANGSMITH_ENDPOINT (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langsmith.api_key"):
            logger.error("LANGSMITH_API_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
    if config.get_bool("langsmith.prompts", False):
        if not config.get("langsmith.endpoint"):
            logger.error("LANGSMITH_ENDPOINT (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langsmith.api_key"):
            logger.error("LANGSMITH_API_KEY (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)

    if config.get_bool("langfuse.tracing", False):
        if not config.get("langfuse.host"):
            logger.error("LANGFUSE_HOST (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langfuse.public_key"):
            logger.error("LANGFUSE_PUBLIC_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langfuse.secret_key"):
            logger.error("LANGFUSE_SECRET_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
    if config.get_bool("langfuse.prompts", False):
        if not config.get("langfuse.host"):
            logger.error("LANGFUSE_HOST (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langfuse.public_key"):
            logger.error("LANGFUSE_PUBLIC_KEY (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langfuse.secret_key"):
            logger.error("LANGFUSE_SECRET_KEY (required if prompts are enabled) not found in configuration. Exiting.")
            sys.exit(1)

    embedding_model = config.get("models.embedding")
    embedding_dimensions = config.get_int("models.embedding_dimensions")
    llm_model = config.get("models.llm")
    contextualisation_model = config.get("models.contextualisation")

    vector.store = load_vector_store(index_vendor)

    wrapper_api_base = config.get("wrapper.api_base", "0.0.0.0:8080")
    parts = wrapper_api_base.split(":")
    wrapper_server = parts[0]
    if len(parts) > 1:
        wrapper_port = int(parts[1])
    else:
        wrapper_port = 8000

    wrapper_chat_max_turns = config.get_int("wrapper.chat_max_turns", 0)
    wrapper_chat_max_tokens = config.get_int("wrapper.chat_max_tokens", 0)
    wrapper_model_name = config.get("wrapper.model_name") or config.get("collection.name")

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
        max_completion_tokens=1536,
        temperature=0.05,
        top_p=0.85,
        stream=False,
        wrapper_chat_max_turns=wrapper_chat_max_turns,
        wrapper_chat_max_tokens=wrapper_chat_max_tokens,
        wrapper_model_name=wrapper_model_name,
        langfuse_callback=None,
    )

    if config.get_bool("langfuse.tracing", False):
        server.context["langfuse_callback"] = CallbackHandler()

    logger.info("Building the graph")
    server.graph = build_graph(server.context)

    from wiki_rag.server.server import app

    uvicorn.run(
        app,
        host=wrapper_server,
        port=wrapper_port,
    )

    logger.info("wiki_rag-server finished.")


if __name__ == "__main__":
    main()