#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main entry point for the KB retriever and search system."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import traceback

from pathlib import Path

from langchain_core.messages import AIMessageChunk
from langfuse.langchain import CallbackHandler

import wiki_rag.vector as vector

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__
from wiki_rag.config import get_config
from wiki_rag.search.util import ContextSchema, build_graph
from wiki_rag.util import setup_logging
from wiki_rag.vector import load_vector_store


async def run():
    """Perform a search with all the configuration in place."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-search starting up...")

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
        os.environ["LANGSMITH_PROJECT"] = collection_name
        if not config.get("langsmith.endpoint"):
            logger.error("LANGSMITH_ENDPOINT (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
        if not config.get("langsmith.api_key"):
            logger.error("LANGSMITH_API_KEY (required if tracing is enabled) not found in configuration. Exiting.")
            sys.exit(1)
    if config.get_bool("langsmith.prompts", False):
        os.environ["LANGSMITH_PROJECT"] = collection_name
        os.environ["LANGSMITH_PROMPT_PREFIX"] = config.get("langsmith.prompt_prefix", "")
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

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="The question to be asked.")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream the output.")

    args = parser.parse_args()

    question = " ".join(args.question)
    stream = args.stream if args.stream else False
    logger.debug(f'Question: "{question}"')

    context = ContextSchema(
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
        stream=stream,
        wrapper_chat_max_turns=0,
        wrapper_chat_max_tokens=0,
        wrapper_model_name=llm_model,
        langfuse_callback=None,
    )

    if config.get_bool("langfuse.tracing", False):
        context["langfuse_callback"] = CallbackHandler()

    logger.info("Building the graph")
    graph = build_graph(context)

    if not stream:
        logger.info("Running the search (non-streaming)")
        response = await graph.ainvoke(
            input={"question": question, "history": []},
            context=context
        )
        print(f"\033[93m{response["answer"]}\033[0m", end="")
    else:
        logger.info("Running the search (streaming)")
        async for mode, info in graph.astream(
            input={"question": question, "history": []},
            context=context,
            stream_mode=["custom", "messages"],
        ):
            process_event = False
            content = ""
            if (
                mode == "custom"
                and isinstance(info, dict)
                and info.get("type", "") == "chitchat"
            ):
                process_event = True
                content = info.get(
                    "content", "There was a problem talking with you, I'm sorry."
                )

            if mode == "messages":
                message = info[0]
                metadata = info[1]
                if (
                    isinstance(message, AIMessageChunk)
                    and isinstance(metadata, dict)
                    and metadata.get("langgraph_node", "") == "generate"
                    and message.content
                ):
                    process_event = True
                    content = message.content

            if process_event:
                print(f"\033[93m{content}\033[0m", end="", flush=True)

    print("", end="\n\n")

    logger.info("wiki_rag-search finished.")


async def release(signum: int) -> None:
    """Signal handler for SIGINT and SIGTERM. It will release resources and exit gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, releasing resources.")

    for task in asyncio.all_tasks():
        logger.debug(f"Cancelling task {task}")
        task.cancel()

    logger.info("Resources released. Exiting.")


def add_signal_handlers(run_loop) -> None:
    """Register async signal handlers (SIGINT/ctrl+c, SIGTERM/kill)."""
    run_loop.add_signal_handler(
        signal.SIGINT,
        lambda: asyncio.create_task(release(signal.SIGINT))
    )
    run_loop.add_signal_handler(
        signal.SIGTERM,
        lambda: asyncio.create_task(release(signal.SIGTERM))
    )


def main():
    """Prepare the async loop for operation and graceful shutdown, then run()."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    add_signal_handlers(loop)
    exitcode = 0
    try:
        loop.run_until_complete(run())
    except Exception:
        traceback.print_exc()
        exitcode = 1
    finally:
        loop.close()
        sys.exit(exitcode)


if __name__ == "__main__":
    main()