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

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig

import wiki_rag.index as index

from wiki_rag import LOG_LEVEL, ROOT_DIR, __version__
from wiki_rag.search.util import ConfigSchema, build_graph
from wiki_rag.util import setup_logging


async def run():
    """Perform a search with all the configuration in place."""
    setup_logging(level=LOG_LEVEL)
    logger = logging.getLogger(__name__)
    logger.info("wiki_rag-search starting up...")

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

    # Let's accept arg[1] as the question to be asked.
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="The question to be asked.")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream the output.")

    args = parser.parse_args()

    question = " ".join(args.question)

    stream = args.stream if args.stream else False
    logger.debug(f'Question: "{question}"')

    logger.info("Building the graph")
    graph = build_graph()

    # Prepare the configuration schema.
    # TODO, make prompt name, task_def, kb_*, cutoff, max tokens, temperature, top_p
    #  configurable. With defaults applied if not configured.
    config_schema = ConfigSchema(
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
        stream=stream,
        wrapper_chat_max_turns=0,
        wrapper_chat_max_tokens=0,
        wrapper_model_name=llm_model,
    ).items()

    # Prepare the configuration.
    config = RunnableConfig(configurable=dict(config_schema))

    # display(Image((graph.get_graph().draw_mermaid_png())))

    # And, finally, run a search.
    if not stream:
        logger.info("Running the search (non-streaming)")
        response = await graph.ainvoke({"question": question, "history": []}, config=config)
        print(f"\033[93m{response["answer"]}\033[0m", end="")
    else:
        logger.info("Running the search (streaming)")
        # TODO: Encapsulate this, it's duplicated in the server.
        async for mode, info in graph.astream(
                {"question": question, "history": []},
                config=config,
                stream_mode=["custom", "messages"]):
            # See if the streamed event needs to be considered.
            process_event = False
            content = ""
            # Accept custom events coming from the query_rewrite node.
            if (
                mode == "custom"
                and isinstance(info, dict)
                and info.get("type", "") == "chitchat"
            ):
                process_event = True
                content = info.get(
                    "content", "There was a problem talking with you, I'm sorry."
                )

            # Accept AI message chunks events coming from the generate node.
            if mode == "messages":
                # Message events (when using multiple stream mode, like here) are
                # tuples with the message and metadata. When not using multiple stream mode,
                # but only one, they come as 2 returned values (message, metadata), without mode.
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

    # Add all the cleanup code here.

    # Cancel all tasks, so we can exit.
    for task in asyncio.all_tasks():
        logger.debug(f"Cancelling task {task}")
        task.cancel()

    logger.info("Resources released. Exiting.")


def add_signal_handlers(run_loop) -> None:
    """Register async signal handlers (SIGINT/ctrl+c, SIGTERM/kill).

    Do it in own async loop, so we can release stuff gracefully.
    """
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
    # Create the event loop, set it as current and add the signal handlers.
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.get_event_loop_policy().set_event_loop(loop)
    add_signal_handlers(loop)
    exitcode = 0
    try:
        loop.run_until_complete(run())  # Run the main loop.
    except Exception:
        traceback.print_exc()
        exitcode = 1
    finally:
        loop.close()
        sys.exit(exitcode)


if __name__ == "__main__":
    main()
