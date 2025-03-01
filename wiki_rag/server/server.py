#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Fast API server for the OpenAI-compatible API."""

import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessage

from wiki_rag import __version__, server
from wiki_rag.server.util import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceResponse,
    Message,
    ModelResponse,
    ModelsListResponse,
    convert_from_openai_to_langchain,
    filter_completions_history,
)

logger = logging.getLogger(__name__)

app_description = """Enables easy-to-execute completion callbacks to be able to run, virtually, any (Python) stuff.

Only a few endpoints are implemented, but they should be enough to integrate it with
any OpenAI-compatible [client](https://platform.openai.com/docs/libraries).

It supports authentication, streaming completions, chat history, and more.
"""

tags_metadata = [
    {
        "name": "models",
        "description": "Operations related to models.",
    },
    {
        "name": "chat",
        "description": "Operations related to chat completions.",
    }
]
app = FastAPI(
    title="OpenAI-compatible API",
    summary="A lightweight OpenAI-compatible API implementation to serve applications as LLMs.",
    description=app_description,
    version=__version__,
    contact={
        "name": "Moodle HQ - Research",
        "url": "https://git.in.moodle.com/research",
        "email": "research@moodle.com",
    },
    license_info={
        "name": "BSD-3-Clause",
        "url": "https://opensource.org/license/bsd-3-clause",
    },
    openapi_tags=tags_metadata,
    docs_url=None,  # Disable Swagger UI
    redoc_url="/docs",  # Put the ReDoc UI at /docs
)


@app.get(
    path="/models",
    tags=["models"],
    deprecated=True
)
@app.get(
    path="/v1/models",
    tags=["models"]
)
async def models_list() -> ModelsListResponse:
    """List the models available in the API."""
    assert server.config is not None and "configurable" in server.config
    return ModelsListResponse(
        object="list",
        data=[
            ModelResponse(
                id=server.config["configurable"]["collection_name"],
                object="model",
                created=int(time.time()),
                owned_by=server.config["configurable"]["kb_name"],
            )
        ]
    )


@app.post(
    path="/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["chat"],
    deprecated=True
)
@app.post(
    path="/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["chat"]
)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
    """Generate chat completions based on the given messages and model."""
    assert server.config is not None and "configurable" in server.config
    assert server.graph is not None

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    if not request.model:
        raise HTTPException(status_code=400, detail="No model provided.")

    if request.model != server.config["configurable"]["collection_name"]:
        raise HTTPException(status_code=400, detail="Model not supported.")

    logger.debug(f"Request: {request}")

    # Update the configuration with the values coming from the request.
    # TODO, Apply defaults applied if not configured.
    server.config["configurable"]["temperature"] = request.temperature
    server.config["configurable"]["top_p"] = request.top_p
    server.config["configurable"]["max_completion_tokens"] = request.max_completion_tokens or request.max_tokens
    server.config["configurable"]["stream"] = request.stream or False

    # Filter the messages to ensure they don't exceed the maximum number of turns and tokens.
    history = filter_completions_history(
        request.messages,
        max_turns_allowed=server.config["configurable"]["wrapper_chat_max_turns"],
        max_tokens_allowed=server.config["configurable"]["wrapper_chat_max_tokens"],
        remove_system_messages=True
    )

    # Extract the last message, our new question, out from history.
    question = history.pop()["content"]

    # Convert the messages to the format expected by langgraph.
    history = convert_from_openai_to_langchain(history)

    logger.debug(f"Configuration: {server.config['configurable']}")
    logger.debug(f"Filtered history: {history}")
    logger.debug(f"Question: {question}")

    # Run the search.
    if server.config["configurable"]["stream"]:
        logger.info("Running the search (streaming)")

        async def open_ai_langgraph_stream(question: str, history: list[BaseMessage]):
            assert server.graph is not None

            index: int = 0
            async for message, metadata in server.graph.astream(
                    {"question": question, "history": history},
                    config=server.config,
                    stream_mode="messages"
            ):
                if (
                    isinstance(message, AIMessageChunk) and
                    isinstance(metadata, dict) and
                    metadata.get("langgraph_node", "") == "generate" and
                    message.content
                ):
                    chunk = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": index,
                            "delta": {
                                "role": "assistant",
                                "content": message.content
                            }
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    index += 1
            yield "data: [DONE]\n\n"

        return StreamingResponse(open_ai_langgraph_stream(question, history), media_type="text/event-stream")
    else:
        logger.info("Running the search (non-streaming)")
        completion = await server.graph.ainvoke(
            {"question": question, "history": history},
            config=server.config
        )

        return ChatCompletionResponse(
            id=uuid.uuid4(),
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChoiceResponse(
                    index=0,
                    message=Message(role="assistant", content=completion["answer"]),
                ),
            ],
        )
