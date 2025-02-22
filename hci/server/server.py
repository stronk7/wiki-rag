#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import asyncio
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import hci.server

from hci import __version__
from hci.server.util import ModelsListResponse, ModelResponse, ChatCompletionRequest, Message
from hci.server.util import filter_completions_history, convert_messages_to_langchain

from langchain_core.messages import AIMessageChunk

logger = logging.getLogger(__name__)

model_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 2048,
}

app_description = """
This is an **OpenAI-compatible [API](https://platform.openai.com/docs/api-reference/introduction)**
that provides a chat completion endpoint.

It enables easy-to-execute callbacks to be able to run, virtually, any (Python) stuff.

Only a few endpoints are implemented, but they should be enough to integrate it with
any OpenAI-compatible [client](https://platform.openai.com/docs/libraries).

It supports authentication, streaming completions, chat history, and more.
"""

model = "moodledocs405" # This is the only model supported by this API.

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
)

@app.get("/models", tags=["models"])
@app.get("/v1/models", tags=["models"])
async def models_list() -> ModelsListResponse:
    return ModelsListResponse(
        object="list",
        data=[
            ModelResponse(
                id=hci.server.config["configurable"]["collection_name"],
                object="model",
                created=int(time.time()),
                owned_by="research.moodle.com"
            )
        ]
    )


@app.post("/chat/completions", tags=["chat"])
@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    if not request.model:
        raise HTTPException(status_code=400, detail="No model provided.")

    if request.model != model:
        raise HTTPException(status_code=400, detail="Model not supported.")
    
    logger.debug(f"Request: {request}")
    
    # Update the configuration with the values coming from the request.
    hci.server.config["configurable"]["temperature"] = request.temperature
    hci.server.config["configurable"]["top_p"] = request.top_p
    hci.server.config["configurable"]["max_completion_tokens"] = request.max_completion_tokens or request.max_tokens
    hci.server.config["configurable"]["stream"] = request.stream or False
    
    # Filter the messages to ensure they don't exceed the maximum number of turns and tokens.
    history = filter_completions_history(
        request.messages,
        max_turns_allowed=hci.server.config["configurable"]["wrapper_chat_max_turns"],
        max_tokens_allowed=hci.server.config["configurable"]["wrapper_chat_max_tokens"],
        remove_system_messages=True
    )
    
    # Extract the last message, our new question, out from history.
    question = history.pop()["content"]
    
    # Convert the messages to the format expected by langgraph.
    history = convert_messages_to_langchain(history)

    logger.debug(f"Configuration: {hci.server.config['configurable']}")
    logger.debug(f"Filtered history: {history}")
    logger.debug(f"Question: {question}")

    # Run the search.
    if hci.server.config["configurable"]["stream"]:
        logger.info(f"Running the search (streaming)")
        
        async def open_ai_langgraph_stream(question: str, history: list[Message]):
            index: int = 0
            for message, metadata in hci.server.graph.stream({"question": question, "history": history}, config=hci.server.config, stream_mode="messages"):
                assert isinstance(message, AIMessageChunk)
                assert isinstance(metadata, dict)
                if metadata.get("langgraph_node") == "generate" and message.content:
                    chunk = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": index,
                            "delta": {
                                "role": "assistant",
                                "content": f"{message.content}"
                            }
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    index += 1
            yield "data: [DONE]\n\n"

        return StreamingResponse(open_ai_langgraph_stream(question, history), media_type="text/event-stream")
    else:
        logger.info(f"Running the search (non-streaming)")
        completion = hci.server.graph.invoke({"question": question, "history": history}, config=hci.server.config)
        return {
            "id": uuid.uuid4(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"message": Message(role="assistant", content=completion["answer"])}],
        }