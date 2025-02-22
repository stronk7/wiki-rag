#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

""" Util functions to serve langgraph behind an OpenAI API wrapper."""
import asyncio
import json
import logging
import time

from typing import List, Optional, TypedDict

from langchain_community.adapters import openai
from langchain_core.messages import BaseMessage

import tiktoken
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class Message(TypedDict):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    max_tokens: Optional[int] = 768  # Max tokens to generate (not all models support this). Deprecated.
    max_completion_tokens: Optional[int] = 768  # Max tokens to generate (not all models support this).
    temperature: Optional[float] = 0.1  # Temperature for sampling (0.0, deterministic to 1.0, creative).
    top_p: Optional[float] = 0.95  # Which probability (0.0 - 1.0) is used to consider the next token (0.95 default).
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "research.moodle.com"


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelResponse]

def filter_completions_history(
        history: list[Message],
        max_turns_allowed: int,
        max_tokens_allowed: int,
        remove_system_messages: bool) -> list[Message]:

    # Remove system messages.
    if remove_system_messages:
        history = [msg for msg in history if (msg["role"] != "system" and msg["role"] != "developer")]

    # Filter turns.
    if max_turns_allowed > 0:
        current_turns = count_turns(history)
        if current_turns > max_turns_allowed:
            keep_messages = max_turns_allowed * 2 + 1
            history = history[-keep_messages:]

    # Filter tokens.
    if max_tokens_allowed > 0:
        current_tokens = count_tokens(history)
        if current_tokens > max_tokens_allowed:
            # Calculate the start and end pointers to keep.
            end = len(history) - 2  # The last message isn't ever deleted (it's the actual request).
            start = 0
            while start < len(history) and (history[start]["role"] == 'assistant' or start + 1 > end):
                start += 1

            # Go calculating start and end until the number of tokens is less than the max.
            while start <= end:
                # The first remaining message cannot be assistant ever.
                if history[start]["role"] == 'assistant':
                    current_tokens -= count_tokens([history[start]])
                    start += 1
                    continue

                # Still too many tokens, remove the first user message.
                if history[start]["role"] == 'user':
                    if current_tokens > max_tokens_allowed:
                        current_tokens -= count_tokens([history[start]])
                        start += 1
                        continue
                    else:
                        break # We are done, user message is within the allowed tokens.

            history = history[start:]

    return history

def count_turns(messages: list[Message]) -> int:
    return (len(messages) - 1) // 2

def count_tokens(messages: list[Message]) -> int:
    encoder = tiktoken.get_encoding("gpt2") # Model is not very important here, differences are minimal.
    return sum([len(encoder.encode(msg["content"])) for msg in messages])


def convert_openai_messages(messages: list[Message]) -> list[BaseMessage]:
    """Convert the messages to the format expected by langgraph."""

    return [
        openai.convert_dict_to_message(message) for message in messages
    ]

async def _async_resp_generator(text_resp: str, request: ChatCompletionRequest):

    for chunk in model.generate_content(text_resp, stream=True):
        for i, token in enumerate(chunk.split(" ")):
            chunk = {
                "id": i,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "content": f"{token} "
                    }
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"

def convert_messages_to_langchain(history: list[Message]) -> list:
    return convert_openai_messages(history)