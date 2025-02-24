#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to serve langgraph behind an OpenAI API wrapper."""

import logging
import time

from typing import TypedDict

import tiktoken

from langchain_community.adapters import openai
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Message(TypedDict):
    """Base message class. Simple but enough for OpenAI."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request model. Includes the messages history and the model to use."""

    max_tokens: int | None = 768  # Max tokens to generate (not all models support this). Deprecated.
    max_completion_tokens: int | None = 768  # Max tokens to generate (not all models support this).
    temperature: float | None = 0.1  # Temperature for sampling (0.0, deterministic to 1.0, creative).
    top_p: float | None = 0.95  # Which probability (0.0 - 1.0) is used to consider the next token (0.95 default).
    model: str
    messages: list[Message]
    stream: bool | None = False


class ModelResponse(BaseModel):
    """Information about a LLM model."""

    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "research.moodle.com"


class ModelsListResponse(BaseModel):
    """Information about the available models."""

    object: str = "list"
    data: list[ModelResponse]


def filter_completions_history(
        history: list[Message],
        max_turns_allowed: int,
        max_tokens_allowed: int,
        remove_system_messages: bool) -> list[Message]:
    """Filter the history of completions based on the number of turns and tokens.

    This is used to avoid the model to receive too much history (some clients are
    always sending the whole conversation history)
    """
    # Remove system messages.
    if remove_system_messages:
        history = [msg for msg in history if (msg["role"] != "system" and msg["role"] != "developer")]

    # Filter turns.
    if max_turns_allowed > 0:
        current_turns = _count_turns(history)
        if current_turns > max_turns_allowed:
            keep_messages = max_turns_allowed * 2 + 1
            history = history[-keep_messages:]

    # Filter tokens.
    if max_tokens_allowed > 0:
        current_tokens = _count_tokens(history)
        if current_tokens > max_tokens_allowed:
            # Calculate the start and end pointers to keep.
            end = len(history) - 2  # The last message isn't ever deleted (it's the actual request).
            start = 0
            while start < len(history) and (history[start]["role"] == "assistant" or start + 1 > end):
                start += 1

            # Go calculating start and end until the number of tokens is less than the max.
            while start <= end:
                # The first remaining message cannot be assistant ever.
                if history[start]["role"] == "assistant":
                    current_tokens -= _count_tokens([history[start]])
                    start += 1
                    continue

                # Still too many tokens, remove the first user message.
                if history[start]["role"] == "user":
                    if current_tokens > max_tokens_allowed:
                        current_tokens -= _count_tokens([history[start]])
                        start += 1
                        continue
                    else:
                        break  # We are done, user message is within the allowed tokens.

            history = history[start:]

    return history


def _count_turns(messages: list[Message]) -> int:
    """Count how many "turns" are in the message history."""
    return (len(messages) - 1) // 2


def _count_tokens(messages: list[Message]) -> int:
    """Count how many tokens are in the message history."""
    encoder = tiktoken.get_encoding("gpt2")  # Model is not very important here, differences are minimal.
    return sum([len(encoder.encode(msg["content"])) for msg in messages])


def convert_from_openai_to_langchain(history: list[Message]) -> list[BaseMessage]:
    """Convert the messages to the format expected by langgraph."""
    return [
        openai.convert_dict_to_message(message) for message in history
    ]
