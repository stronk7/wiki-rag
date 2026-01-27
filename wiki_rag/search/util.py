#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to use langgraph to conduct simple searches against the indexed database."""

import logging
import os
import pprint

from typing import Annotated, Literal, TypedDict

from cachetools import TTLCache, cached
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langfuse.model import TextPromptClient
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langsmith.client import Client

import wiki_rag.vector as vector

from wiki_rag import LOG_LEVEL
from wiki_rag.config import settings

logger = logging.getLogger(__name__)


class ContextSchema(TypedDict):
    """Define the context (immutable properties) that the graph will use.

    Used to validate the configuration passed to the RunnableConfig of the graph.
    """

    prompt_name: str
    product: str
    task_def: str
    kb_name: str
    kb_url: str
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    contextualisation_model: str | None
    search_distance_cutoff: float
    max_completion_tokens: int
    temperature: float
    top_p: float
    stream: bool
    wrapper_chat_max_turns: int
    wrapper_chat_max_tokens: int
    wrapper_model_name: str
    langfuse_callback: CallbackHandler | None


class RagState(TypedDict):
    """Overall state to follow the RAG graph execution."""

    history: list[BaseMessage]
    question: str
    vector_search: list[dict]
    context: list[str]
    sources: list[str]
    answer: str | None


class ContextualisedAnswer(TypedDict):
    """Contextualised question to be used in the RAG graph execution."""

    type: Annotated[str, "question", "The type of the answer (question or chitchat"]
    content: Annotated[str, ..., "The answer, that can be a question or some chitchat text"]


def build_graph(context: ContextSchema) -> CompiledStateGraph:
    """Build the graph for the langgraph search."""
    graph_builder = StateGraph(RagState, ContextSchema)
    graph_builder.add_node(query_rewrite)
    graph_builder.add_sequence([
        retrieve,
        optimise,
        generate
    ])
    graph_builder.add_conditional_edges("query_rewrite", retrieve_or_chitchat, {
        "retrieve": "retrieve",
        "chitchat": END,
    })

    graph_builder.add_edge(START, "query_rewrite")
    graph: CompiledStateGraph = graph_builder.compile().with_config(
        {"callbacks": [context["langfuse_callback"]]} if context["langfuse_callback"] else None
    )   # pyright: ignore[reportAssignmentType]. Note this is correct, but for some reason, pyright is not able.

    return graph


def retrieve_or_chitchat(state: RagState) -> Literal["retrieve", "chitchat"]:
    """Check if the answer is a chitchat or a rewrite.

    This is used to decide if we need to continue with the search or not.
    """
    # If the answer has already being set, we are done (probably a chitchat).
    if "answer" in state:
        return "chitchat"

    return "retrieve"  # Continue with the search.


@cached(cache=TTLCache(maxsize=64, ttl=0 if LOG_LEVEL == "DEBUG" else 300))
def load_prompts_for_rag(prompt_name: str) -> ChatPromptTemplate:
    """Load the prompts for the RAG model.

    This function results are cached for 5 minutes to avoid unnecessary calls to the LangSmith API.
    """
    chat_prompt = ChatPromptTemplate([])
    prefixed_prompt_name = prompt_name  # We'll add the prefix later, depending on the provider.
    prompt_provider = "local"

    # TODO: Be able to fallback to env/config based prompts too. Or also from other prompt providers.
    try:
        if settings.get_bool("LANGSMITH_PROMPTS", False):
            prefixed_prompt_name = f"{settings.get_str('LANGSMITH_PROMPT_PREFIX', '')}{prompt_name}"
            logger.info(f"Loading the prompt {prefixed_prompt_name} from LangSmith.")
            prompt_provider = "LangSmith"
            chat_prompt = Client().pull_prompt(prefixed_prompt_name)
        elif settings.get_bool("LANGFUSE_PROMPTS", False):
            langfuse = Langfuse()
            prefixed_prompt_name = f"{settings.get_str('LANGFUSE_PROMPT_PREFIX', '')}{prompt_name}"
            logger.info(f"Loading the prompt {prefixed_prompt_name} from Langfuse.")
            prompt_provider = "Langfuse"
            langfuse_prompt = langfuse.get_prompt(prefixed_prompt_name)
            # Convert the prompt to a LangChain compatible one.
            chat_prompt = convert_prompts_for_rag_from_langfuse(langfuse_prompt)
            langfuse.shutdown()
        else:
            chat_prompt = load_prompts_for_rag_from_local(prompt_name)
    except Exception as e:
        logger.warning(
            f"Error loading the prompt {prefixed_prompt_name} from {prompt_provider}: {e}. Applying default one."
        )
        chat_prompt = load_prompts_for_rag_from_local(prompt_name)

    logger.debug(f"Returning the prompt {prompt_name}: {chat_prompt}")
    return chat_prompt


def convert_prompts_for_rag_from_langfuse(langfuse_prompt: TextPromptClient) -> ChatPromptTemplate:
    """Convert the prompt from the Langfuse API.

    We need to make this prompt truly langchain compatible, so we need to
      - Create the system prompt template (from the first element coming from langfuse).
      - Insert the MessagesPlaceholder for the history.
      - Create the user message template (from the second element coming from langfuse).

      We do all the above by iterating over the langfuse prompt and creating the needed
      messages. Special attention to the {history} placeholder that is only supported by
      LangSmith/LangChain, so we simulate it in Langfuse with a normal user message with
      {{history}} as the content.
    """
    logger.debug(f"Converting Langfuse prompt {langfuse_prompt.name} to langchain format")

    langchain_list = langfuse_prompt.get_langchain_prompt()

    messages = []
    for langchain in langchain_list:
        if langchain[0] == "system":
            messages.append(SystemMessagePromptTemplate.from_template(langchain[1]))
        elif langchain[0] == "user":
            if langchain[1] == "{history}":  # Special placeholder to be replaced by the history.
                messages.append(MessagesPlaceholder("history", optional=True))
            else:
                messages.append(HumanMessagePromptTemplate.from_template(langchain[1]))
        else:
            logger.warning(f"Unknown prompt type ({langchain[0]}) detected in langfuse prompt: {langfuse_prompt.name}")

    chat_prompt = ChatPromptTemplate.from_messages(messages)
    return chat_prompt


def load_prompts_for_rag_from_local(prompt_name: str) -> ChatPromptTemplate:
    """Load the prompts from the local configuration."""
    # TODO: Be able to fallback to env/config based prompts too. Or also from other prompt providers.
    logger.info(f"Loading the prompt {prompt_name} from local.")

    # TODO: We should move the local handling of prompts to another place, this is not
    #   the best solution (say some local prompt management or whatever).

    # This is the default prompt that we use in the wiki-rag project.
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an assistant for question-answering tasks related to {task_def}, "
        "using the information present in {kb_name}, publicly available at {kb_url}."
        ""
        "Try to answer with a few phrases in a concise and clear way."
        "If the user asks for more details or explanations the answer can be longer."
        ""
        "You are provided with a <CONTEXT> XML element, that will be used to generate the answer."
        'Only the information present in the "Context" element will be used to generate the answer.'
        "This is the unique knowledge that can be used."
        ""
        "You are provided with a <SOURCES> XML element, with a list of URLs, that will be used "
        "to generate up to three references at the end of the answer."
        'Only the information present in the "Sources" element will be used to generate the references.'
        ""
        "If the Context information doesn't lead to a good answer, don't invent anything, "
        "just say that you don't know."
        'Avoid the term "context" in the answer.'
        "Avoid repeating the question in the answer."
        ""
        "The information is in mediawiki format, convert it to markdown in the answer."
        'Convert the links from mediawiki format "[url|text]" to markdown format [text](url)".'
    )
    user_message = HumanMessagePromptTemplate.from_template(
        "Question: {question}"
        ""
        "<CONTEXT>{context}</CONTEXT>"
        ""
        "<SOURCES>{sources}</SOURCES>"
        ""
        "Answer: "
    )

    # If we are using the wiki-rag-context-query prompt, let's update the needed pieces.
    if prompt_name == "wiki-rag-context-query":
        system_prompt = SystemMessagePromptTemplate.from_template(
            "Given the chat history and the original question which might reference "
            "context in the chat history, rephrase and expand the original question "
            "so it  can be understood without the chat history. Do NOT answer to the "
            "original question or ask for clarifications or details about it."
            ""
            "When not sure, always assume that the request is related to some {product} concept, "
            "for example: 'What is a capability' should be transformed to include the "
            "'{product}' text and look like 'What is a {product} capability'."
            ""
            "Always return a valid JSON structure with two elements:"
            '1. A "type" element with value "rewrite".'
            '2. A "content" element containing the rephrased question.'
            ""
            "If the original user question is not a question or a request for help, but a expression or some "
            "unrelated text, then answer to it in an educated and positive way. In this case, "
            'the "type" element on the required JSON structure will be "chitchat" instead of "rewrite".'
            ""
            "Do NOT make any reference in the answer to these guidelines."
        )
        user_message = HumanMessagePromptTemplate.from_template(
            "Original Question: {question}"
            ""
            "Answer: "
        )

    # Let's build the complete prompt with the system, the history and the user message.
    messages = (
        system_prompt,
        MessagesPlaceholder("history", optional=True),
        user_message,
    )
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    return chat_prompt


async def query_rewrite(state: RagState, runtime: Runtime[ContextSchema]) -> dict:
    """Rewrite the question using the query rewrite model.

    This is a simple query rewrite model that will be used to rewrite the question
    before sending it to the search engine. It will be used to improve the search results.

    The model will return a JSON object with the following fields:
    - type: The type of the answer (rewrite or chitchat)
    - content: The answer, that can be a question or some chitchat text

    In the case of a chitchat, the answer will be used as the final answer, otherwise
    the question will be rewritten (made context-aware) and used for the search.
    """
    if runtime.context["contextualisation_model"]:
        contextualise_prompt = load_prompts_for_rag(
            f"{runtime.context['prompt_name']}-context-query"
        )

        contextualised_answer = await contextualise_question(
            prompt=contextualise_prompt,
            question=state["question"],
            history=state["history"],
            product=runtime.context["product"],
            model=runtime.context["contextualisation_model"],
        )

        if contextualised_answer["type"] == "chitchat":
            # If the answer is a chitchat, we are going to use it as the final answer,
            # Let's generate a custom event to notify it to the graph caller about the chitchat answer.
            writer = get_stream_writer()
            writer(contextualised_answer)
            return {"answer": contextualised_answer["content"]}

        if contextualised_answer["type"] == "rewrite":
            # If the answer is a question, we are going to use it as the new question.
            return {"question": contextualised_answer["content"]}

    # No changes to the state, if arrived here.
    return {}


async def contextualise_question(
        prompt: ChatPromptTemplate,
        question: str,
        history: list[BaseMessage],
        product: str,
        model: str,
) -> dict:
    """Contextualise the question with the history and the model.

    This makes the RAG questions way better, context/history aware. The question
    only will be contextualised if there is some history and the model decides
    to provide a better alternative.

    Note that the history at this point has been already filtered, does not contain
    any system prompts and is already in the format expected.
    """
    logger.debug(f"Contextualising the question: {question}")
    chat = await prompt.ainvoke({
        "question": question,
        "history": history,
        "product": product,
    })

    llm = ChatOpenAI(
        model=model,
        max_completion_tokens=1536,  # TODO: Make these 3 configurable.
        top_p=0.85,
        temperature=0.1,
        disable_streaming=True,
    )
    json_llm = llm.with_structured_output(ContextualisedAnswer)

    answer = await json_llm.ainvoke(chat)

    logger.debug(f"Contextualised result: {answer}")

    return answer


async def retrieve(state: RagState, runtime: Runtime[ContextSchema]) -> dict:
    """Retrieve the best matches from the indexed database."""
    results = vector.store.retrieve(
        collection_name=runtime.context["collection_name"],
        embedding_model=runtime.context["embedding_model"],
        embedding_dimensions=runtime.context["embedding_dimension"],
        query=state["question"],
    )

    # TODO: Return only the docs which distance is below the cutoff.
    # distance_cutoff = runtime.context["search_distance_cutoff"]
    # return {"vector_search": [doc for doc in retrieved_docs[0] if doc["distance"] >= distance_cutoff]}
    return {"vector_search": results}


async def optimise(state: RagState, runtime: Runtime[ContextSchema]) -> dict:
    """Optimise the retrieved documents to build the context for the answer.

    First, we'll weight the elements retrieved by "popularity" (how many times they are mentioned
    by other elements in the retrieved docs). Then, we'll build the context by using some strategy,
    like Parent, Own and Children (poc) elements.
    """
    # TODO: Play with alternative strategies, we also have prev/nex, related, ...
    if not state["vector_search"]:  # No results, no context.
        return {"context": [], "sources": []}

    top = 15  # TODO: Make this part of the state, configurable.

    # Let's count how many times each element is mentioned as id, parent, children, previous or next,
    # making a dictionary with the counts. They will be weighted differently, depending on where they
    # appear, multiplied by their original distance and with a decay by position applied.
    weight_as_id = 1.5  # The document appears as id
    weight_as_parent = 1.1  # The document appears as parent
    weight_as_children = 1.01  # The document appears as children
    weight_as_previous = 1.005  # The document appears as previous
    weight_as_next = 1.005   # The document appears as next
    # TODO: Also weight relations once we have them.

    element_counts = {}
    for position in range(len(state["vector_search"])):
        doc = state["vector_search"][position]
        distance = doc["distance"]
        decay = 0.93 ** position  # Decay (exponentially) by position.
        for element in ["id", "parent", "children", "previous", "next"]:
            if element in doc["entity"]:
                el = doc["entity"][element]
                # Empty elements are not counted.
                if not el:
                    continue
                # If it's a single string element, we'll count it.
                if el and isinstance(el, str):
                    if el not in element_counts:
                        element_counts[el] = 0
                    # Only "id" and "parent" can be here (str).
                    element_counts[el] += distance * decay * (weight_as_id if element == "id"
                        else weight_as_parent if element == "parent"
                        else 1)
                # Else for sure it's a list/array like (iterable) element, so we'll count each element in the list.
                else:
                    for el in doc["entity"][element]:
                        if isinstance(el, str):
                            if el not in element_counts:
                                element_counts[el] = 0
                            # Only "children", "previous" and "next" can be here (list)
                            element_counts[el] += distance * decay * (weight_as_children if element == "children"
                                else weight_as_previous if element == "previous"
                                else weight_as_next if element == "next"
                                else 1)

    # Sort them by the weighted popularity results, we'll build the context following this order.
    sorted_items = sorted(element_counts.items(), key=lambda item: item[1], reverse=True)
    logger.debug(f"Sorted items: {pprint.PrettyPrinter(2).pformat(sorted_items)}")

    # TODO: Add support for other variations, including prev/next, related, etc.
    # Build the POC (parent, own, children) context
    new_context, new_sources = build_poc_context(
        retrieved_docs=state["vector_search"],
        sorted_items=sorted_items,
        collection_name=runtime.context["collection_name"],
        top=top
    )

    return {"context": new_context, "sources": new_sources}


def build_poc_context(retrieved_docs, sorted_items, collection_name: str, top=5) -> list[list[str]]:
    """Given the originally retrieved docs and the sorted, weighted items, build the rag final context.

    POC: Build the new context by using Parent, Own and Children elements.
    """
    context_list = []
    sources_list = []
    not_retrieved = []
    current = 0
    while current < min(len(sorted_items), top):
        # Let's examine the element in the sorted_items list.
        element_id = sorted_items[current][0]
        # Find the element in the retrieved_docs list.
        if element := [doc for doc in retrieved_docs if doc["entity"]["id"] == element_id]:
            element = element[0]
            # If the element has a parent, let's find it and add it to the context list (if not added already).
            if element["entity"]["parent"] and element["entity"]["parent"] not in context_list:
                context_list.append(element["entity"]["parent"])
            # Now, add the element itself to the context list (if not added already).
            if element["entity"]["id"] not in context_list:
                context_list.append(element["entity"]["id"])
                # Build the mediawiki link for the source.
                link = ""
                if element["entity"]["parent"]:
                    link = f"{element['entity']['doc_title']}: "
                link = f"[{element['entity']['source']}|{link}{element['entity']['title']}]"
                sources_list.append(link)
            # If the element has children, let's find them and add them to the context list (if not added already).
            if element["entity"]["children"]:
                for child in element["entity"]["children"]:
                    if child not in context_list:
                        context_list.append(child)

        else:
            # The element is not in the list of retrieved docs.
            # Just it won't be so useful in isolation, we will add it at the end, without further processing.
            not_retrieved.append(element_id)
            # This is not going to be very useful, let's allow one more iteration to happen.
            top += 1

        current += 1

    # Add all the not retrieved elements to the context list (at the end), if they are not already there.
    for not_retrieved_id in not_retrieved:
        if not_retrieved_id not in context_list:
            context_list.append(not_retrieved_id)

    return [
        retrieve_all_elements(retrieved_docs, context_list, collection_name),
        sources_list
    ]


def retrieve_all_elements(retrieved_docs, context_list, collection_name: str) -> list[str]:
    """Given the already built content_List, let's retrieve all the texts for the elements in the list."""
    context_texts = {}
    context_missing = []
    for id in context_list:
        # First, verify if we already have the text in the retrieved_docs list.
        retrieved = [doc for doc in retrieved_docs if doc["entity"]["id"] == id]
        if retrieved:
            context_texts[id] = f"{retrieved[0]["entity"]["title"]}\n\n{retrieved[0]["entity"]["text"]}"
        else:
            context_texts[id] = None
            # If not, let's accumulate it for later id based retrieval.
            context_missing.append(id)

    missing_docs = get_missing_from_vector_store(context_missing, collection_name)

    # Finally, fill the gaps in the context_texts dictionary with the missing docs.
    for id, text in missing_docs.items():
        context_texts[id] = text

    # Now, iterate over the dictionary and return the list of texts.
    return [context_texts[id] for id in context_list]


def get_missing_from_vector_store(context_missing: list, collection_name: str) -> dict:
    """Given the missing elements, let's retrieve them from the vector store."""
    if not context_missing:  # No missing elements, nothing extra to retrieve.
        return {}

    return vector.store.get_documents_contents_by_id(collection_name, context_missing)


async def generate(state: RagState, runtime: Runtime[ContextSchema]) -> dict:
    """Generate the final answer for the question.

    This is the final generation step where the prompt, the chat history and the context
    are used to generate the final answer.
    """
    # Let's make the generation LLM call normally.
    llm = ChatOpenAI(
        model=runtime.context["llm_model"],
        max_completion_tokens=runtime.context["max_completion_tokens"],
        top_p=runtime.context["top_p"],
        temperature=runtime.context["temperature"],
    )

    docs_content = "\n\n".join(f"{doc}" for doc in state["context"])
    sources_content = "\n".join(f"* {source}" for source in state["sources"])

    chat_prompt = load_prompts_for_rag(runtime.context["prompt_name"])
    chat = await chat_prompt.ainvoke({
        "task_def": runtime.context["task_def"],
        "kb_name": runtime.context["kb_name"],
        "kb_url": runtime.context["kb_url"],
        "context": docs_content,
        "sources": sources_content,
        "question": state["question"],
        "history": state["history"]
    })

    response = await llm.ainvoke(chat)

    return {"answer": response.content}
