#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to use langgraph to conduct simple searches against the indexed database."""

import logging
import pprint

from typing import TypedDict

from cachetools import TTLCache, cached
from langchain import hub
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.state import START, CompiledStateGraph, StateGraph
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from wiki_rag import LOG_LEVEL

logger = logging.getLogger(__name__)


class ConfigSchema(TypedDict):
    """Define the configuration that the graph will use.

    Used to validate the configuration passed to the RunnableConfig of the graph.
    """

    prompt_name: str
    task_def: str
    kb_name: str
    kb_url: str
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    search_distance_cutoff: float
    max_completion_tokens: int
    top_p: float
    temperature: float
    stream: bool
    wrapper_chat_max_turns: int
    wrapper_chat_max_tokens: int


class RagState(TypedDict):
    """Overall state to follow the RAG graph execution."""

    history: list[BaseMessage]
    question: str
    vector_search: list[dict]
    context: list[str]
    sources: list[str]
    answer: str | None


def build_graph() -> CompiledStateGraph:
    """Build the graph for the langgraph search."""
    graph_builder = StateGraph(RagState, ConfigSchema).add_sequence([
        retrieve,
        optimise,
        generate
    ])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


@cached(cache=TTLCache(maxsize=64, ttl=0 if LOG_LEVEL == "DEBUG" else 300))
def load_prompts_for_rag(prompt_name: str) -> ChatPromptTemplate:
    """Load the prompts for the RAG model.

    This function results are cached for 10 minutes to avoid unnecessary calls to the LangSmith API.
    """
    chat_prompt = ChatPromptTemplate([])

    # TODO: Be able to fallback to env/config based prompts too. Or also from other prompt providers.
    logger.info(f"Loading the prompt {prompt_name} from LangSmith.")
    try:
        chat_prompt = hub.pull(prompt_name)
    except Exception as e:
        logger.warning(f"Error loading the prompt {prompt_name} from LangSmith: {e}. Applying default one.")
        # Use the manual prompt building instead.
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks related to {task_def}."
            ""
            "The sources for you knowledge are the {kb_name}, available at {kb_url}."
            ""
            "Try to answer with a few phrases in a concise and clear way."
            "If the user asks for more details or explanations the answer can be longer."
            ""
            "You are provided with a  <CONTEXT> XML element, that will be used to generate the answer."
            'Only the information present in the "Context" element will be used to generate the answer.'
            "This is the unique knowledge that can be used."
            ""
            "You are provided with a <SOURCES> XML element, with a list of URLs, that will be used"
            "to generate up to three references at the end of the answer."
            'Only the information present in the "Sources" element will be used to generate the references.'
            ""
            "If the Context information doesn't lead to a good answer, don't invent anything,"
            "just say that you don't know."
            'Avoid the term "context" in the answer.'
            "Avoid repeating the question in the answer."
            ""
            "All the information is in mediawiki format, convert it to markdown in the answer."
            'Specially the links, convert them from the mediawiki format "[url|text]" to the markdown "[text](url)".'
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
        messages = (
            system_prompt,
            MessagesPlaceholder("history", optional=True),
            user_message,
        )
        chat_prompt = ChatPromptTemplate.from_messages(messages)
    finally:
        return chat_prompt


async def retrieve(state: RagState, config: RunnableConfig) -> RagState:
    """Retrieve the best matches from the indexed database.

    Here we'll be using Milvus hybrid search that performs a vector search (dense, embeddings)
    and a BM25 search (sparse, full text). And then will rerank results with the weighted
    reranker.
    """
    # Note that here we are using the Milvus own library instead of the LangChain one because
    # the LangChain one doesn't support many of the features used here.
    assert ("configurable" in config)
    embeddings = OpenAIEmbeddings(
        model=config["configurable"]["embedding_model"],
        dimensions=config["configurable"]["embedding_dimension"]
    )
    query_embedding = embeddings.embed_query(state["question"])

    milvus = MilvusClient("http://localhost:19530")

    # TODO: Make a bunch of the defaults used here configurable.
    dense_search_limit = 15
    sparse_search_limit = 15
    sparse_search_drop_ratio = 0.2
    hybrid_rerank_limit = 15
    rerank_weights = (0.7, 0.3)

    # Define the dense search and its parameters.
    dense_search_params = {
        "metric_type": "IP",
        "params": {
            "ef": dense_search_limit,
        }
    }
    dense_search = AnnSearchRequest(
        [query_embedding], "dense_vector", dense_search_params, limit=dense_search_limit,
    )

    # Define the sparse search and its parameters.
    sparse_search_params = {
        "metric_type": "BM25",
        "drop_ratio_search": sparse_search_drop_ratio,
    }
    sparse_search = AnnSearchRequest(
        [state["question"]], "sparse_vector", sparse_search_params, limit=sparse_search_limit,
    )

    # Perform the hybrid search.
    retrieved_docs = milvus.hybrid_search(
        config["configurable"]["collection_name"],
        [dense_search, sparse_search],
        WeightedRanker(*rerank_weights),
        limit=hybrid_rerank_limit,
        output_fields=[
            "id",
            "title",
            "text",
            "source",
            "doc_id",
            "doc_title",
            "doc_hash",
            "parent",
            "children",
            "previous",
            "next",
            "relations",
            "page_id",
        ]
    )
    milvus.close()

    # TODO: Return only the docs which distance is below the cutoff.
    # distance_cutoff = config["configurable"]["search_distance_cutoff"]
    # return {"vector_search": [doc for doc in retrieved_docs[0] if doc["distance"] >= distance_cutoff]}
    state["vector_search"] = retrieved_docs[0]
    return state


async def optimise(state: RagState, config: RunnableConfig) -> RagState:
    """Optimise the retrieved documents to build the context for the answer.

    First, we'll weight the elements retrieved by "popularity" (how many times they are mentioned
    by other elements in the retrieved docs). Then, we'll build the context by using some strategy,
    like Parent, Own and Children (poc) elements.
    """
    # TODO: Play with alternative strategies, we also have prev/nex, related, ...
    if not state["vector_search"]:  # No results, no context.
        state["context"] = []
        state["sources"] = []
        return state

    assert ("configurable" in config)
    top = 5  # TODO: Make this part of the state, configurable.
    # Let's count how many times each element is mentioned as id, parent, children, previous or next,
    # making a dictionary with the counts. They will be weighted differently, following this order:
    # id (weight 5), children (weight 3), parent (weight 1), previous (weight 1), next (weight 1)
    # multiplied by their original distance and with a decay by position applied.
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
                    element_counts[el] += distance * decay * (5 if element == "id"
                        else 3 if element == "children"
                        else 1)
                # Else for sure it's a list/array like (iterable) element, so we'll count each element in the list.
                else:
                    for el in doc["entity"][element]:
                        if isinstance(el, str):
                            if el not in element_counts:
                                element_counts[el] = 0
                            element_counts[el] += distance * decay * (5 if element == "id"
                                else 3 if element == "children"
                                else 1)

    # Sort them by the weighted popularity results, we'll build the context following this order.
    sorted_items = sorted(element_counts.items(), key=lambda item: item[1], reverse=True)
    logger.debug(f"Sorted items: {pprint.PrettyPrinter(2).pformat(sorted_items)}")

    # TODO: Add support for other variations, including prev/next, related, etc.
    # Build the POC (parent, own, children) context
    new_context, new_sources = build_poc_context(
        retrieved_docs=state["vector_search"],
        sorted_items=sorted_items,
        collection_name=config["configurable"]["collection_name"],
        top=top
    )

    state["context"] = new_context
    state["sources"] = new_sources
    return state


def build_poc_context(retrieved_docs, sorted_items, collection_name: str, top=5) -> list[list[str]]:
    """Given the originally retrieved docs and the sorted, weighted items, build the rag final context.

    POC: Build the new context by using Parent, Own and Children elements.
    """
    context_list = []
    sources_list = []
    not_retrieved = []
    current = 0
    while current < top:
        # Let's examine the element in the sorted_items list.
        if not (element_id := sorted_items[current][0]):
            break
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
            # If not, let's retrieve it from the milvus collection.
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

    milvus = MilvusClient("http://localhost:19530")

    # Let's find in the collection, the missing elements and get their titles and texts.
    missing_docs_db = milvus.query(
        collection_name,
        ids=context_missing,
        output_fields=["id", "title", "text"])
    missing_docs = {doc["id"]: f"{doc["title"]}\n\n{doc["text"]}" for doc in missing_docs_db}
    milvus.close()
    return missing_docs


async def generate(state: RagState, config: RunnableConfig) -> RagState | dict:
    """Generate the final answer for the question.

    This is the final generation step where the prompt, the chat history and the context
    are used to generate the final answer.
    """
    assert ("configurable" in config)
    llm = ChatOpenAI(
        model=config["configurable"]["llm_model"],
        max_completion_tokens=config["configurable"]["max_completion_tokens"],
        top_p=config["configurable"]["top_p"],
        temperature=config["configurable"]["temperature"],
    )

    docs_content = "\n\n".join(f"{doc}" for doc in state["context"])
    sources_content = "\n".join(f"* {source}" for source in state["sources"])

    chat_prompt = load_prompts_for_rag(config["configurable"]["prompt_name"])
    chat = await chat_prompt.ainvoke({
        "task_def": config["configurable"]["task_def"],
        "kb_name": config["configurable"]["kb_name"],
        "kb_url": config["configurable"]["kb_url"],
        "context": docs_content,
        "sources": sources_content,
        "question": state["question"],
        "history": state["history"]
    })

    response = await llm.ainvoke(chat, config)

    return {"answer": response.content}
