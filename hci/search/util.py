#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

""" Util functions to use langgraph to conduct simple searches against the indexed database."""

import logging
from typing import TypedDict

from hci.server.util import Message

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph.state import StateGraph, CompiledStateGraph, START

from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

logger = logging.getLogger(__name__)


# Define the configuration that the graph will use.
# TODO: Check how this is used/validated and how to better integrate in the OpenAI API (FastAPI).
class Config(TypedDict):
    prompt_name: str
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    search_distance_cutoff: float
    max_completion_tokens: int
    top_p: float
    temperature: float
    stream: bool

# Define the overall state for the graph
class OverallState(TypedDict):
    history: list[BaseMessage]
    question: str
    context: list[dict]
    answer: str | None


# To track intermediate information during the graph execution
class InternalState(OverallState, TypedDict):
    vector_search: list[dict]

def build_graph() -> CompiledStateGraph:
    """Build the graph for the langgraph search."""

    graph_builder = StateGraph(OverallState, Config).add_sequence([prepare, retrieve, optimise, generate])
    graph_builder.add_edge(START, "prepare")
    graph = graph_builder.compile()
    return graph

def load_prompts_for_rag(prompt_name: str, messages_history: list[BaseMessage]) -> ChatPromptTemplate:
    """ Load the prompts for the RAG model."""

    # Try to load the prompt from langsmith, falling back to hardcoded one.
    # Note this (to pull the prompt) requires langsmith to be available and configured with:
    # LANGSMITH_ENDPOINT = "https://xxx.api.smith.langchain.com"
    # LANGSMITH_API_KEY = "<your langsmith api key>"
    # Optionally, automatic tracing can be enabled with:
    # LANGSMITH_TRACING = True
    chat_prompt = ChatPromptTemplate([])
    logger.debug(f"Loading the prompt {prompt_name} from LangSmith.")
    try:
        prompt_name="mediawiki-rag"
        chat_prompt = hub.pull(prompt_name)
    except Exception as e:
        logger.warning(f"Error loading the prompt {prompt_name} from LangSmith: {e}")
        # Use the manual prompt building instead.
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks related to Moodle user documentation."
            "The sources for you knowledge are the \"Moodle Docs\", available at https://docs.moodle.org"
            "Avoid the term \"context\" in the answer."
            "Avoid repeating the question in the answer."
            "Try to answer with a few phrases in a concise and clear way."
            "If the user asks for more details or explanations the answer can be longer."
            "If you don't know the answer, just say that you don't know. Never invent an answer."
            "Use the provided context to answer the question only if the context is relevant."
            "The provided context is in mediawiki format, try to convert that to markdown in the answer."
        )
        user_message = HumanMessagePromptTemplate.from_template(
            "Question: {question} \n\nContext: {context}\n\nAnswer:"
        )
        messages = (
            system_prompt,
            MessagesPlaceholder("history", optional=True),
            user_message,
        )
        chat_prompt = ChatPromptTemplate.from_messages(messages)
    finally:
        return chat_prompt

def prepare(state: OverallState) -> dict:
    """Given the overall state, prepare the internal state for the graph."""
    return InternalState()

def retrieve(state: InternalState, config: RunnableConfig) -> dict:
    """Retrieve the best matches from the indexed database.

    Here we'll be using Milvus hybrid search that performs a vector search (dense, embeddings)
    and a BM25 search (sparse, full text). And then will rerank results with the weighted
    reranker.
    """

    # Note that here we are using the Milvus own library instead of the LangChain one because
    # the LangChain one doesn't support many of the features used here.

    embeddings = OpenAIEmbeddings(
        model=config["configurable"].get("embedding_model"),
        dimensions=config["configurable"].get("embedding_dimension")
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
        config["configurable"].get("collection_name"),
        [dense_search, sparse_search],
        WeightedRanker(*rerank_weights),
        limit=hybrid_rerank_limit,
        output_fields=[
            "id",
            "title",
            "text",
            "source",
            "doc_id",
            "doc_hash",
            "doc_title",
            "parent",
            "children",
            "previous",
            "next",
        ]
    )
    milvus.close()

    # Return only the docs which distance is below the cutoff.
    #distance_cutoff = config["configurable"].get("search_distance_cutoff")
    #return {"vector_search": [doc for doc in retrieved_docs[0] if doc["distance"] >= distance_cutoff]}
    return {"vector_search": retrieved_docs[0]}

def optimise(state: InternalState, config: RunnableConfig) -> dict:
    # Only if there are vector search results.
    if not state["vector_search"]:
        return {"context": []}

    top = 5 # TODO: Make this part of the state.
    # Let's count how many times each element is mentioned as id, parent, children, previous or next,
    # making a dictionary with the counts. They will be weighted differently, following this order:
    # id (weight 10) > parent (weight 5) > children (weight 2) > previous (weight 1) > next (weight 1)
    # multiplied by their original distance.
    element_counts = {}
    for doc in state["vector_search"]:
        distance = doc["distance"]
        for element in ["id", "parent", "children", "previous", "next"]:
            if element in doc["entity"]:
                el = doc["entity"][element]
                # Empty (None, list) elements are not counted.
                if not el:
                    continue
                # If it's a single string element, we'll count it.
                if el and isinstance(el, str):
                    if el not in element_counts:
                        element_counts[el] = 0
                    element_counts[el] += distance * (5 if element == "id"
                        else 3 if element == "parent"
                        else 2 if element == "children"
                        else 1)
                # Else for sure it's a list/array like (iterable) element, so we'll count each element in the list.
                else:
                    for el in doc["entity"][element]:
                        if isinstance(el, str):
                            if el not in element_counts:
                                element_counts[el] = 0
                            element_counts[el] += distance * (5 if element == "id"
                                else 3 if element == "parent"
                                else 2 if element == "children"
                                else 1)
    # Sort them by the counts.
    sorted_items = sorted(element_counts.items(), key=lambda item: item[1], reverse=True)
    # Build the POC (parent, own, children) context
    # TODO: Add other variations, including prev/next, related, etc.
    new_context = build_poc_context(
        retrieved_docs=state["vector_search"],
        sorted_items=sorted_items,
        collection_name=config["configurable"].get("collection_name"),
        top=top
    )
    return {"context": new_context}

def build_poc_context(retrieved_docs, sorted_items, collection_name: str, top=5, ):
    """ Given the originally retrieved docs and the sorted (weighted items, build the rag final context.

    POC: Build the new context by using Parent, Own and Children elements."""
    context_list = []
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

    return retrieve_all_elements(retrieved_docs, context_list, collection_name)

def retrieve_all_elements(retrieved_docs, context_list, collection_name: str):
    """ Given the already built content_List, let's retrieve all the texts for the elements in the list."""

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
    """ Given the missing elements, let's retrieve them from the vector store."""

    if not context_missing:
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


def generate(state: InternalState, config: RunnableConfig) -> OverallState:
    """Generate the answer from the retrieved documents."""

    llm = ChatOpenAI(
        model=config["configurable"].get("llm_model"),
        max_tokens=config["configurable"].get("max_completion_tokens"),
        top_p=config["configurable"].get("top_p"),
        temperature=config["configurable"].get("temperature"),
    )

    docs_content = "\n\n".join(f"{doc}" for doc in state["context"])

    chat_prompt = load_prompts_for_rag(
        config["configurable"].get("prompt_name"),
        state["history"])
    chat = chat_prompt.invoke({"context": docs_content, "question": state["question"], "history": state["history"]})

    response = llm.invoke(chat, config)

    return {"answer": response.content}
