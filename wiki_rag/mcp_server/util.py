#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to generate the graphs used by the MCP server."""

import logging

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from wiki_rag.search.util import ContextSchema, RagState, optimise, retrieve

logger = logging.getLogger(__name__)


def build_retrieve_graph(context: ContextSchema) -> CompiledStateGraph:
    """Build the retrieve only graph for the retrieve MCP tool."""
    graph_builder = StateGraph(RagState, ContextSchema).add_sequence([
        retrieve,
    ])
    graph_builder.add_edge(START, "retrieve")
    graph: CompiledStateGraph = graph_builder.compile().with_config(
        {"callbacks": [context["langfuse_callback"]]} if context["langfuse_callback"] else None
    )   # pyright: ignore[reportAssignmentType]. Note this is correct, but for some reason, pyright is not able.

    return graph


def build_optimise_graph(context: ContextSchema) -> CompiledStateGraph:
    """Build the retrieve and optimise graph for the optimise MCP tool."""
    graph_builder = StateGraph(RagState, ContextSchema).add_sequence([
        retrieve,
        optimise,
    ])
    graph_builder.add_edge(START, "retrieve")
    graph: CompiledStateGraph = graph_builder.compile().with_config(
        {"callbacks": [context["langfuse_callback"]]} if context["langfuse_callback"] else None
    )   # pyright: ignore[reportAssignmentType]. Note this is correct, but for some reason, pyright is not able.

    return graph
