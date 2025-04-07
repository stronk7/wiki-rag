#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Util functions to generate the graphs used by the MCP server."""

import logging

from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from wiki_rag.search.util import ConfigSchema, RagState, optimise, retrieve

logger = logging.getLogger(__name__)


def build_retrieve_graph() -> CompiledStateGraph:
    """Build the retrieve only graph for the retrieve MCP tool."""
    graph_builder = StateGraph(RagState, ConfigSchema).add_sequence([
        retrieve,
    ])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


def build_optimise_graph() -> CompiledStateGraph:
    """Build the retrieve and optimise graph for the optimise MCP tool."""
    graph_builder = StateGraph(RagState, ConfigSchema).add_sequence([
        retrieve,
        optimise,
    ])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
