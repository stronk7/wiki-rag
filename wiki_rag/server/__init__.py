#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.server package."""
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

# The configuration for the graph.
config: RunnableConfig | None = None

# The graph that will be executed for the server chat completion requests.
graph: CompiledStateGraph | None = None
