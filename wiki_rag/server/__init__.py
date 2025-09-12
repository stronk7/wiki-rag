#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""wiki_rag.server package."""
from langgraph.graph.state import CompiledStateGraph

from wiki_rag.search.util import ContextSchema

# The configuration for the graph.
context: ContextSchema | None = None

# The graph that will be executed for the server chat completion requests.
graph: CompiledStateGraph | None = None
