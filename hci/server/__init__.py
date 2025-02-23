#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""hci.server package."""

from langgraph.graph.state import CompiledStateGraph

# The graph that will be executed for the server chat completion requests.
graph: CompiledStateGraph

# The configuration for the graph.
config: dict
