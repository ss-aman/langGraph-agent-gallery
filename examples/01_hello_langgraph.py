"""
Example 1: Hello LangGraph - Your First Graph
=============================================

The simplest possible LangGraph example — no LLM required.

We build a text-processing pipeline:
  [START] --> greet_node --> shout_node --> [END]

Key concepts introduced:
  - StateGraph    : the main class used to build a graph
  - State (TypedDict): a typed dictionary that carries data between nodes
  - Nodes         : plain Python functions that read and update state
  - Edges         : connections that define the flow between nodes
  - compile()     : converts the graph definition into a runnable object
  - invoke()      : runs the graph with an initial state

Run:
    python examples/01_hello_langgraph.py
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. Define State
# ---------------------------------------------------------------------------
# State is the "memory" of your graph — every node reads from it and can
# write back to it. Use Python's TypedDict for type safety and clarity.

class State(TypedDict):
    message: str       # the text that flows through the graph
    step_log: list     # a record of which steps were executed


# ---------------------------------------------------------------------------
# 2. Define Nodes
# ---------------------------------------------------------------------------
# A node is just a Python function that:
#   - accepts the current State as its only argument
#   - returns a dict with the fields it wants to update
# Fields you don't return are left unchanged.

def greet_node(state: State) -> dict:
    """Prepend a greeting to the message."""
    print(f"[greet_node] received: '{state['message']}'")

    updated_message = f"Hello! {state['message']}"
    updated_log = state["step_log"] + ["greet_node"]

    return {"message": updated_message, "step_log": updated_log}


def shout_node(state: State) -> dict:
    """Convert the message to uppercase to 'shout' it."""
    print(f"[shout_node] received: '{state['message']}'")

    updated_message = state["message"].upper()
    updated_log = state["step_log"] + ["shout_node"]

    return {"message": updated_message, "step_log": updated_log}


# ---------------------------------------------------------------------------
# 3. Build the Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    # Initialise StateGraph with our State schema
    graph = StateGraph(State)

    # Add nodes — first arg is the node name, second is the function
    graph.add_node("greet", greet_node)
    graph.add_node("shout", shout_node)

    # Add edges — define the execution order
    graph.add_edge(START, "greet")   # graph starts at "greet"
    graph.add_edge("greet", "shout") # after "greet", go to "shout"
    graph.add_edge("shout", END)     # after "shout", the graph finishes

    # compile() validates the graph and returns a Runnable
    return graph.compile()


# ---------------------------------------------------------------------------
# 4. Run It
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    # Provide an initial state to kick off the graph
    initial_state: State = {
        "message": "welcome to LangGraph",
        "step_log": [],
    }

    print("=== Running the graph ===")
    print(f"Input : {initial_state['message']!r}\n")

    result = app.invoke(initial_state)

    print(f"\nOutput  : {result['message']!r}")
    print(f"Steps ran: {result['step_log']}")

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Add a third node (e.g. add punctuation) and wire it into the graph.
    # 2. Change the State to carry an integer counter; have each node increment it.
    # 3. Print the graph structure: print(app.get_graph().draw_ascii())
