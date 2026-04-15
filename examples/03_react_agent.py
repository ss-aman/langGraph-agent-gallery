"""
Example 3: ReAct Agent with Tools
==================================

ReAct (Reasoning + Acting) is the most common agent pattern:
the LLM reasons about a problem, decides which tool to call, observes the
result, and repeats until it has enough information to answer.

Graph structure:
  [START] --> agent_node ---(has tool calls?)---> tool_node
                  ^                                    |
                  |____________(loop back)_____________|
                  |
               (no tool calls → END)

Key concepts introduced:
  - @tool decorator    : turn any Python function into an LLM-callable tool
  - ToolNode           : a built-in LangGraph node that executes tool calls
  - tools_condition    : a built-in conditional edge — routes to ToolNode if
                         the last message contains tool calls, else to END
  - bind_tools()       : attaches tool schemas to the LLM so it knows what's
                         available
  - Agentic loops      : how the graph keeps cycling until work is done

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python examples/03_react_agent.py
"""

import math
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Define Tools
# ---------------------------------------------------------------------------
# The @tool decorator reads the function's docstring and type hints to
# generate a JSON schema that the LLM uses to know when and how to call it.
# Always write a clear docstring — it's the tool's instruction manual.

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.

    Supports: +, -, *, /, ** (power), sqrt(), sin(), cos(), log(), etc.
    Example: '2 ** 10', 'sqrt(144)', '3.14 * 5 ** 2'
    """
    try:
        # We use a restricted namespace for safety
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["__builtins__"] = {}
        result = eval(expression, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words and characters in a piece of text.

    Returns a summary string with the word count and character count.
    """
    words = len(text.split())
    chars = len(text)
    return f"'{text}' has {words} word(s) and {chars} character(s)."


@tool
def reverse_text(text: str) -> str:
    """
    Reverse a string of text and return it.

    Example: 'hello' → 'olleh'
    """
    return text[::-1]


TOOLS = [calculator, word_counter, reverse_text]


# ---------------------------------------------------------------------------
# 2. Initialise the LLM with Tools Bound
# ---------------------------------------------------------------------------
# bind_tools() tells the LLM about available tools so it can decide when to
# call them and with what arguments.

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 3. Define Nodes
# ---------------------------------------------------------------------------

def agent_node(state: MessagesState) -> dict:
    """
    The reasoning step: ask the LLM what to do next.

    The LLM either:
      a) Returns a plain text answer  → graph goes to END
      b) Returns tool call(s)         → graph goes to ToolNode
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode is a prebuilt node that:
#   1. reads all tool_calls from the last AIMessage
#   2. executes each tool with the provided arguments
#   3. wraps results as ToolMessage objects and appends them to state
tool_node = ToolNode(TOOLS)


# ---------------------------------------------------------------------------
# 4. Build the Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    # tools_condition is a built-in helper:
    #   if last message has tool_calls → route to "tools"
    #   otherwise                      → route to END
    graph.add_conditional_edges("agent", tools_condition)

    # After the tools run, go back to the agent so it can reason about results
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------------------------

def run_query(app, query: str) -> str:
    """Send a query to the agent and return the final text answer."""
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    app = build_graph()

    questions = [
        "What is the square root of 1764?",
        "How many words are in the sentence 'The quick brown fox jumps over the lazy dog'?",
        "What is 2 to the power of 16?",
        "Reverse the word 'LangGraph' for me.",
        "What is sin(pi/2) rounded to 2 decimal places? Use the calculator.",
    ]

    print("=== ReAct Agent with Tools ===\n")
    for q in questions:
        print(f"Q: {q}")
        answer = run_query(app, q)
        print(f"A: {answer}\n")

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Add a new @tool (e.g. one that fetches current weather from an API).
    # 2. Ask a multi-step question that requires chaining multiple tools.
    # 3. Inspect intermediate steps: print all messages in result["messages"].
