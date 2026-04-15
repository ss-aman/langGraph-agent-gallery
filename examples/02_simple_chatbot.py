"""
Example 2: Simple Chatbot with Conversation Memory
===================================================

Build a multi-turn chatbot whose conversation history persists across turns
— so the model always "remembers" what was said earlier.

Graph structure:
  [START] --> chatbot_node --> [END]
                  ^
                  | (each new message is appended; full history is passed)

Key concepts introduced:
  - MessagesState   : built-in LangGraph state that manages a list of messages
  - add_messages    : a reducer that appends new messages instead of replacing
  - HumanMessage / AIMessage: LangChain message types
  - ChatAnthropic   : the Claude LLM wrapper from langchain-anthropic
  - Multi-turn loop : calling invoke() repeatedly on the same graph

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python examples/02_simple_chatbot.py
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()  # loads ANTHROPIC_API_KEY from a .env file if present


# ---------------------------------------------------------------------------
# 1. Initialise the LLM
# ---------------------------------------------------------------------------
# MessagesState uses a built-in "messages" key whose value is a list of
# LangChain BaseMessage objects (HumanMessage, AIMessage, SystemMessage …).
# The add_messages reducer automatically appends new messages to the list
# rather than overwriting it — that's how conversation history is kept.

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.7,
)


# ---------------------------------------------------------------------------
# 2. Define the Single Node
# ---------------------------------------------------------------------------

def chatbot_node(state: MessagesState) -> dict:
    """
    Pass the full conversation history to the LLM and return its reply.

    state["messages"] contains every message exchanged so far.
    We append the new AI reply so it becomes part of the history for the
    next turn.
    """
    response = llm.invoke(state["messages"])
    # Returning {"messages": [response]} triggers add_messages, which
    # *appends* the AIMessage to the existing list.
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 3. Build the Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 4. Run a Multi-Turn Conversation
# ---------------------------------------------------------------------------

def chat(app, history: list, user_input: str) -> tuple[str, list]:
    """
    Send one user message, get the AI reply, and update history.

    Args:
        app     : compiled LangGraph app
        history : current list of messages (HumanMessage / AIMessage)
        user_input : the user's new message text

    Returns:
        (ai_reply_text, updated_history)
    """
    history = history + [HumanMessage(content=user_input)]
    result = app.invoke({"messages": history})
    updated_history = result["messages"]
    ai_reply = updated_history[-1].content
    return ai_reply, updated_history


if __name__ == "__main__":
    app = build_graph()
    conversation_history: list = []

    print("=== Simple Chatbot (type 'quit' to exit) ===\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        reply, conversation_history = chat(app, conversation_history, user_input)
        print(f"Bot: {reply}\n")

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Add a SystemMessage at the start of conversation_history to give the
    #    bot a persona: SystemMessage(content="You are a pirate assistant.")
    # 2. Limit history length by slicing: history[-10:] before calling invoke.
    # 3. Print the full history at the end to see all messages.
