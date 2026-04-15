"""
Example 4: Conditional Routing (Smart Triage Agent)
=====================================================

Real agents rarely follow a single fixed path — they branch based on context.
This example builds a customer-support triage bot that:
  1. Classifies the incoming message into a category
  2. Routes it to the right specialist handler
  3. Each handler crafts a tailored response

Graph structure:
  [START]
     |
  classify_node
     |
  (router function — picks branch based on state["category"])
     |
  ┌──┴────┬──────────┬──────────┐
  billing  tech    general    escalate
     └──┬──┘          │           │
        └─────────────┴───────────┘
                      |
                  respond_node
                      |
                    [END]

Key concepts introduced:
  - add_conditional_edges() : route to *different* nodes based on a function
  - Router function         : a plain Python function (not a node) that reads
                              state and returns a node name as a string
  - Branching and merging   : multiple paths can lead back to one node
  - Structured LLM output   : using with_structured_output() to get a typed
                              classification result

Setup:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python examples/04_conditional_routing.py
"""

import os
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Define State
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    user_message: str                          # the incoming customer message
    category: str                              # assigned by classify_node
    response: str                              # final reply built by a handler


# ---------------------------------------------------------------------------
# 2. Define the Category Schema for Structured Output
# ---------------------------------------------------------------------------
# Pydantic model lets us extract a typed, validated classification from
# the LLM instead of parsing raw text ourselves.

class MessageCategory(BaseModel):
    category: Literal["billing", "technical", "general", "escalate"] = Field(
        description=(
            "Message category: "
            "'billing' for payment/invoice questions, "
            "'technical' for bugs/errors/how-to, "
            "'general' for other enquiries, "
            "'escalate' for complaints or urgent/angry messages."
        )
    )
    reason: str = Field(description="One sentence explaining why this category was chosen.")


# ---------------------------------------------------------------------------
# 3. Initialise LLMs
# ---------------------------------------------------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
classifier_llm = llm.with_structured_output(MessageCategory)


# ---------------------------------------------------------------------------
# 4. Define Nodes
# ---------------------------------------------------------------------------

def classify_node(state: SupportState) -> dict:
    """Classify the incoming message into a support category."""
    print(f"\n[classify_node] Classifying: '{state['user_message']}'")

    result: MessageCategory = classifier_llm.invoke([
        SystemMessage(content="You are a customer support triage agent."),
        HumanMessage(content=state["user_message"]),
    ])

    print(f"[classify_node] → category='{result.category}', reason='{result.reason}'")
    return {"category": result.category}


def billing_handler(state: SupportState) -> dict:
    """Handle billing and payment enquiries."""
    print("[billing_handler] handling billing query...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a billing specialist. Be concise, friendly, and helpful. "
            "Focus on payment, invoices, refunds, and subscription questions."
        )),
        HumanMessage(content=state["user_message"]),
    ])
    return {"response": response.content}


def technical_handler(state: SupportState) -> dict:
    """Handle technical/product issues."""
    print("[technical_handler] handling technical query...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a technical support engineer. Provide clear, step-by-step "
            "troubleshooting guidance. Ask clarifying questions if needed."
        )),
        HumanMessage(content=state["user_message"]),
    ])
    return {"response": response.content}


def general_handler(state: SupportState) -> dict:
    """Handle general / misc enquiries."""
    print("[general_handler] handling general query...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a friendly general support agent. Answer concisely and "
            "helpfully. If you cannot help, direct the user to the right team."
        )),
        HumanMessage(content=state["user_message"]),
    ])
    return {"response": response.content}


def escalate_handler(state: SupportState) -> dict:
    """Handle urgent / escalation cases without LLM latency."""
    print("[escalate_handler] escalating to human agent...")
    return {
        "response": (
            "I can see this is an urgent matter and I want to make sure you get "
            "the best help possible. I'm escalating your case to a senior support "
            "agent who will contact you within 1 hour. Thank you for your patience."
        )
    }


# ---------------------------------------------------------------------------
# 5. Router Function (not a node — used only in add_conditional_edges)
# ---------------------------------------------------------------------------

def route_by_category(state: SupportState) -> str:
    """
    Return the name of the next node based on the assigned category.

    This function is called *between* nodes by LangGraph's conditional edge
    mechanism. It must return a valid node name (or END).
    """
    routing_map = {
        "billing": "billing_handler",
        "technical": "technical_handler",
        "general": "general_handler",
        "escalate": "escalate_handler",
    }
    return routing_map.get(state["category"], "general_handler")


# ---------------------------------------------------------------------------
# 6. Build the Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(SupportState)

    # Add the classifier and all four handler nodes
    graph.add_node("classify", classify_node)
    graph.add_node("billing_handler", billing_handler)
    graph.add_node("technical_handler", technical_handler)
    graph.add_node("general_handler", general_handler)
    graph.add_node("escalate_handler", escalate_handler)

    # Entry: always start with classification
    graph.add_edge(START, "classify")

    # After classification, branch to the right handler
    graph.add_conditional_edges(
        "classify",          # source node
        route_by_category,   # function that decides the next node
        # (optional) explicit map for documentation / validation:
        {
            "billing_handler": "billing_handler",
            "technical_handler": "technical_handler",
            "general_handler": "general_handler",
            "escalate_handler": "escalate_handler",
        },
    )

    # All handlers lead to END
    for handler in ["billing_handler", "technical_handler", "general_handler", "escalate_handler"]:
        graph.add_edge(handler, END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 7. Run It
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    test_messages = [
        "I was charged twice for my subscription last month. Please help!",
        "My app keeps crashing whenever I try to upload a file larger than 10 MB.",
        "What are your business hours?",
        "This is absolutely unacceptable! I've been waiting 3 weeks for a refund!",
    ]

    print("=== Customer Support Triage Agent ===")
    for message in test_messages:
        print(f"\n{'='*60}")
        print(f"Customer: {message}")
        result = app.invoke({"user_message": message, "category": "", "response": ""})
        print(f"Category : [{result['category'].upper()}]")
        print(f"Response : {result['response']}")

    # -----------------------------------------------------------------------
    # What to explore next:
    # -----------------------------------------------------------------------
    # 1. Add a "sentiment" field to State and set it in classify_node.
    # 2. Create a priority queue by routing angry messages to escalate_handler
    #    regardless of category.
    # 3. Log all interactions to a file by adding a final "log_node" before END.
