"""
src/graph/builder.py — LangGraph StateGraph assembly.

This module wires together all agent nodes, edges, and conditional routes
into a compiled LangGraph application.

LangGraph APIs demonstrated:
  - StateGraph           : graph definition class
  - START / END          : sentinels for entry and exit points
  - add_node()           : register an async node function
  - add_edge()           : unconditional directed edge
  - add_conditional_edges(): route to different nodes based on state
  - RetryPolicy          : automatic retry on transient errors
  - MemorySaver          : in-process checkpointer for session persistence
  - RunnableConfig       : thread-through configuration (thread_id, etc.)
  - graph.ainvoke()      : async single-result execution
  - graph.astream()      : async streaming execution (yields node outputs)
  - graph.get_state()    : inspect persisted state for a thread
  - graph.get_state_history(): full snapshot history for a thread

Session / horizontal scaling:
  - Each session gets a unique `thread_id` in config["configurable"]
  - Swapping MemorySaver → AsyncSqliteSaver / AsyncPostgresSaver enables
    multi-process / multi-host scaling with shared external state
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy

# ── Import all agent nodes ───────────────────────────────────────────────────
from src.agents.audit_agent import audit_node
from src.agents.auth_agent import auth_node
from src.agents.intent_agent import intent_node
from src.agents.nlp_to_sql_agent import nlp_to_sql_node
from src.agents.query_agent import query_node
from src.agents.rate_limit_agent import rate_limit_node
from src.agents.response_agent import response_node
from src.agents.sql_security_agent import sql_security_node
from src.agents.state import ClaimSystemState, ErrorCode, Intent

# ── Checkpointer ─────────────────────────────────────────────────────────────
# MemorySaver stores checkpoints in-process (suitable for single-instance dev).
#
# For production horizontal scaling, swap to an external checkpointer:
#
#   from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
#   checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")
#
#   from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#   checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)

_checkpointer = MemorySaver()


# ── Conditional edge functions (routers) ─────────────────────────────────────
# These functions READ state but never modify it.
# They return the NAME of the next node as a string (or END).

def route_after_rate_limit(state: ClaimSystemState) -> str:
    """After rate_limit_node: proceed to auth or short-circuit to audit."""
    if not state.get("rate_limit_ok", True):
        return "audit"
    return "auth"


def route_after_auth(state: ClaimSystemState) -> str:
    """After auth_node: proceed to intent or short-circuit to audit."""
    if state.get("auth_status") != "ok":
        return "audit"
    return "intent"


def route_after_intent(state: ClaimSystemState) -> str:
    """
    After intent_node: route to the right handler.

    - DB-required intents → nlp_to_sql (needs a database query)
    - Conversational intents → response directly (no DB)
    """
    intent = state.get("intent", Intent.UNKNOWN)
    if intent in Intent.DB_REQUIRED:
        return "nlp_to_sql"
    return "response"


def route_after_sql_security(state: ClaimSystemState) -> str:
    """After sql_security_node: execute query or short-circuit to response."""
    if not state.get("sql_safe", False):
        return "response"
    return "query"


def route_after_query(state: ClaimSystemState) -> str:
    """After query_node: always format the response."""
    return "response"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Assemble and compile the full claim-status LangGraph.

    Returns a compiled CompiledGraph (Runnable) ready for ainvoke/astream.
    """
    graph = StateGraph(ClaimSystemState)

    # ── Register nodes ────────────────────────────────────────────────────
    # RetryPolicy on query_node retries up to 3× on transient DB errors.
    graph.add_node("rate_limit", rate_limit_node)
    graph.add_node("auth",       auth_node)
    graph.add_node("intent",     intent_node)
    graph.add_node("nlp_to_sql", nlp_to_sql_node)
    graph.add_node("sql_security", sql_security_node)
    graph.add_node(
        "query",
        query_node,
        retry=RetryPolicy(max_attempts=3, backoff_factor=1.5),
    )
    graph.add_node("response",   response_node)
    graph.add_node("audit",      audit_node)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.add_edge(START, "rate_limit")

    # ── Conditional edges ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "rate_limit",
        route_after_rate_limit,
        {"auth": "auth", "audit": "audit"},
    )
    graph.add_conditional_edges(
        "auth",
        route_after_auth,
        {"intent": "intent", "audit": "audit"},
    )
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {"nlp_to_sql": "nlp_to_sql", "response": "response"},
    )
    graph.add_conditional_edges(
        "sql_security",
        route_after_sql_security,
        {"query": "query", "response": "response"},
    )

    # ── Unconditional edges ───────────────────────────────────────────────
    graph.add_edge("nlp_to_sql",   "sql_security")
    graph.add_edge("query",        "response")
    graph.add_edge("response",     "audit")
    graph.add_edge("audit",        END)

    # ── Compile with checkpointer ─────────────────────────────────────────
    compiled = graph.compile(checkpointer=_checkpointer)
    return compiled


# ── Application factory (module-level singleton) ──────────────────────────────
# Import `get_app()` wherever you need the compiled graph.

_app = None


def get_app() -> Any:
    """Return the compiled LangGraph application (singleton)."""
    global _app
    if _app is None:
        _app = build_graph()
    return _app


# ── High-level async query interface ─────────────────────────────────────────

async def query_claim_status(
    *,
    user_token: str,
    user_query: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    High-level interface for querying the claim status system.

    Args:
        user_token:  JWT bearer token.
        user_query:  Natural-language question from the user.
        session_id:  Reuse an existing session for multi-turn conversations.
                     If None, a new session is started.

    Returns:
        dict with keys:
          - response_text : the final user-facing answer
          - intent        : classified intent
          - session_id    : session ID (reuse for follow-up questions)
          - error_code    : set if something went wrong (else None)
    """
    session_id = session_id or str(uuid.uuid4())
    app = get_app()

    initial_state: ClaimSystemState = {
        "session_id": session_id,
        "user_token": user_token,
        "raw_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "rate_limit_ok": False,
        "rate_limit_remaining": 0,
        "auth_status": "pending",
        "user_id": None,
        "user_email": None,
        "user_role": "",
        "policy_ids": [],
        "intent": "",
        "entities": {},
        "intent_confidence": 0.0,
        "generated_sql": "",
        "sql_params": [],
        "sql_tables_used": [],
        "sql_explanation": "",
        "sql_safe": False,
        "sql_rejection_reason": "",
        "query_results": [],
        "query_row_count": 0,
        "query_error": None,
        "response_text": "",
        "error": None,
        "error_code": None,
        "audit_trail": [],
        "request_start_time": time.monotonic(),
        "response_time_ms": None,
    }

    config: RunnableConfig = {
        "configurable": {
            # thread_id groups all state snapshots for one session.
            # The same thread_id on a subsequent call restores prior state
            # (enabling multi-turn conversation continuity).
            "thread_id": session_id,
        },
        # Maximum node-to-node hops before aborting (prevents infinite loops)
        "recursion_limit": 25,
    }

    final_state = await app.ainvoke(initial_state, config=config)

    return {
        "response_text": final_state.get("response_text", "No response generated."),
        "intent": final_state.get("intent"),
        "session_id": session_id,
        "error_code": final_state.get("error_code"),
        "response_time_ms": final_state.get("response_time_ms"),
    }


async def stream_claim_status(
    *,
    user_token: str,
    user_query: str,
    session_id: str | None = None,
):
    """
    Streaming variant: yields one dict per node as it completes.

    Useful for showing real-time progress indicators in a UI.

    Usage:
        async for update in stream_claim_status(token=..., query=...):
            print(update)   # {"node": "intent", "state_delta": {...}}
    """
    session_id = session_id or str(uuid.uuid4())
    app = get_app()

    initial_state: ClaimSystemState = {
        "session_id": session_id,
        "user_token": user_token,
        "raw_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "rate_limit_ok": False,
        "rate_limit_remaining": 0,
        "auth_status": "pending",
        "user_id": None,
        "user_email": None,
        "user_role": "",
        "policy_ids": [],
        "intent": "",
        "entities": {},
        "intent_confidence": 0.0,
        "generated_sql": "",
        "sql_params": [],
        "sql_tables_used": [],
        "sql_explanation": "",
        "sql_safe": False,
        "sql_rejection_reason": "",
        "query_results": [],
        "query_row_count": 0,
        "query_error": None,
        "response_text": "",
        "error": None,
        "error_code": None,
        "audit_trail": [],
        "request_start_time": time.monotonic(),
        "response_time_ms": None,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    async for chunk in app.astream(initial_state, config=config, stream_mode="updates"):
        node_name = next(iter(chunk))
        yield {"node": node_name, "state_delta": chunk[node_name], "session_id": session_id}
