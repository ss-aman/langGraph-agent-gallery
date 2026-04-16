"""
src/agents/state.py — Shared LangGraph state for the claim status system.

`ClaimSystemState` is the single source of truth that flows through every
node.  TypedDict is used (not dataclass or Pydantic) because LangGraph
requires dict-compatible state objects.

Design rules:
  - Every field has a default so nodes only need to return the keys they change.
  - Sensitive fields (user_token, raw_query) are never logged directly.
  - `messages` uses Annotated + add_messages so conversation history accumulates.
  - `audit_trail` is an in-memory append-only list (written to DB by audit_node).
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ClaimSystemState(TypedDict, total=False):
    """
    Complete state schema for the claim status multi-agent graph.

    Fields marked (→ set by) show which node is responsible for populating them.
    """

    # ── Session / input ──────────────────────────────────────────────────
    session_id: str                    # unique ID for this user session
    user_token: str                    # raw JWT bearer token (never logged)
    raw_query: str                     # original natural-language query

    # Accumulates the full conversation across multi-turn sessions.
    # add_messages reducer APPENDS rather than replaces.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Rate limiting → rate_limit_node ──────────────────────────────────
    rate_limit_ok: bool                # True if request is within limits
    rate_limit_remaining: int          # tokens remaining in the bucket

    # ── Authentication → auth_node ────────────────────────────────────────
    auth_status: str                   # "pending" | "ok" | "failed"
    user_id: Optional[str]             # decoded from JWT
    user_email: Optional[str]          # decoded from JWT (masked in logs)
    user_role: str                     # "policyholder" | "agent" | "admin"
    policy_ids: list[str]              # policies this user may access (RLS)

    # ── Intent classification → intent_node ──────────────────────────────
    intent: str                        # e.g. "claim_status", "help"
    entities: dict[str, Any]           # extracted: claim_id, date_range, …
    intent_confidence: float           # 0.0–1.0

    # ── NLP→SQL → nlp_to_sql_node ────────────────────────────────────────
    generated_sql: str                 # raw SQL from LLM
    sql_params: list[Any]              # parameterised values (user_id first)
    sql_tables_used: list[str]         # tables referenced by the query
    sql_explanation: str               # human-readable description

    # ── SQL validation → sql_security_node ───────────────────────────────
    sql_safe: bool                     # True if validation passed
    sql_rejection_reason: str          # populated if sql_safe == False

    # ── Query execution → query_node ─────────────────────────────────────
    query_results: list[dict[str, Any]]
    query_row_count: int
    query_error: Optional[str]

    # ── Response formatting → response_node ──────────────────────────────
    response_text: str                 # final user-facing message

    # ── Error tracking (any node can set these) ───────────────────────────
    error: Optional[str]               # internal error description
    error_code: Optional[str]          # machine-readable code (e.g. AUTH_FAILED)

    # ── Audit / telemetry ─────────────────────────────────────────────────
    audit_trail: list[dict[str, Any]]  # append-only event log (in-memory)
    request_start_time: float          # monotonic clock at graph entry
    response_time_ms: Optional[int]    # computed by audit_node


# ── Intent constants ──────────────────────────────────────────────────────────
class Intent:
    CLAIM_STATUS = "claim_status"
    CLAIM_HISTORY = "claim_history"
    CLAIM_DETAILS = "claim_details"
    POLICY_INFO = "policy_info"
    HELP = "help"
    GREETING = "greeting"
    UNKNOWN = "unknown"

    # Intents that require a database query
    DB_REQUIRED = {CLAIM_STATUS, CLAIM_HISTORY, CLAIM_DETAILS, POLICY_INFO}


# ── Error codes ───────────────────────────────────────────────────────────────
class ErrorCode:
    RATE_LIMITED = "RATE_LIMITED"
    TOKEN_MISSING = "TOKEN_MISSING"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"
    AUTH_FAILED = "AUTH_FAILED"
    INPUT_INVALID = "INPUT_INVALID"
    SQL_UNSAFE = "SQL_UNSAFE"
    DB_ERROR = "DB_ERROR"
    LLM_ERROR = "LLM_ERROR"
