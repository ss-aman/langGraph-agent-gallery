"""
src/agents/sql_security_agent.py — Multi-layer SQL security validation.

This node validates the LLM-generated SQL before any database execution.
It acts as a defence-in-depth layer: even if the NLP-to-SQL agent
produces malformed or malicious output, this gate prevents execution.

Validation layers:
  1. Statement type check       — must start with SELECT
  2. Blocked keyword scan       — no DDL/DML/meta-table keywords
  3. Table whitelist            — only allowed tables may be referenced
  4. Row-level security check   — user_id = ? must be in the WHERE clause
  5. Parameter count validation — ? count must match params length
  6. UNION injection check      — UNION SELECT is blocked
"""

from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig

from src.agents.state import ClaimSystemState, ErrorCode
from src.config import settings

logger = logging.getLogger(__name__)


def _count_placeholders(sql: str) -> int:
    """Count unquoted ? placeholders in a SQL string."""
    # Remove string literals first to avoid counting ? inside quoted values
    cleaned = re.sub(r"'[^']*'", "''", sql)
    return cleaned.count("?")


def validate_sql(
    sql: str,
    params: list,
    user_id: str,
    allowed_tables: list[str],
    blocked_keywords: list[str],
) -> tuple[bool, str]:
    """
    Validate a SQL string against all security rules.

    Returns:
        (is_safe: bool, rejection_reason: str)
        rejection_reason is empty string if safe.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL statement."

    sql_stripped = sql.strip().rstrip(";")
    sql_lower = sql_stripped.lower()

    # ── 1. Must be a SELECT statement ─────────────────────────────────────
    if not sql_lower.lstrip().startswith("select"):
        return False, f"Only SELECT statements are allowed. Got: {sql_stripped[:30]}..."

    # ── 2. Blocked keywords (DDL/DML/system access) ───────────────────────
    for keyword in blocked_keywords:
        # Use word boundaries to avoid false positives (e.g. "selected")
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, sql_lower):
            return False, f"Blocked keyword detected: '{keyword}'."

    # ── 3. Table whitelist ────────────────────────────────────────────────
    # Extract identifiers that follow FROM or JOIN
    table_refs = re.findall(
        r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_lower
    )
    for table in table_refs:
        if table not in allowed_tables:
            return False, f"Table '{table}' is not in the allowed list: {allowed_tables}."

    # ── 4. Row-level security — user_id = ? must be present ──────────────
    # Accept both forms: "user_id = ?" and "c.user_id = ?" (aliased tables)
    rls_pattern = re.compile(r"\buser_id\s*=\s*\?", re.I)
    if not rls_pattern.search(sql_lower):
        return False, "Missing mandatory row-level security filter: user_id = ?"

    # ── 5. Parameter count must match placeholder count ───────────────────
    placeholder_count = _count_placeholders(sql_stripped)
    if placeholder_count != len(params):
        return (
            False,
            f"Parameter mismatch: {placeholder_count} placeholders but "
            f"{len(params)} params provided.",
        )

    # ── 6. UNION SELECT injection ─────────────────────────────────────────
    # Legitimate queries in this domain never need UNION
    if re.search(r"\bunion\s+(?:all\s+)?select\b", sql_lower):
        return False, "UNION SELECT is not permitted."

    # ── 7. First param must be user_id (RLS enforcement) ─────────────────
    if not params or str(params[0]) != user_id:
        return (
            False,
            "First query parameter must be the user_id for row-level security.",
        )

    return True, ""


async def sql_security_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Validate the LLM-generated SQL before database execution.

    If validation fails: sets sql_safe=False, populates error fields,
    and sets a user-facing response_text.
    If validation passes: sets sql_safe=True and allows the graph to
    proceed to query_node.
    """
    sql = state.get("generated_sql", "")
    params = state.get("sql_params", [])
    user_id = state.get("user_id", "")

    is_safe, rejection_reason = validate_sql(
        sql=sql,
        params=params,
        user_id=user_id,
        allowed_tables=settings.allowed_sql_tables,
        blocked_keywords=settings.sql_blocked_keywords,
    )

    if not is_safe:
        logger.error(
            "SQL security validation FAILED for user=%s | reason=%s | sql=%s",
            user_id, rejection_reason, sql[:100]
        )
        return {
            "sql_safe": False,
            "sql_rejection_reason": rejection_reason,
            "error": f"SQL security validation failed: {rejection_reason}",
            "error_code": ErrorCode.SQL_UNSAFE,
            "response_text": (
                "I couldn't safely retrieve your data due to a query generation issue. "
                "Please try rephrasing your request."
            ),
        }

    logger.debug("SQL security validation PASSED for user=%s", user_id)
    return {
        "sql_safe": True,
        "sql_rejection_reason": "",
    }
