"""
src/agents/query_agent.py — Async database query execution node.

Executes the validated, parameterised SQL against the claims database.
Results are stored in state as a list of dicts for the response_node
to format into natural language.

Retry policy: transient DB errors are retried up to 3 times with
exponential back-off (handled by LangGraph's RetryPolicy on the node).
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from src.agents.state import ClaimSystemState, ErrorCode
from src.config import settings
from src.database.repository import ClaimsRepository

logger = logging.getLogger(__name__)

# Maximum rows returned — hard cap independent of SQL LIMIT
_MAX_ROWS = settings.max_result_rows


async def query_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Execute the pre-validated SQL query against the database.

    Row-level security is enforced at two levels:
      1. The SQL itself contains WHERE user_id = ? (validated by sql_security_node)
      2. We cap result rows to prevent large data exfiltration
    """
    sql = state.get("generated_sql", "")
    params = state.get("sql_params", [])
    user_id = state.get("user_id", "")

    logger.debug(
        "Executing query for user=%s | sql=%s | params_count=%d",
        user_id, sql[:80], len(params)
    )

    try:
        async with ClaimsRepository() as repo:
            rows = await repo.execute_safe_query(
                sql=sql,
                params=params,
                max_rows=_MAX_ROWS,
            )
    except Exception as exc:
        logger.error("Database query error for user=%s: %s", user_id, exc, exc_info=True)
        return {
            "query_results": [],
            "query_row_count": 0,
            "query_error": str(exc),
            "error": f"Database error: {exc}",
            "error_code": ErrorCode.DB_ERROR,
            "response_text": (
                "I encountered an error retrieving your data. "
                "Please try again in a moment."
            ),
        }

    row_count = len(rows)
    logger.info(
        "Query completed for user=%s | rows_returned=%d", user_id, row_count
    )

    return {
        "query_results": rows,
        "query_row_count": row_count,
        "query_error": None,
    }
