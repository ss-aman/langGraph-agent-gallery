"""
src/agents/audit_agent.py — Audit logging node (always last in the graph).

Builds a PII-masked audit record from final state and persists it
asynchronously. Using asyncio.create_task() means the graph returns
the response to the caller while the DB write completes in the background
— this keeps response latency low and decouples the audit path from
the critical path.

Also computes the total request duration (response_time_ms) which is
useful for latency monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.runnables import RunnableConfig

from src.agents.state import ClaimSystemState
from src.config import settings
from src.security.audit import build_audit_record, write_audit_log_async

logger = logging.getLogger(__name__)


async def audit_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Build and persist an audit log entry for this request.

    This node always runs — both successful and failed requests are logged.
    The response_time_ms is computed here using the request_start_time
    that was set in the graph's entry node.
    """
    start_time = state.get("request_start_time", time.monotonic())
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    record = build_audit_record({**state, "response_time_ms": elapsed_ms})

    logger.info(
        "Audit | session=%s user=%s intent=%s status=%s latency=%dms",
        record["session_id"],
        record.get("user_id", "unauthenticated"),
        record.get("intent", "n/a"),
        record["status"],
        elapsed_ms,
    )

    # Fire-and-forget: schedule the DB write without awaiting it
    # so the graph can return immediately.
    asyncio.create_task(
        write_audit_log_async(record, db_path=settings.database_path)
    )

    return {
        "response_time_ms": elapsed_ms,
        "audit_trail": state.get("audit_trail", []) + [record],
    }
