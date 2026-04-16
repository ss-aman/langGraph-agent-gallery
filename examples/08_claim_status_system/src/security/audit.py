"""
src/security/audit.py — Audit event construction and async persistence.

Every request — successful or not — generates an audit record.
PII is masked before persistence (see utils/pii_masker.py).
The audit_agent node calls `build_audit_record()` then
`write_audit_log_async()` as a fire-and-forget background task.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from src.utils.pii_masker import mask_pii

logger = logging.getLogger(__name__)


def build_audit_record(state: dict[str, Any]) -> dict[str, Any]:
    """
    Construct a PII-masked audit record from the current graph state.

    Only safe / masked fields are included in the record written to the DB.
    Never log: raw JWT tokens, full names, phone numbers, account numbers.
    """
    return {
        "log_id": str(uuid.uuid4()),
        "session_id": state.get("session_id", "unknown"),
        # user_id is a stable pseudonym — acceptable in audit logs
        "user_id": state.get("user_id"),
        "intent": state.get("intent"),
        # Mask PII in the query before logging
        "query_masked": mask_pii(state.get("raw_query", "")),
        "status": _derive_status(state),
        "error_code": state.get("error_code"),
        "response_time_ms": state.get("response_time_ms"),
    }


def _derive_status(state: dict[str, Any]) -> str:
    """Map graph state to an audit status string."""
    error_code = state.get("error_code", "")
    if error_code == "RATE_LIMITED":
        return "rate_limited"
    if error_code and "AUTH" in error_code or error_code and "TOKEN" in error_code:
        return "auth_failed"
    if error_code:
        return "failed"
    return "success"


async def write_audit_log_async(record: dict, db_path: str) -> None:
    """
    Persist an audit record to the database without blocking the caller.

    Called as asyncio.create_task() from audit_agent_node so the graph
    can return immediately while the write completes in the background.
    """
    try:
        # Import here to avoid circular imports at module load
        from src.database.repository import ClaimsRepository
        async with ClaimsRepository(db_path) as repo:
            await repo.write_audit_log(record)
        logger.debug("Audit record written: session=%s status=%s",
                     record["session_id"], record["status"])
    except Exception as exc:
        # Audit failures must NEVER propagate to the user —
        # log and continue.
        logger.error("Failed to write audit log: %s", exc, exc_info=True)
