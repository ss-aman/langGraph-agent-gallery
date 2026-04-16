"""
src/agents/auth_agent.py — JWT authentication and RBAC node.

Validates the bearer token, extracts the user profile, and enforces
that the user is active. Sets user_id, role, and policy_ids in state —
all downstream nodes rely on these for row-level security.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.agents.state import ClaimSystemState, ErrorCode
from src.database.repository import ClaimsRepository
from src.security.auth import AuthError, validate_token
from src.config import settings

logger = logging.getLogger(__name__)


async def auth_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Validate JWT and load the user profile from the database.

    Why load from DB and not rely solely on the token?
      - Tokens can be revoked (user deactivated) between issuance and use.
      - policy_ids in the token can be stale; DB is the source of truth.
    """
    token = state.get("user_token", "")

    # ── 1. JWT validation ─────────────────────────────────────────────────
    try:
        claims = validate_token(token)
    except AuthError as exc:
        logger.warning("Auth failed: %s (%s)", exc, exc.error_code)
        return {
            "auth_status": "failed",
            "error": str(exc),
            "error_code": exc.error_code,
            "response_text": str(exc),
            "messages": [AIMessage(content=str(exc))],
        }

    # ── 2. Check user is still active in the DB ────────────────────────────
    async with ClaimsRepository() as repo:
        user = await repo.get_user_by_id(claims.user_id)

    if not user:
        logger.warning("Authenticated user %s not found in DB", claims.user_id)
        return {
            "auth_status": "failed",
            "error": "User account not found.",
            "error_code": ErrorCode.AUTH_FAILED,
            "response_text": "Authentication failed. Please contact support.",
            "messages": [AIMessage(content="Authentication failed.")],
        }

    if not user["is_active"]:
        logger.warning("Deactivated user attempted access: %s", claims.user_id)
        return {
            "auth_status": "failed",
            "error": "Account is deactivated.",
            "error_code": ErrorCode.AUTH_FAILED,
            "response_text": "Your account is deactivated. Please contact support.",
            "messages": [AIMessage(content="Account deactivated.")],
        }

    # ── 3. Load authoritative policy IDs from DB ───────────────────────────
    async with ClaimsRepository() as repo:
        policy_ids = await repo.get_policy_ids_for_user(claims.user_id)

    logger.info("User authenticated: id=%s role=%s policies=%d",
                claims.user_id, claims.role, len(policy_ids))

    return {
        "auth_status": "ok",
        "user_id": claims.user_id,
        "user_email": claims.email,
        "user_role": claims.role,
        "policy_ids": policy_ids,
    }
