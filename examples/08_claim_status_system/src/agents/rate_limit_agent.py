"""
src/agents/rate_limit_agent.py — Rate limiting node.

First node in the graph. Checks the token bucket for the requesting user
(or session_id as a fallback key) before any LLM or DB work is done.
This prevents runaway costs from abusive or misconfigured clients.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.agents.state import ClaimSystemState, ErrorCode
from src.security.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


async def rate_limit_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Check per-user rate limit.

    Uses session_id as the bucket key (pre-auth, user_id not yet known).
    After authentication, you could call `limiter.check(user_id)` for
    more precise per-user limiting.
    """
    key = state.get("session_id", "anonymous")
    limiter = get_rate_limiter()
    allowed, remaining = await limiter.check(key)

    if not allowed:
        logger.warning("Rate limit exceeded for key=%s", key)
        return {
            "rate_limit_ok": False,
            "rate_limit_remaining": 0,
            "error": "Rate limit exceeded. Please wait before sending another request.",
            "error_code": ErrorCode.RATE_LIMITED,
            "response_text": (
                "You have sent too many requests. "
                "Please wait a moment and try again."
            ),
            "messages": [
                AIMessage(content="Rate limit exceeded. Please try again shortly.")
            ],
        }

    logger.debug("Rate limit OK for key=%s (remaining=%d)", key, remaining)
    return {
        "rate_limit_ok": True,
        "rate_limit_remaining": remaining,
    }
