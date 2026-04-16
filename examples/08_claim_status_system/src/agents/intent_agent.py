"""
src/agents/intent_agent.py — Intent classification and entity extraction.

Uses the LLM with structured output to:
  1. Classify the user's request into one of the defined intent categories
  2. Extract domain entities: claim_id, policy_number, date ranges, etc.

Also applies input sanitisation here (first LLM-facing step) as a final
defence-in-depth measure after the rate limit and auth checks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agents.state import ClaimSystemState, ErrorCode, Intent
from src.security.sanitizer import SanitisationError, sanitise_query
from src.utils.llm_factory import create_structured_llm

logger = logging.getLogger(__name__)

VALID_INTENTS = [
    Intent.CLAIM_STATUS,
    Intent.CLAIM_HISTORY,
    Intent.CLAIM_DETAILS,
    Intent.POLICY_INFO,
    Intent.HELP,
    Intent.GREETING,
    Intent.UNKNOWN,
]


# ── Structured output schema ──────────────────────────────────────────────────

class IntentOutput(BaseModel):
    intent: str = Field(
        description=(
            "The classified intent. Must be exactly one of: "
            "claim_status, claim_history, claim_details, policy_info, "
            "help, greeting, unknown"
        )
    )
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extracted entities. Possible keys: "
            "claim_id (e.g. CLM-2024-0001), "
            "policy_number (e.g. AUT-2024-000001), "
            "status_filter (e.g. pending), "
            "date_from (YYYY-MM-DD), date_to (YYYY-MM-DD), "
            "claim_type (accident/theft/medical/fire/flood)"
        ),
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence score for the intent classification."
    )
    reasoning: str = Field(
        default="", description="Brief explanation of the classification decision."
    )


_SYSTEM_PROMPT = """You are an intent classifier for an insurance claim status system.

Classify the user's query into exactly one of these intents:
  - claim_status    : asking about the current status of a specific claim
  - claim_history   : asking to see a list of all their claims
  - claim_details   : asking for detailed information about a specific claim (events, timeline)
  - policy_info     : asking about their insurance policies
  - help            : asking what the system can do, or how to use it
  - greeting        : a greeting with no specific request ("hi", "hello", etc.)
  - unknown         : anything else that doesn't fit the above

Extract any entities you find (claim IDs, policy numbers, dates, status filters).
If a claim ID like CLM-2024-0001 is mentioned, extract it as claim_id.

Be strict: only output intents from the list above."""


async def intent_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Sanitise the raw query, then classify intent and extract entities.
    """
    raw_query = state.get("raw_query", "").strip()

    # ── 1. Input sanitisation ─────────────────────────────────────────────
    try:
        clean_query = sanitise_query(raw_query)
    except SanitisationError as exc:
        logger.warning("Input sanitisation failed: %s", exc)
        return {
            "intent": Intent.UNKNOWN,
            "entities": {},
            "intent_confidence": 0.0,
            "error": str(exc),
            "error_code": exc.error_code,
            "response_text": "Your query contains invalid content. Please rephrase.",
        }

    # ── 2. LLM classification ─────────────────────────────────────────────
    try:
        classifier = create_structured_llm(IntentOutput, temperature=0.0)
        result: IntentOutput = await classifier.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=clean_query),
        ])
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        return {
            "intent": Intent.UNKNOWN,
            "entities": {},
            "intent_confidence": 0.0,
            "error": f"LLM error: {exc}",
            "error_code": ErrorCode.LLM_ERROR,
            "response_text": "I'm having trouble understanding your request. Please try again.",
        }

    # Guard: ensure the LLM returned a valid intent
    if result.intent not in VALID_INTENTS:
        result.intent = Intent.UNKNOWN

    logger.info(
        "Intent classified: %s (confidence=%.2f) entities=%s",
        result.intent, result.confidence, list(result.entities.keys())
    )

    return {
        "intent": result.intent,
        "entities": result.entities,
        "intent_confidence": result.confidence,
    }
