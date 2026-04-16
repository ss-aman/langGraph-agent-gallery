"""
src/agents/response_agent.py — Response formatting node.

All graph paths converge here. This node:
  - For DB results: uses the LLM to format raw rows into a natural,
    friendly response tailored to the user's intent
  - For direct answers (help, greeting): generates without DB context
  - For errors already set: returns the pre-built error message unchanged
  - Applies PII masking to the response before returning

This node also handles the edge case where no data was found
(empty query_results) gracefully.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.state import ClaimSystemState, ErrorCode, Intent
from src.utils.llm_factory import create_llm
from src.utils.pii_masker import mask_pii

logger = logging.getLogger(__name__)

_FORMATTER_SYSTEM = """You are a helpful insurance claims assistant.
Format the provided data into a clear, friendly response for the policyholder.

Guidelines:
- Use plain language, avoid jargon
- For claim statuses, clearly state what the status means for the customer
- For monetary amounts, format as dollar values (e.g. $4,500.00)
- For dates, use readable format (e.g. March 15, 2024)
- If no data was found, explain this clearly and suggest next steps
- Keep responses concise but complete (under 200 words)
- Never make up data — only use what is provided"""

_HELP_TEXT = """I can help you with the following:

• **Check claim status** — "What is the status of my claim CLM-2024-0001?"
• **View all claims** — "Show me all my claims" or "List my recent claims"
• **Claim details & timeline** — "Tell me about the history of claim CLM-2024-0001"
• **Policy information** — "What insurance policies do I have?"

Just describe what you need in plain language and I'll look it up for you."""

_GREETING_RESPONSE = (
    "Hello! I'm your insurance claims assistant. "
    "I can check claim statuses, show your claim history, "
    "and provide policy information. How can I help you today?"
)


async def response_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Format and return the final user-facing response.

    Handles four cases:
      A) Error already set (auth failure, rate limit, etc.) → return as-is
      B) Help / greeting intent → static or simple LLM response
      C) DB query with results → LLM formats the data
      D) DB query with no results → friendly "nothing found" message
    """

    # ── A. Error already set upstream ─────────────────────────────────────
    if state.get("error_code") and state.get("response_text"):
        # The error response was already set; just add it to messages
        response_text = state["response_text"]
        return {
            "messages": [AIMessage(content=response_text)],
        }

    intent = state.get("intent", Intent.UNKNOWN)

    # ── B. Help / greeting — no DB needed ─────────────────────────────────
    if intent == Intent.HELP:
        return {
            "response_text": _HELP_TEXT,
            "messages": [AIMessage(content=_HELP_TEXT)],
        }

    if intent == Intent.GREETING:
        return {
            "response_text": _GREETING_RESPONSE,
            "messages": [AIMessage(content=_GREETING_RESPONSE)],
        }

    if intent == Intent.UNKNOWN:
        unknown_msg = (
            "I'm not sure what you're asking. "
            "Try asking about a specific claim, your claim history, or your policies. "
            "Type 'help' for examples."
        )
        return {
            "response_text": unknown_msg,
            "messages": [AIMessage(content=unknown_msg)],
        }

    # ── C/D. Format database results ──────────────────────────────────────
    results = state.get("query_results", [])
    query_error = state.get("query_error")

    if query_error:
        error_msg = state.get("response_text", "An error occurred retrieving your data.")
        return {
            "response_text": error_msg,
            "messages": [AIMessage(content=error_msg)],
        }

    if not results:
        no_data_msg = (
            "I didn't find any records matching your request. "
            "This could mean the claim doesn't exist, or it's not associated with "
            "your account. Please check the claim ID and try again, or contact "
            "support if you believe this is an error."
        )
        return {
            "response_text": no_data_msg,
            "messages": [AIMessage(content=no_data_msg)],
        }

    # Serialise results for the LLM (compact JSON)
    results_json = json.dumps(results, indent=2, default=str)
    sql_explanation = state.get("sql_explanation", "")

    user_prompt = (
        f"User intent: {intent}\n"
        f"Query description: {sql_explanation}\n\n"
        f"Data retrieved from database:\n{results_json}\n\n"
        "Please format this into a clear, friendly response for the policyholder."
    )

    try:
        llm = create_llm(temperature=0.3)
        response = await llm.ainvoke([
            SystemMessage(content=_FORMATTER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        response_text = response.content
    except Exception as exc:
        logger.error("Response formatting failed: %s", exc)
        # Fallback: return raw data as a simple text list
        response_text = _fallback_format(results, intent)

    # Apply PII masking to the formatted response before returning
    # (defence-in-depth: LLM shouldn't echo raw PII anyway)
    response_text = mask_pii(response_text)

    return {
        "response_text": response_text,
        "messages": [AIMessage(content=response_text)],
    }


def _fallback_format(results: list[dict], intent: str) -> str:
    """Plain-text fallback if the LLM formatter fails."""
    lines = [f"Here are your results ({len(results)} record(s) found):\n"]
    for i, row in enumerate(results, 1):
        lines.append(f"Record {i}:")
        for key, value in row.items():
            if value is not None:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines)
