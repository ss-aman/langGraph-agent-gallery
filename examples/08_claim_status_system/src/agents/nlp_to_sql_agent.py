"""
src/agents/nlp_to_sql_agent.py — Natural-language to parameterised SQL.

Converts the classified intent + extracted entities into a safe SELECT
statement. Row-level security (RLS) is baked into the prompt: the LLM
is instructed to always begin the WHERE clause with `user_id = ?` and
always put the user_id as the first parameter.

The generated SQL is NOT executed here — it goes to sql_security_node
for validation first, then to query_node for execution.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.agents.state import ClaimSystemState, ErrorCode
from src.config import settings
from src.database.schema import SCHEMA_CONTEXT
from src.utils.llm_factory import create_structured_llm

logger = logging.getLogger(__name__)


# ── Structured output: what we expect from the LLM ───────────────────────────

class SQLQueryOutput(BaseModel):
    sql: str = Field(
        description=(
            "A single parameterised SQLite SELECT statement. "
            "Use ? for ALL literal values. "
            "First WHERE condition MUST be: user_id = ?"
        )
    )
    params: list[str | int | float] = Field(
        description=(
            "Ordered list of parameter values for each ? placeholder. "
            "The first element MUST be the user_id string."
        )
    )
    tables_used: list[str] = Field(
        description="List of table names referenced in the SQL."
    )
    explanation: str = Field(
        description="One sentence describing what this query does in plain English."
    )


def _build_prompt(
    intent: str,
    entities: dict,
    user_id: str,
    policy_ids: list[str],
) -> str:
    """
    Build the user-turn prompt for the NLP-to-SQL LLM.

    Includes intent, extracted entities, and user context.
    The system prompt (schema + rules) is passed separately.
    """
    entities_str = "\n".join(f"  {k}: {v}" for k, v in entities.items()) or "  (none)"
    policies_str = ", ".join(policy_ids) or "(none)"

    return f"""Generate a safe SQL query for the following request.

Intent: {intent}
Extracted entities:
{entities_str}

User context (for row-level security):
  user_id:    {user_id}     ← MUST be the first ? param and first WHERE condition
  policy_ids: {policies_str}

Remember:
  - Output ONLY a SELECT statement.
  - Use ? placeholders for all values.
  - The first ? in the WHERE clause must correspond to the user_id.
  - Limit results to {settings.max_result_rows} rows.
"""


async def nlp_to_sql_node(
    state: ClaimSystemState,
    config: RunnableConfig,
) -> dict:
    """
    Convert the classified intent into a parameterised SQL query.

    The generated SQL + params are stored in state for the security
    guard to inspect before any database access occurs.
    """
    intent = state.get("intent", "")
    entities = state.get("entities", {})
    user_id = state.get("user_id", "")
    policy_ids = state.get("policy_ids", [])

    schema_with_limit = SCHEMA_CONTEXT.format(max_rows=settings.max_result_rows)

    system_prompt = f"""You are a SQL query generator for an insurance claim database.

{schema_with_limit}

Your output must be valid SQLite syntax. Always use ? placeholders.
Never generate DDL (DROP, CREATE, ALTER) or DML (INSERT, UPDATE, DELETE).
"""

    user_prompt = _build_prompt(intent, entities, user_id, policy_ids)

    try:
        generator = create_structured_llm(SQLQueryOutput, temperature=0.0)
        result: SQLQueryOutput = await generator.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.error("NLP-to-SQL generation failed: %s", exc)
        return {
            "generated_sql": "",
            "sql_params": [],
            "sql_tables_used": [],
            "sql_explanation": "",
            "error": f"Failed to generate SQL: {exc}",
            "error_code": ErrorCode.LLM_ERROR,
            "response_text": "I couldn't understand your query well enough to look it up. "
                             "Please try rephrasing.",
        }

    # Ensure user_id is the first param (safety net — security node also checks)
    params = result.params
    if params and str(params[0]) != user_id:
        logger.warning(
            "LLM did not place user_id first in params — correcting. "
            "Generated params[0]=%s", params[0]
        )
        params = [user_id] + [p for p in params if str(p) != user_id]

    logger.debug(
        "SQL generated: %s | params count=%d | tables=%s",
        result.explanation, len(params), result.tables_used
    )

    return {
        "generated_sql": result.sql,
        "sql_params": params,
        "sql_tables_used": result.tables_used,
        "sql_explanation": result.explanation,
    }
