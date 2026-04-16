"""
src/database/repository.py — Async repository pattern for database access.

The Repository class is the ONLY place that touches the database.
All SQL is parameterised (? placeholders) to prevent injection.
Connection pooling is handled by keeping a single aiosqlite connection
per instance; for production use a proper pool (e.g. aiosqlite + asyncio.Queue
or switch to asyncpg for PostgreSQL).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import aiosqlite

from src.config import settings
from src.database.schema import SCHEMA_SQL
from src.database.seed_data import CLAIM_EVENTS, CLAIMS, POLICIES, USERS

logger = logging.getLogger(__name__)


class ClaimsRepository:
    """Async data-access layer for the claims database.

    Usage (async context manager):
        async with ClaimsRepository() as repo:
            claims = await repo.get_claims_for_user("USR-001")
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.database_path
        self._conn: aiosqlite.Connection | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the database connection and enable row factories."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row   # rows accessible as dicts
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA journal_mode=WAL")
        logger.debug("Database connection opened: %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.debug("Database connection closed.")

    async def __aenter__(self) -> "ClaimsRepository":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT and return all rows as list[dict]."""
        assert self._conn, "Repository not connected. Use async context manager."
        async with self._conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute a SELECT and return the first row as dict, or None."""
        assert self._conn, "Repository not connected. Use async context manager."
        async with self._conn.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def _execute(self, sql: str, params: tuple = ()) -> int:
        """Execute a write statement and return rows affected."""
        assert self._conn, "Repository not connected. Use async context manager."
        async with self._conn.execute(sql, params) as cursor:
            affected = cursor.rowcount
        await self._conn.commit()
        return affected

    # ── User queries ──────────────────────────────────────────────────────

    async def get_user_by_id(self, user_id: str) -> dict | None:
        return await self._fetchone(
            "SELECT user_id, email, full_name, role, is_active FROM users WHERE user_id = ?",
            (user_id,),
        )

    async def get_policy_ids_for_user(self, user_id: str) -> list[str]:
        rows = await self._fetchall(
            "SELECT policy_id FROM policies WHERE user_id = ?",
            (user_id,),
        )
        return [r["policy_id"] for r in rows]

    # ── Claim queries ─────────────────────────────────────────────────────

    async def get_claims_for_user(
        self, user_id: str, limit: int | None = None
    ) -> list[dict]:
        limit = min(limit or settings.max_result_rows, settings.max_result_rows)
        return await self._fetchall(
            """
            SELECT c.claim_id, c.claim_type, c.status,
                   c.amount_claimed, c.amount_approved,
                   c.submitted_at, c.updated_at,
                   p.policy_type, p.policy_number
            FROM   claims c
            JOIN   policies p ON p.policy_id = c.policy_id
            WHERE  c.user_id = ?
            ORDER  BY c.submitted_at DESC
            LIMIT  ?
            """,
            (user_id, limit),
        )

    async def get_claim_by_id(self, user_id: str, claim_id: str) -> dict | None:
        """Fetch a single claim — user_id ensures row-level security."""
        return await self._fetchone(
            """
            SELECT c.*, p.policy_type, p.policy_number
            FROM   claims c
            JOIN   policies p ON p.policy_id = c.policy_id
            WHERE  c.user_id = ? AND c.claim_id = ?
            """,
            (user_id, claim_id),
        )

    async def get_claim_events(self, user_id: str, claim_id: str) -> list[dict]:
        """Return timeline events for a claim (user_id enforces RLS)."""
        return await self._fetchall(
            """
            SELECT ce.*
            FROM   claim_events ce
            JOIN   claims c ON c.claim_id = ce.claim_id
            WHERE  c.user_id = ? AND ce.claim_id = ?
            ORDER  BY ce.created_at ASC
            """,
            (user_id, claim_id),
        )

    # ── Execute arbitrary validated SQL (NLP-to-SQL path) ─────────────────

    async def execute_safe_query(
        self,
        sql: str,
        params: list[Any],
        max_rows: int | None = None,
    ) -> list[dict]:
        """
        Execute a pre-validated SELECT statement.

        This method trusts that sql_security_node has already validated:
          - SELECT-only
          - Parameterised with ?
          - user_id = ? is the first WHERE condition
          - Tables are whitelisted

        An additional LIMIT is appended as a safeguard.
        """
        max_rows = min(max_rows or settings.max_result_rows, settings.max_result_rows)
        # Append LIMIT if not already present
        sql_lower = sql.lower().strip().rstrip(";")
        if "limit" not in sql_lower:
            sql_lower += f" LIMIT {max_rows}"

        return await self._fetchall(sql_lower, tuple(params))

    # ── Audit log ─────────────────────────────────────────────────────────

    async def write_audit_log(self, record: dict) -> None:
        """Persist an audit log entry asynchronously."""
        await self._execute(
            """
            INSERT INTO audit_logs
                (log_id, session_id, user_id, intent,
                 query_masked, status, error_code, response_time_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            """,
            (
                record.get("log_id", str(uuid.uuid4())),
                record["session_id"],
                record.get("user_id"),
                record.get("intent"),
                record.get("query_masked"),
                record["status"],
                record.get("error_code"),
                record.get("response_time_ms"),
            ),
        )


# ── Database initialisation ───────────────────────────────────────────────────

async def init_database(db_path: str | None = None) -> None:
    """Create schema and seed test data (idempotent)."""
    db_path = db_path or settings.database_path
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()
        logger.info("Schema created/verified at: %s", db_path)


async def seed_database(db_path: str | None = None) -> None:
    """Insert seed data if the users table is empty."""
    db_path = db_path or settings.database_path
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM users")
        count = (await cursor.fetchone())[0]
        if count > 0:
            logger.info("Seed data already present, skipping.")
            return

        for user in USERS:
            await conn.execute(
                """INSERT OR IGNORE INTO users
                   (user_id, email, full_name, phone, role, is_active, created_at, updated_at)
                   VALUES (:user_id, :email, :full_name, :phone, :role,
                           :is_active, :created_at, :updated_at)""",
                user,
            )
        for policy in POLICIES:
            await conn.execute(
                """INSERT OR IGNORE INTO policies
                   (policy_id, user_id, policy_type, policy_number,
                    start_date, end_date, premium_amount, coverage_amount,
                    status, created_at)
                   VALUES (:policy_id, :user_id, :policy_type, :policy_number,
                           :start_date, :end_date, :premium_amount, :coverage_amount,
                           :status, :created_at)""",
                policy,
            )
        for claim in CLAIMS:
            await conn.execute(
                """INSERT OR IGNORE INTO claims
                   (claim_id, policy_id, user_id, claim_type, description,
                    amount_claimed, amount_approved, status, submitted_at, updated_at)
                   VALUES (:claim_id, :policy_id, :user_id, :claim_type, :description,
                           :amount_claimed, :amount_approved, :status,
                           :submitted_at, :updated_at)""",
                claim,
            )
        for event in CLAIM_EVENTS:
            await conn.execute(
                """INSERT OR IGNORE INTO claim_events
                   (event_id, claim_id, event_type, event_description, agent_id, created_at)
                   VALUES (:event_id, :claim_id, :event_type, :event_description,
                           :agent_id, :created_at)""",
                event,
            )
        await conn.commit()
        logger.info("Seed data inserted: %d users, %d policies, %d claims.",
                    len(USERS), len(POLICIES), len(CLAIMS))
