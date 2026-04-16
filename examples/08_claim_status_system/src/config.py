"""
src/config.py — Centralised application configuration.

Uses Pydantic-Settings so every value can be overridden via environment
variable or a .env file.  Import the singleton `settings` object everywhere
instead of reading os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings, loaded from env / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_provider: str = Field(default="openai", description="LangChain provider name")
    llm_model: str = Field(default="gpt-4o-mini", description="Model identifier")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── JWT ───────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(
        default="dev-secret-please-change-in-production",
        min_length=16,
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_hours: int = Field(default=24, gt=0, le=720)

    # ── Database ──────────────────────────────────────────────────────────
    database_path: str = Field(default="./data/claims.db")

    # ── Rate limiting ─────────────────────────────────────────────────────
    rate_limit_rpm: int = Field(default=10, gt=0, description="Requests per minute (sustained)")
    rate_limit_burst: int = Field(default=20, gt=0, description="Max burst above sustained")

    # ── Security ──────────────────────────────────────────────────────────
    max_query_length: int = Field(default=500, gt=0)
    max_result_rows: int = Field(default=50, gt=0, le=1000)

    # SQL table whitelist — agents may only query these tables
    allowed_sql_tables: list[str] = Field(
        default=["claims", "claim_events", "policies"],
        description="Whitelisted tables for NLP-to-SQL queries",
    )

    # Keywords that must never appear in generated SQL
    sql_blocked_keywords: list[str] = Field(
        default=[
            "drop", "delete", "insert", "update", "alter", "create",
            "truncate", "exec", "execute", "xp_", "sp_", "information_schema",
            "sqlite_master", "pragma", "--", "/*", "*/",
        ]
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = {"HS256", "HS384", "HS512"}
        if v not in allowed:
            raise ValueError(f"jwt_algorithm must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Module-level singleton for convenience
settings: Settings = get_settings()
