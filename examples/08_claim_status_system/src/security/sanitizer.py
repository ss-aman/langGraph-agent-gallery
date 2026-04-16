"""
src/security/sanitizer.py — Input sanitisation and validation.

Applied to raw user queries BEFORE any LLM or database processing.
Guards against: prompt injection, excessively long inputs, HTML/script
injection, and common SQL injection patterns in the raw query text.

Note: SQL injection in the *generated* SQL is handled separately by the
sql_security_node with parameterised queries + keyword blocklists.
"""

from __future__ import annotations

import html
import re
import unicodedata

from src.config import settings


class SanitisationError(Exception):
    """Raised when input fails safety checks."""
    def __init__(self, message: str, error_code: str = "INPUT_INVALID") -> None:
        super().__init__(message)
        self.error_code = error_code


# Patterns that suggest prompt-injection attempts
_PROMPT_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(your\s+)?system\s+prompt", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"<\s*script", re.I),                        # XSS
    re.compile(r"javascript\s*:", re.I),                     # XSS
]

# SQL keywords that should never appear in a natural-language query
# (legitimate users don't type raw SQL)
_SUSPICIOUS_SQL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(drop|delete|truncate|alter|exec|execute)\b\s+\w+", re.I),
    re.compile(r"union\s+select", re.I),
    re.compile(r";\s*(drop|delete|insert|update)", re.I),
    re.compile(r"--\s*$", re.M),                            # SQL comment
    re.compile(r"/\*.*?\*/", re.S),                         # block comment
]


def sanitise_query(raw: str) -> str:
    """
    Sanitise a raw user query string.

    Steps:
      1. Unicode normalisation (NFKC) — neutralise homoglyph attacks
      2. HTML-entity decode then escape — neutralise XSS payloads
      3. Strip control characters (except standard whitespace)
      4. Trim to MAX_QUERY_LENGTH
      5. Check for prompt injection patterns
      6. Check for suspicious raw SQL patterns
      7. Collapse excess whitespace

    Returns the cleaned query string, or raises SanitisationError.
    """
    if not isinstance(raw, str):
        raise SanitisationError("Query must be a string.", "INPUT_TYPE_ERROR")

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", raw)

    # 2. HTML escape (neutralise <script>, onclick=, etc.)
    text = html.escape(text, quote=True)

    # 3. Remove control characters (keep standard whitespace: space, \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4. Length cap — BEFORE pattern checks to avoid ReDoS on huge inputs
    if len(text) > settings.max_query_length:
        raise SanitisationError(
            f"Query exceeds maximum length of {settings.max_query_length} characters.",
            "QUERY_TOO_LONG",
        )

    # 5. Prompt injection
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            raise SanitisationError(
                "Query contains disallowed content.", "PROMPT_INJECTION_DETECTED"
            )

    # 6. Suspicious SQL in raw query
    for pattern in _SUSPICIOUS_SQL_PATTERNS:
        if pattern.search(text):
            raise SanitisationError(
                "Query contains disallowed content.", "SUSPICIOUS_SQL_IN_QUERY"
            )

    # 7. Collapse excess whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text
