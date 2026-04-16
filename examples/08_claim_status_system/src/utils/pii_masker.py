"""
src/utils/pii_masker.py — Regex-based PII detection and masking.

Used by the audit layer to ensure no personal data is persisted in logs.
Patterns cover common US/international formats.

For production: consider a dedicated PII library (e.g. Microsoft Presidio,
spaCy NER, or a cloud DLP service) for higher recall.
"""

from __future__ import annotations

import re

# ── PII patterns ──────────────────────────────────────────────────────────────
# Each entry: (compiled_pattern, replacement_string)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),

    # US phone numbers (various formats)
    (re.compile(r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),

    # Credit / debit card numbers (Luhn-format, 13-19 digits with optional spaces/dashes)
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),

    # US Social Security Numbers
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN]"),

    # Policy numbers matching our seed data format (e.g. AUT-2024-000001)
    (re.compile(r"\b[A-Z]{3}-\d{4}-\d{6}\b"), "[POLICY_NUM]"),

    # Claim IDs (CLM-YYYY-NNNN)
    (re.compile(r"\bCLM-\d{4}-\d{4}\b"), "[CLAIM_ID]"),

    # User IDs (USR-NNN)
    (re.compile(r"\bUSR-\d{3,}\b"), "[USER_ID]"),

    # Dates (MM/DD/YYYY or DD-MM-YYYY or YYYY-MM-DD)
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z?)?\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"), "[DATE]"),

    # Dollar amounts
    (re.compile(r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?"), "[AMOUNT]"),
]


def mask_pii(text: str) -> str:
    """
    Replace detected PII in `text` with placeholder tokens.

    Safe to call on None or empty string.
    """
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def contains_pii(text: str) -> bool:
    """Return True if any PII pattern matches in `text`."""
    if not text:
        return False
    return any(p.search(text) for p, _ in _PATTERNS)
