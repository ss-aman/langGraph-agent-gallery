"""
src/security/auth.py — JWT creation and validation with RBAC.

Tokens carry: user_id, email, role, policy_ids, issued-at, expiry.
The auth_agent node calls `validate_token()` and uses the returned
`TokenClaims` to populate the graph state.

Security decisions:
  - HS256 with a secret key (swap to RS256 + public key for multi-service)
  - Short expiry (24h by default) + no refresh token in demo
  - Role hierarchy: policyholder < agent < admin
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

# Role hierarchy (higher index = more permissions)
ROLE_HIERARCHY: dict[str, int] = {
    "policyholder": 0,
    "agent": 1,
    "admin": 2,
}


class TokenClaims(BaseModel):
    """Validated, deserialised JWT payload."""
    user_id: str
    email: str
    role: str
    policy_ids: list[str] = Field(default_factory=list)
    issued_at: datetime
    expires_at: datetime


class AuthError(Exception):
    """Raised when token validation fails."""
    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def create_token(
    user_id: str,
    email: str,
    role: str,
    policy_ids: list[str],
    expiry_hours: int | None = None,
) -> str:
    """
    Create a signed JWT for a user.

    In production this would be called by your login/OAuth endpoint.
    Here it's used in main.py to generate test tokens.
    """
    expiry_hours = expiry_hours or settings.jwt_expiry_hours
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "policy_ids": policy_ids,
        "iat": now,
        "exp": now + timedelta(hours=expiry_hours),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def validate_token(token: str) -> TokenClaims:
    """
    Decode and validate a JWT.

    Raises AuthError with a specific error_code on any failure so the
    auth_agent can set the appropriate response and routing.
    """
    if not token:
        raise AuthError("No token provided", "TOKEN_MISSING")

    # Strip common prefixes
    token = token.removeprefix("Bearer ").strip()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "email", "role", "exp", "iat"]},
        )
    except ExpiredSignatureError:
        raise AuthError("Token has expired. Please log in again.", "TOKEN_EXPIRED")
    except InvalidTokenError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise AuthError("Invalid token. Authentication required.", "TOKEN_INVALID")

    role = payload.get("role", "policyholder")
    if role not in ROLE_HIERARCHY:
        raise AuthError(f"Unknown role '{role}'.", "ROLE_UNKNOWN")

    return TokenClaims(
        user_id=payload["sub"],
        email=payload["email"],
        role=role,
        policy_ids=payload.get("policy_ids", []),
        issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )


def has_permission(role: str, minimum_role: str) -> bool:
    """Check if a role meets or exceeds the minimum required role level."""
    return ROLE_HIERARCHY.get(role, -1) >= ROLE_HIERARCHY.get(minimum_role, 999)
