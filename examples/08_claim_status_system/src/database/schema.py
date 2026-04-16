"""
src/database/schema.py — SQL schema definitions.

All CREATE TABLE and CREATE INDEX statements live here.
The repository imports `SCHEMA_SQL` to initialise the database.
The NLP-to-SQL agent imports `SCHEMA_CONTEXT` so the LLM knows the structure.
"""

# ── DDL ───────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;        -- better concurrent read performance
PRAGMA foreign_keys=ON;         -- enforce FK constraints

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    full_name   TEXT NOT NULL,
    phone       TEXT,
    role        TEXT NOT NULL DEFAULT 'policyholder'
                     CHECK (role IN ('policyholder', 'agent', 'admin')),
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id       TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(user_id),
    policy_type     TEXT NOT NULL
                         CHECK (policy_type IN ('auto', 'health', 'home', 'life')),
    policy_number   TEXT UNIQUE NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    premium_amount  REAL NOT NULL CHECK (premium_amount > 0),
    coverage_amount REAL NOT NULL CHECK (coverage_amount > 0),
    status          TEXT NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active', 'expired', 'cancelled')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id        TEXT PRIMARY KEY,
    policy_id       TEXT NOT NULL REFERENCES policies(policy_id),
    user_id         TEXT NOT NULL REFERENCES users(user_id),
    claim_type      TEXT NOT NULL
                         CHECK (claim_type IN ('accident', 'theft', 'medical', 'fire', 'flood', 'other')),
    description     TEXT NOT NULL,
    amount_claimed  REAL NOT NULL CHECK (amount_claimed > 0),
    amount_approved REAL,
    status          TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'under_review', 'approved', 'rejected', 'settled', 'withdrawn')),
    submitted_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS claim_events (
    event_id          TEXT PRIMARY KEY,
    claim_id          TEXT NOT NULL REFERENCES claims(claim_id),
    event_type        TEXT NOT NULL,
    event_description TEXT NOT NULL,
    agent_id          TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id           TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    user_id          TEXT,
    intent           TEXT,
    query_masked     TEXT,
    status           TEXT NOT NULL,
    error_code       TEXT,
    response_time_ms INTEGER,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_claims_user_id     ON claims(user_id);
CREATE INDEX IF NOT EXISTS idx_claims_policy_id   ON claims(policy_id);
CREATE INDEX IF NOT EXISTS idx_claims_status      ON claims(status);
CREATE INDEX IF NOT EXISTS idx_policies_user_id   ON policies(user_id);
CREATE INDEX IF NOT EXISTS idx_claim_events_claim ON claim_events(claim_id);
CREATE INDEX IF NOT EXISTS idx_audit_session      ON audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_created      ON audit_logs(created_at);
"""


# ── Schema context fed to the NLP-to-SQL LLM ─────────────────────────────────
# This is the "prompt engineering" for the NLP-to-SQL agent.
# Keep it accurate and concise — every token costs money.

SCHEMA_CONTEXT = """
AVAILABLE TABLES (you may ONLY query these):

  claims(claim_id, policy_id, user_id, claim_type, description,
         amount_claimed, amount_approved, status, submitted_at, updated_at)
    status values: pending | under_review | approved | rejected | settled | withdrawn
    claim_type:    accident | theft | medical | fire | flood | other

  claim_events(event_id, claim_id, event_type, event_description,
               agent_id, created_at)
    Use to retrieve the history / timeline of a specific claim.

  policies(policy_id, user_id, policy_type, policy_number,
           start_date, end_date, premium_amount, coverage_amount, status)
    policy_type: auto | health | home | life
    status:      active | expired | cancelled

MANDATORY SECURITY RULES — violating any rule makes the query UNSAFE:
  1. Generate SELECT queries ONLY — no INSERT, UPDATE, DELETE, DROP, ALTER, etc.
  2. The FIRST condition in every WHERE clause MUST be: user_id = ?
     The FIRST element in the params list MUST be the user_id placeholder string.
  3. Use ? placeholders for ALL literal values — NEVER interpolate strings.
  4. Do not reference tables outside the three listed above.
  5. Do not use subqueries that read from system tables.
  6. Limit result rows with LIMIT {max_rows}.
"""
