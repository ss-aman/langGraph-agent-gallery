# Insurance Claim Status System — Architecture

## System Overview

A production-grade, async multi-agent LangGraph system that lets policyholders
query their insurance claim status using natural language. Every request passes
through authentication, intent classification, NLP-to-SQL translation, SQL
security validation, and database retrieval before a formatted response is
returned — with full audit logging at every step.

---

## Full Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          INSURANCE CLAIM STATUS SYSTEM — LANGGRAPH FLOW                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

  User (JWT token + natural-language query)
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        LangGraph Async Graph                            │
  │                                                                         │
  │  [START]                                                                │
  │     │                                                                   │
  │     ▼                                                                   │
  │  ┌──────────────────────────────────┐                                  │
  │  │         rate_limit_node          │  Token-bucket per user/IP        │
  │  │  • Sliding-window rate check     │  (swap dict → Redis for HA)      │
  │  │  • Async asyncio.Lock            │                                   │
  │  └──────────┬───────────────────────┘                                  │
  │             │ ALLOWED                    EXCEEDED                       │
  │             │                    ┌──────────────────────┐              │
  │             │                    │  Set error + 429 msg │              │
  │             │                    └──────────┬───────────┘              │
  │             │                               │                          │
  │             ▼                               │                          │
  │  ┌──────────────────────────────────┐       │                          │
  │  │           auth_node              │  JWT Validation + RBAC           │
  │  │  • Decode & verify JWT           │                                  │
  │  │  • Load user role, policy IDs    │                                  │
  │  │  • Row-level security setup      │                                  │
  │  └──────────┬───────────────────────┘                                  │
  │             │ AUTHENTICATED              AUTH_FAILED                   │
  │             │                    ┌──────────────────────┐              │
  │             │                    │  Set error + 401 msg │              │
  │             │                    └──────────┬───────────┘              │
  │             │                               │                          │
  │             ▼                               │                          │
  │  ┌──────────────────────────────────┐       │                          │
  │  │          intent_node             │  LLM + structured output         │
  │  │  • Classify: claim_status        │                                  │
  │  │             claim_history        │  Pydantic-validated intent       │
  │  │             claim_details        │  + entity extraction             │
  │  │             policy_info          │  (claim_id, date ranges…)        │
  │  │             help / greeting      │                                  │
  │  │             unknown              │                                  │
  │  └──────────┬───────────────────────┘                                  │
  │             │                                                           │
  │     ┌───────┴──────────────────────────────────────┐                  │
  │     │ DB-NEEDED?                                   │                  │
  │  YES│ (claim_status, claim_history,                │NO                │
  │     │  claim_details, policy_info)           (help, greeting,         │
  │     ▼                                         unknown)                │
  │  ┌──────────────────────────────────┐             │                   │
  │  │        nlp_to_sql_node           │             │                   │
  │  │  • Schema-aware LLM prompt       │             │                   │
  │  │  • Parameterised SQL (? only)    │             │                   │
  │  │  • RLS: WHERE user_id = ?        │             │                   │
  │  │  • Pydantic SQLQueryOutput       │             │                   │
  │  └──────────┬───────────────────────┘             │                   │
  │             │                                      │                   │
  │             ▼                                      │                   │
  │  ┌──────────────────────────────────┐              │                   │
  │  │       sql_security_node          │  Multi-layer validation          │
  │  │  • Keyword blocklist (DDL/DML)   │                                  │
  │  │  • Table whitelist check         │                                  │
  │  │  • RLS WHERE-clause enforcement  │                                  │
  │  │  • Param count validation        │                                  │
  │  └──────────┬───────────────────────┘                                  │
  │             │ SAFE                    UNSAFE                           │
  │             │                    ┌──────────────────────┐              │
  │             │                    │ Set error + 400 msg  │              │
  │             │                    └──────────┬───────────┘              │
  │             │                               │                          │
  │             ▼                               │                          │
  │  ┌──────────────────────────────────┐       │                          │
  │  │          query_node              │  Async aiosqlite                 │
  │  │  • Execute parameterised SQL     │  RetryPolicy on transient err    │
  │  │  • Paginate results (MAX_ROWS)   │                                  │
  │  │  • Result serialisation          │                                  │
  │  └──────────┬───────────────────────┘                                  │
  │             │                                                           │
  │             ▼                          ◄────────────────────────────── │
  │  ┌──────────────────────────────────┐                                  │
  │  │        response_node             │  All paths converge here         │
  │  │  • LLM formats DB results        │                                  │
  │  │  • PII masking on output         │                                  │
  │  │  • Error → user-friendly message │                                  │
  │  └──────────┬───────────────────────┘                                  │
  │             │                                                           │
  │             ▼                                                           │
  │  ┌──────────────────────────────────┐                                  │
  │  │          audit_node              │  Fire-and-forget async write     │
  │  │  • PII-masked audit record       │  to audit_logs table             │
  │  │  • session_id, user_id, intent   │                                  │
  │  │  • status, error_code, latency   │                                  │
  │  └──────────┬───────────────────────┘                                  │
  │             │                                                           │
  │           [END] ──► Formatted response returned to caller              │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Session & Horizontal Scaling Architecture

```
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Instance 1  │   │  Instance 2  │   │  Instance 3  │  (stateless workers)
  │  LangGraph   │   │  LangGraph   │   │  LangGraph   │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            │ thread_id = session_id
                            ▼
              ┌─────────────────────────┐
              │  Shared Checkpointer    │  ← MemorySaver (dev)
              │  (AsyncSqliteSaver or   │    AsyncPostgresSaver (prod)
              │   Redis-based store)    │
              └─────────────┬───────────┘
                            │
              ┌─────────────────────────┐
              │  Shared Database        │  ← SQLite (dev)
              │  (aiosqlite pool)       │    PostgreSQL (prod)
              └─────────────────────────┘

  Rate Limiter: ─── asyncio.Lock + dict (dev)
                     Redis INCR + EXPIRE (prod)
```

---

## Directory Structure

```
08_claim_status_system/
│
├── ARCHITECTURE.md          ← This file (diagrams + design decisions)
├── README.md                ← Setup, quickstart, test scenarios
├── requirements.txt         ← All dependencies
├── .env.example             ← Environment variable template
│
├── setup_db.py              ← One-time: create schema + seed test data
├── main.py                  ← Entry point: async demo with test scenarios
│
└── src/
    ├── __init__.py
    │
    ├── config.py            ← Pydantic Settings (env-based config)
    │
    ├── database/
    │   ├── __init__.py
    │   ├── schema.py        ← SQL CREATE TABLE statements + indexes
    │   ├── seed_data.py     ← Test users, policies, claims
    │   └── repository.py    ← Async repository pattern (aiosqlite)
    │
    ├── security/
    │   ├── __init__.py
    │   ├── auth.py          ← JWT creation/validation + RBAC
    │   ├── sanitizer.py     ← Input sanitization + length checks
    │   ├── rate_limiter.py  ← Async token-bucket rate limiter
    │   └── audit.py         ← Audit event builder + async writer
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── state.py         ← ClaimSystemState TypedDict (shared state)
    │   ├── rate_limit_agent.py
    │   ├── auth_agent.py
    │   ├── intent_agent.py
    │   ├── nlp_to_sql_agent.py
    │   ├── sql_security_agent.py
    │   ├── query_agent.py
    │   ├── response_agent.py
    │   └── audit_agent.py
    │
    ├── graph/
    │   ├── __init__.py
    │   └── builder.py       ← StateGraph assembly + all routing logic
    │
    └── utils/
        ├── __init__.py
        ├── llm_factory.py   ← Generic LLM factory (init_chat_model)
        └── pii_masker.py    ← PII detection + masking for logs
```

---

## Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Repository** | `database/repository.py` | Decouple DB access from business logic |
| **Factory** | `utils/llm_factory.py` | Swap LLM providers without touching agents |
| **Strategy** | `security/auth.py` | Pluggable auth validation strategies |
| **Observer** | `security/audit.py` | Audit logging decoupled from business logic |
| **Chain of Responsibility** | LangGraph graph nodes | Each node handles or passes to next |
| **Guard Clause** | Every node | Fail fast, return early on bad state |
| **Typed State** | `agents/state.py` | TypedDict enforces schema across all nodes |

---

## Security Controls

| Control | Implementation |
|---------|---------------|
| Authentication | JWT HS256 + expiry + signature verification |
| Authorisation (RBAC) | Role checked in auth_node; policy_ids extracted |
| Row-Level Security | `WHERE user_id = ?` enforced in sql_security_node |
| SQL Injection | Parameterised queries only; `?` placeholders |
| DDL/DML Prevention | Keyword blocklist in sql_security_node |
| Table Whitelisting | Only `claims`, `claim_events`, `policies` accessible |
| Input Sanitisation | sanitizer.py strips HTML/JS, truncates to MAX_QUERY_LENGTH |
| Rate Limiting | Token bucket per user; configurable RPM + burst |
| PII Masking | Regex-based masker applied to audit logs |
| Audit Trail | Every request logged (even failures) with masked data |

---

## LangGraph APIs Demonstrated

```python
# Graph construction
StateGraph, START, END, add_messages

# Prebuilt components
ToolNode, tools_condition

# Checkpointing (session management)
MemorySaver
# → swap to AsyncSqliteSaver / AsyncPostgresSaver for production

# Advanced control flow
Command, interrupt, Send, RetryPolicy

# Runtime configuration
RunnableConfig

# Execution
graph.ainvoke(), graph.astream()
graph.get_state(), graph.get_state_history()
```
