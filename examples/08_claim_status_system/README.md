# Insurance Claim Status System

A production-grade, async multi-agent LangGraph system for querying insurance
claim status via natural language. Demonstrates every major LangGraph pattern
plus real-world concerns: authentication, NLP-to-SQL, SQL security, rate
limiting, PII masking, and audit logging.

---

## What It Demonstrates

| Category | What's Shown |
|----------|-------------|
| **LangGraph** | `StateGraph`, `START/END`, `add_conditional_edges`, `RetryPolicy`, `MemorySaver`, `RunnableConfig`, `ainvoke`, `astream`, `get_state`, `get_state_history` |
| **Multi-agent** | 8 specialised async agent nodes with a clear separation of concerns |
| **NLP → SQL** | LLM converts natural language to parameterised SQL with mandatory RLS |
| **Security** | JWT auth, RBAC, input sanitisation, SQL keyword blocklist, table whitelist, prompt injection detection, PII masking |
| **Async** | All nodes are `async def`; DB via `aiosqlite`; audit is fire-and-forget |
| **Session mgmt** | `thread_id` in config enables multi-turn conversations via checkpointing |
| **Scaling** | Stateless nodes + swappable checkpointer (MemorySaver → Postgres) |
| **Design patterns** | Repository, Factory, Strategy, Observer, Guard Clause, Chain of Responsibility |
| **Generic LLM** | `init_chat_model` — swap provider by changing two env vars |

---

## Quick Start

### 1. Install dependencies

```bash
cd examples/08_claim_status_system
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your API key:
#   OPENAI_API_KEY=sk-...         (for OpenAI)
#   ANTHROPIC_API_KEY=sk-ant-...  (for Anthropic — also set LLM_PROVIDER=anthropic)
#   GOOGLE_API_KEY=...            (for Google — also set LLM_PROVIDER=google_genai)
```

### 3. Initialise the database

```bash
python setup_db.py
# Creates ./data/claims.db with schema + test data
```

### 4. Run the demo

```bash
python main.py
```

---

## Switching LLM Providers

No code changes needed — just update `.env`:

```bash
# OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6

# Google Gemini
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-1.5-pro

# Local Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3
```

---

## Test Scenarios (in main.py)

| # | Scenario | What It Tests |
|---|----------|--------------|
| 01 | Specific claim status | Happy path — `claim_status` intent + NLP→SQL |
| 02 | All claims history | `claim_history` intent — multi-row results |
| 03 | Claim event timeline | `claim_details` intent — JOIN across tables |
| 04 | Multi-turn conversation | Session continuity via `thread_id` |
| 05 | Expired JWT | Auth failure → graceful error |
| 06 | Rapid requests | Rate limit triggers after burst |
| 07 | Prompt injection | Sanitiser blocks malicious input |
| 08 | Streaming mode | `astream()` yields one dict per node |
| 09 | Help request | Direct response, no DB query |

---

## Test Data (seed_data.py)

### Users
| user_id | email | role |
|---------|-------|------|
| USR-001 | alice@example.com | policyholder |
| USR-002 | bob@example.com | policyholder |

### Alice's Claims
| claim_id | type | status | amount |
|----------|------|--------|--------|
| CLM-2024-0001 | accident (auto) | under_review | $4,500 |
| CLM-2024-0002 | flood (home) | approved | $12,000 → $9,800 |
| CLM-2024-0003 | theft (auto) | rejected | $800 |

### Bob's Claims
| claim_id | type | status | amount |
|----------|------|--------|--------|
| CLM-2024-0004 | medical | settled | $28,000 → $25,000 |
| CLM-2024-0005 | medical | pending | $1,800 |

---

## Agent Pipeline

```
[START]
  rate_limit_node   → token-bucket per session (swap dict → Redis for HA)
  auth_node         → JWT validation + DB user check + policy IDs
  intent_node       → LLM classifies + extracts entities + sanitises input
  nlp_to_sql_node   → LLM generates parameterised SELECT with RLS
  sql_security_node → keyword blocklist + table whitelist + RLS check
  query_node        → async DB execution with RetryPolicy(max_attempts=3)
  response_node     → LLM formats results + PII masking
  audit_node        → fire-and-forget async write to audit_logs table
[END]
```

---

## Production Upgrade Path

| Component | Dev (this example) | Production |
|-----------|-------------------|------------|
| Checkpointer | `MemorySaver` (in-process) | `AsyncPostgresSaver` |
| Rate limiter | `asyncio.Lock` + dict | Redis `INCR` + `EXPIRE` |
| Database | SQLite (`aiosqlite`) | PostgreSQL (`asyncpg`) |
| LLM auth | API key in `.env` | Secret manager (AWS/GCP/Azure) |
| JWT secret | Static env var | Rotated via KMS |
| Audit logs | SQLite table | Dedicated SIEM / data lake |
| PII masking | Regex patterns | Microsoft Presidio / cloud DLP |

---

## File Structure

```
08_claim_status_system/
├── ARCHITECTURE.md      ← Full flow diagram + design decisions
├── README.md            ← This file
├── requirements.txt
├── .env.example
├── setup_db.py          ← Run once to create DB + seed data
├── main.py              ← 9 demo scenarios
└── src/
    ├── config.py
    ├── database/
    │   ├── schema.py        SQL DDL + schema context for LLM
    │   ├── seed_data.py     Test users / policies / claims
    │   └── repository.py    Async repository (aiosqlite)
    ├── security/
    │   ├── auth.py          JWT create / validate / RBAC
    │   ├── sanitizer.py     Input sanitisation + injection detection
    │   ├── rate_limiter.py  Async token-bucket limiter
    │   └── audit.py         Audit record builder + async writer
    ├── agents/
    │   ├── state.py         ClaimSystemState TypedDict + constants
    │   ├── rate_limit_agent.py
    │   ├── auth_agent.py
    │   ├── intent_agent.py
    │   ├── nlp_to_sql_agent.py
    │   ├── sql_security_agent.py
    │   ├── query_agent.py
    │   ├── response_agent.py
    │   └── audit_agent.py
    ├── graph/
    │   └── builder.py       StateGraph assembly + routing + public API
    └── utils/
        ├── llm_factory.py   init_chat_model wrapper (provider-agnostic)
        └── pii_masker.py    Regex PII detection + masking
```
