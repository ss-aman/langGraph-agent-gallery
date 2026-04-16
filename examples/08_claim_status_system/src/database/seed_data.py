"""
src/database/seed_data.py — Test data for development and demonstration.

Creates two policyholders (alice, bob) with realistic policies, claims,
and claim event histories so every demo scenario has something to query.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


def _now(offset_days: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(days=offset_days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Users ─────────────────────────────────────────────────────────────────────
USERS = [
    {
        "user_id": "USR-001",
        "email": "alice@example.com",
        "full_name": "Alice Johnson",
        "phone": "+1-555-0101",
        "role": "policyholder",
        "is_active": 1,
        "created_at": _now(-365),
        "updated_at": _now(-30),
    },
    {
        "user_id": "USR-002",
        "email": "bob@example.com",
        "full_name": "Bob Martinez",
        "phone": "+1-555-0202",
        "role": "policyholder",
        "is_active": 1,
        "created_at": _now(-200),
        "updated_at": _now(-10),
    },
    {
        "user_id": "USR-AGENT-001",
        "email": "agent.smith@insure.com",
        "full_name": "Agent Smith",
        "phone": "+1-555-9000",
        "role": "agent",
        "is_active": 1,
        "created_at": _now(-500),
        "updated_at": _now(-1),
    },
]

# ── Policies ──────────────────────────────────────────────────────────────────
POLICIES = [
    # Alice's policies
    {
        "policy_id": "POL-AUTO-001",
        "user_id": "USR-001",
        "policy_type": "auto",
        "policy_number": "AUT-2024-000001",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "premium_amount": 1200.00,
        "coverage_amount": 50000.00,
        "status": "active",
        "created_at": _now(-365),
    },
    {
        "policy_id": "POL-HOME-001",
        "user_id": "USR-001",
        "policy_type": "home",
        "policy_number": "HOM-2024-000001",
        "start_date": "2024-03-01",
        "end_date": "2025-02-28",
        "premium_amount": 800.00,
        "coverage_amount": 250000.00,
        "status": "active",
        "created_at": _now(-310),
    },
    # Bob's policies
    {
        "policy_id": "POL-HEALTH-001",
        "user_id": "USR-002",
        "policy_type": "health",
        "policy_number": "HLT-2024-000002",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "premium_amount": 3600.00,
        "coverage_amount": 500000.00,
        "status": "active",
        "created_at": _now(-200),
    },
]

# ── Claims ────────────────────────────────────────────────────────────────────
CLAIMS = [
    # Alice — auto accident claim (under review)
    {
        "claim_id": "CLM-2024-0001",
        "policy_id": "POL-AUTO-001",
        "user_id": "USR-001",
        "claim_type": "accident",
        "description": "Rear-end collision at intersection. Vehicle damage to bumper and trunk.",
        "amount_claimed": 4500.00,
        "amount_approved": None,
        "status": "under_review",
        "submitted_at": _now(-45),
        "updated_at": _now(-20),
    },
    # Alice — home flood claim (approved)
    {
        "claim_id": "CLM-2024-0002",
        "policy_id": "POL-HOME-001",
        "user_id": "USR-001",
        "claim_type": "flood",
        "description": "Basement flooding from storm drain overflow. Damaged flooring and drywall.",
        "amount_claimed": 12000.00,
        "amount_approved": 9800.00,
        "status": "approved",
        "submitted_at": _now(-90),
        "updated_at": _now(-5),
    },
    # Alice — auto theft claim (rejected)
    {
        "claim_id": "CLM-2024-0003",
        "policy_id": "POL-AUTO-001",
        "user_id": "USR-001",
        "claim_type": "theft",
        "description": "Stereo system stolen from parked vehicle. Police report filed.",
        "amount_claimed": 800.00,
        "amount_approved": None,
        "status": "rejected",
        "submitted_at": _now(-120),
        "updated_at": _now(-100),
    },
    # Bob — medical claim (settled)
    {
        "claim_id": "CLM-2024-0004",
        "policy_id": "POL-HEALTH-001",
        "user_id": "USR-002",
        "claim_type": "medical",
        "description": "Emergency appendectomy. Hospital stay 3 nights.",
        "amount_claimed": 28000.00,
        "amount_approved": 25000.00,
        "status": "settled",
        "submitted_at": _now(-60),
        "updated_at": _now(-15),
    },
    # Bob — pending medical claim
    {
        "claim_id": "CLM-2024-0005",
        "policy_id": "POL-HEALTH-001",
        "user_id": "USR-002",
        "claim_type": "medical",
        "description": "Physiotherapy sessions for post-surgery recovery. 12 sessions.",
        "amount_claimed": 1800.00,
        "amount_approved": None,
        "status": "pending",
        "submitted_at": _now(-5),
        "updated_at": _now(-5),
    },
]

# ── Claim Events ──────────────────────────────────────────────────────────────
CLAIM_EVENTS = [
    # CLM-2024-0001 (under review)
    {"event_id": "EVT-001", "claim_id": "CLM-2024-0001", "event_type": "submitted",
     "event_description": "Claim submitted via online portal.", "agent_id": None,
     "created_at": _now(-45)},
    {"event_id": "EVT-002", "claim_id": "CLM-2024-0001", "event_type": "assigned",
     "event_description": "Claim assigned to adjuster Agent Smith.",
     "agent_id": "USR-AGENT-001", "created_at": _now(-40)},
    {"event_id": "EVT-003", "claim_id": "CLM-2024-0001", "event_type": "document_requested",
     "event_description": "Police report and repair estimate requested.",
     "agent_id": "USR-AGENT-001", "created_at": _now(-35)},
    {"event_id": "EVT-004", "claim_id": "CLM-2024-0001", "event_type": "documents_received",
     "event_description": "Repair estimate ($4,200) and photos received.",
     "agent_id": None, "created_at": _now(-20)},

    # CLM-2024-0002 (approved)
    {"event_id": "EVT-010", "claim_id": "CLM-2024-0002", "event_type": "submitted",
     "event_description": "Claim submitted with contractor quotes.", "agent_id": None,
     "created_at": _now(-90)},
    {"event_id": "EVT-011", "claim_id": "CLM-2024-0002", "event_type": "under_review",
     "event_description": "Field inspection scheduled.", "agent_id": "USR-AGENT-001",
     "created_at": _now(-80)},
    {"event_id": "EVT-012", "claim_id": "CLM-2024-0002", "event_type": "approved",
     "event_description": "Approved for $9,800 after deductible. Payout initiated.",
     "agent_id": "USR-AGENT-001", "created_at": _now(-5)},

    # CLM-2024-0003 (rejected)
    {"event_id": "EVT-020", "claim_id": "CLM-2024-0003", "event_type": "submitted",
     "event_description": "Claim submitted.", "agent_id": None, "created_at": _now(-120)},
    {"event_id": "EVT-021", "claim_id": "CLM-2024-0003", "event_type": "rejected",
     "event_description": "Rejected: stereo accessories not covered under auto policy.",
     "agent_id": "USR-AGENT-001", "created_at": _now(-100)},

    # CLM-2024-0004 (settled)
    {"event_id": "EVT-030", "claim_id": "CLM-2024-0004", "event_type": "submitted",
     "event_description": "Medical claim submitted with hospital bills.", "agent_id": None,
     "created_at": _now(-60)},
    {"event_id": "EVT-031", "claim_id": "CLM-2024-0004", "event_type": "approved",
     "event_description": "Approved $25,000 after co-pay deduction.",
     "agent_id": "USR-AGENT-001", "created_at": _now(-30)},
    {"event_id": "EVT-032", "claim_id": "CLM-2024-0004", "event_type": "settled",
     "event_description": "Payment of $25,000 transferred to hospital.",
     "agent_id": None, "created_at": _now(-15)},

    # CLM-2024-0005 (pending)
    {"event_id": "EVT-040", "claim_id": "CLM-2024-0005", "event_type": "submitted",
     "event_description": "Physiotherapy claim submitted with receipts.", "agent_id": None,
     "created_at": _now(-5)},
]
