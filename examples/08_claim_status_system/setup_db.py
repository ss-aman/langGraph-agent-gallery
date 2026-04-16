"""
setup_db.py — One-time database initialisation script.

Creates the SQLite schema and seeds test data.
Run this ONCE before starting the application:

    python setup_db.py

Safe to re-run (idempotent): schema uses CREATE TABLE IF NOT EXISTS,
and seed data is skipped if the users table is already populated.
"""

import asyncio
import os
import sys

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config import settings
from src.database.repository import init_database, seed_database


async def main() -> None:
    db_path = settings.database_path

    # Ensure the data/ directory exists
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    print(f"Initialising database at: {db_path}")
    await init_database(db_path)
    print("  ✓ Schema created / verified")

    await seed_database(db_path)
    print("  ✓ Seed data inserted (or already present)")
    print("\nDatabase ready. You can now run: python main.py")


if __name__ == "__main__":
    asyncio.run(main())
