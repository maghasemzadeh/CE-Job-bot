#!/usr/bin/env python3
"""
Migration: 0005_add_active_flag
Description: Add active flag to preferred_job_positions
Created: 2025-10-08
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import SessionLocal
import logging
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def up():
    """Apply migration - add active flag to preferred_job_positions"""
    try:
        log.info("Applying migration 0005: Adding active flag to preferred_job_positions...")
        with SessionLocal() as db:
            # Helper to check column existence
            def column_exists(table_name: str, column_name: str) -> bool:
                rows = db.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                return any(row[1] == column_name for row in rows)

            if not column_exists("preferred_job_positions", "active"):
                db.execute(text("ALTER TABLE preferred_job_positions ADD COLUMN active BOOLEAN"))
                log.info("  - Added preferred_job_positions.active (BOOLEAN, default NULL; app sets default True)")
            else:
                log.info("  - preferred_job_positions.active already exists; skipping")
            db.commit()
        log.info("✅ Migration 0005 applied successfully!")
    except Exception as e:
        log.error(f"❌ Error applying migration 0005: {e}")
        raise


def down():
    """Rollback migration - note: SQLite doesn't easily drop columns"""
    try:
        log.info("Rolling back migration 0005: cannot drop column on SQLite; manual intervention required if needed.")
    except Exception as e:
        log.error(f"❌ Error rolling back migration 0005: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "down":
        down()
    else:
        up()
