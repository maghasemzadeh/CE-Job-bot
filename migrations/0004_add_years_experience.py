#!/usr/bin/env python3
"""
Migration: 0004_add_years_experience
Description: Add years_experience to channel_posts and create PreferredJobPosition & user resume fields
Created: 2025-10-08
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import engine, SessionLocal
from app.models import Base
import logging
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def up():
    """Apply migration - add years_experience and new profile tables/columns"""
    try:
        log.info("Applying migration 0004: Adding years_experience, PreferredJobPosition, and user resume fields...")
        
        # 1) Create new tables (idempotent)
        Base.metadata.create_all(engine)

        # 2) Add missing columns for existing SQLite DBs using ALTER TABLE if they don't exist
        with SessionLocal() as db:
            # Helper to check column existence
            def column_exists(table_name: str, column_name: str) -> bool:
                rows = db.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                return any(row[1] == column_name for row in rows)

            # users table columns
            if not column_exists("users", "chat_id"):
                db.execute(text("ALTER TABLE users ADD COLUMN chat_id INTEGER"))
            if not column_exists("users", "resume_file_id"):
                db.execute(text("ALTER TABLE users ADD COLUMN resume_file_id VARCHAR(255)"))
            if not column_exists("users", "resume_file_path"):
                db.execute(text("ALTER TABLE users ADD COLUMN resume_file_path VARCHAR(512)"))
            if not column_exists("users", "created_at"):
                db.execute(text("ALTER TABLE users ADD COLUMN created_at DATETIME"))
            if not column_exists("users", "updated_at"):
                db.execute(text("ALTER TABLE users ADD COLUMN updated_at DATETIME"))

            # channel_posts.years_experience
            if not column_exists("channel_posts", "years_experience"):
                db.execute(text("ALTER TABLE channel_posts ADD COLUMN years_experience INTEGER"))
            # channel_posts.channel_chat_id
            if not column_exists("channel_posts", "channel_chat_id"):
                db.execute(text("ALTER TABLE channel_posts ADD COLUMN channel_chat_id INTEGER"))

            # preferred_job_positions.preferred_position_text (if table pre-existed somehow)
            if column_exists("preferred_job_positions", "id") and not column_exists("preferred_job_positions", "preferred_position_text"):
                db.execute(text("ALTER TABLE preferred_job_positions ADD COLUMN preferred_position_text TEXT"))

            db.commit()

        log.info("✅ Migration 0004 applied successfully!")
        log.info("Schema changes:")
        log.info("  - channel_posts.years_experience (INTEGER, nullable)")
        log.info("  - preferred_job_positions table (includes preferred_position_text)")
        log.info("  - users: chat_id, resume_file_id, resume_file_path, timestamps")
        log.info("  - channel_posts: channel_chat_id")
    except Exception as e:
        log.error(f"❌ Error applying migration 0004: {e}")
        raise

def down():
    """Rollback migration - remove years_experience column"""
    try:
        log.info("Rolling back migration 0004: Removing years_experience column...")
        with SessionLocal() as db:
            # SQLite limitation note
            log.warning("⚠️  SQLite doesn't support dropping columns easily.")
            log.warning("⚠️  Manual table recreation would be required for rollback.")
    except Exception as e:
        log.error(f"❌ Error rolling back migration 0004: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "down":
        down()
    else:
        up()


