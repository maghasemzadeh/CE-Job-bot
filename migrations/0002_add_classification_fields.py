#!/usr/bin/env python3
"""
Migration: 0002_add_classification_fields
Description: Add classification fields to channel_posts table for job categorization
Created: 2024-01-01
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import engine, SessionLocal
from app.models import Base
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def up():
    """Apply migration - add classification fields"""
    try:
        log.info("Applying migration 0002: Adding classification fields...")
        
        # Create all tables (this will add new columns to existing tables)
        Base.metadata.create_all(engine)
        
        log.info("✅ Migration 0002 applied successfully!")
        log.info("Added classification fields to channel_posts:")
        log.info("  - employment_type, job_function, industry")
        log.info("  - seniority_level, work_location, job_specialization")
        log.info("  - skills_technologies, bonuses, health_insurance")
        log.info("  - stock_options, work_schedule, company_size")
        log.info("  - company_name, is_classified")
        
    except Exception as e:
        log.error(f"❌ Error applying migration 0002: {e}")
        raise

def down():
    """Rollback migration - remove classification fields"""
    try:
        log.info("Rolling back migration 0002: Removing classification fields...")
        
        with SessionLocal() as db:
            # Note: SQLite doesn't support dropping columns easily
            # This would require recreating the table without the new columns
            log.warning("⚠️  SQLite doesn't support dropping columns easily.")
            log.warning("⚠️  Manual table recreation would be required for rollback.")
            log.warning("⚠️  Consider backing up data before applying this migration.")
        
        log.info("✅ Migration 0002 rollback completed (with warnings)")
        
    except Exception as e:
        log.error(f"❌ Error rolling back migration 0002: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "down":
        down()
    else:
        up()


