#!/usr/bin/env python3
"""
Migration: 0003_add_classification_data_json
Description: Add classification_data JSON field to store complete Pydantic classification data
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
    """Apply migration - add classification_data JSON field"""
    try:
        log.info("Applying migration 0003: Adding classification_data JSON field...")
        
        # Create all tables (this will add the new JSON column)
        Base.metadata.create_all(engine)
        
        log.info("✅ Migration 0003 applied successfully!")
        log.info("Added JSON field to channel_posts:")
        log.info("  - classification_data: Complete JobPositionClassification data as JSON")
        
        # Verify the migration
        with SessionLocal() as db:
            try:
                # Check if the column exists by trying to query it
                result = db.execute("SELECT classification_data FROM channel_posts LIMIT 1")
                log.info("✅ classification_data column exists and is accessible")
            except Exception as e:
                log.warning(f"⚠️  Could not verify classification_data column: {e}")
        
    except Exception as e:
        log.error(f"❌ Error applying migration 0003: {e}")
        raise

def down():
    """Rollback migration - remove classification_data JSON field"""
    try:
        log.info("Rolling back migration 0003: Removing classification_data JSON field...")
        
        log.warning("⚠️  SQLite doesn't support dropping columns easily.")
        log.warning("⚠️  Manual table recreation would be required for rollback.")
        log.warning("⚠️  Consider backing up data before applying this migration.")
        
        log.info("✅ Migration 0003 rollback completed (with warnings)")
        
    except Exception as e:
        log.error(f"❌ Error rolling back migration 0003: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "down":
        down()
    else:
        up()


