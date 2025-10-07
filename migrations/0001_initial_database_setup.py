#!/usr/bin/env python3
"""
Migration: 0001_initial_database_setup
Description: Initial database setup with basic tables (users, preferences, deliveries, channel_posts)
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
    """Apply migration - create initial tables"""
    try:
        log.info("Applying migration 0001: Initial database setup...")
        
        # Create all tables defined in models
        Base.metadata.create_all(engine)
        
        log.info("✅ Migration 0001 applied successfully!")
        log.info("Created tables:")
        for table_name in Base.metadata.tables.keys():
            log.info(f"  - {table_name}")
            
    except Exception as e:
        log.error(f"❌ Error applying migration 0001: {e}")
        raise

def down():
    """Rollback migration - drop all tables"""
    try:
        log.info("Rolling back migration 0001: Dropping all tables...")
        
        # Drop all tables
        Base.metadata.drop_all(engine)
        
        log.info("✅ Migration 0001 rolled back successfully!")
        
    except Exception as e:
        log.error(f"❌ Error rolling back migration 0001: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "down":
        down()
    else:
        up()
