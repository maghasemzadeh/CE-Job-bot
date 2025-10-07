#!/usr/bin/env python3
"""
Reset database script for CE Job Bot
Usage: python scripts/reset_db.py
WARNING: This will delete all data!
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.db import engine, SessionLocal
from app.models import Base
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def reset_database():
    """Reset the database by dropping and recreating all tables"""
    try:
        log.warning("⚠️  WARNING: This will delete ALL data in the database!")
        log.warning("⚠️  Make sure you have backups if needed.")
        
        # Ask for confirmation
        response = input("Are you sure you want to reset the database? (yes/no): ")
        if response.lower() != 'yes':
            log.info("Database reset cancelled.")
            return
        
        log.info("Resetting database...")
        
        # Drop all tables
        Base.metadata.drop_all(engine)
        log.info("✅ All tables dropped")
        
        # Recreate all tables
        Base.metadata.create_all(engine)
        log.info("✅ All tables recreated")
        
        log.info("✅ Database reset completed successfully!")
        
    except Exception as e:
        log.error(f"❌ Error resetting database: {e}")
        raise

if __name__ == "__main__":
    reset_database()


