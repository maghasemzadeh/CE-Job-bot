#!/usr/bin/env python3
"""
Database initialization script for CE Job Bot
Usage: python scripts/init_db.py
Note: This script now uses the migration system
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.migrate import MigrationRunner
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def init_database():
    """Initialize database using migration system"""
    try:
        log.info("Initializing database using migration system...")
        
        runner = MigrationRunner()
        runner.up()
        
        log.info("✅ Database initialization completed!")
        
    except Exception as e:
        log.error(f"❌ Failed to initialize database: {e}")
        raise

def check_database_status():
    """Check database status using migration system"""
    try:
        log.info("Checking database status...")
        
        runner = MigrationRunner()
        runner.status()
        
    except Exception as e:
        log.error(f"❌ Error checking database status: {e}")
        raise

if __name__ == "__main__":
    init_database()
    check_database_status()
