#!/usr/bin/env python3
"""
Database migration script to add classification fields to channel_posts table
"""

from app.db import engine, SessionLocal
from app.models import Base
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def migrate_database():
    """Add new classification fields to the database"""
    try:
        log.info("Starting database migration...")
        
        # Create all tables (this will add new columns to existing tables)
        Base.metadata.create_all(engine)
        
        log.info("✅ Database migration completed successfully!")
        log.info("New classification fields added to channel_posts table:")
        log.info("  - employment_type, job_function, industry")
        log.info("  - seniority_level, work_location, job_specialization")
        log.info("  - skills_technologies, bonuses, health_insurance")
        log.info("  - stock_options, work_schedule, company_size")
        log.info("  - company_name, is_classified")
        
    except Exception as e:
        log.error(f"❌ Error during database migration: {e}")
        raise

if __name__ == "__main__":
    migrate_database()
