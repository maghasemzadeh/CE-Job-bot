#!/usr/bin/env python3
"""
Database information script for CE Job Bot
Usage: python scripts/db_info.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.db import SessionLocal
from app.models import User, Preference, Delivery, ChannelPost
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def show_database_info():
    """Show database information and statistics"""
    try:
        with SessionLocal() as db:
            log.info("Database Information")
            log.info("=" * 50)
            
            # Count records in each table
            user_count = db.query(User).count()
            preference_count = db.query(Preference).count()
            delivery_count = db.query(Delivery).count()
            channel_post_count = db.query(ChannelPost).count()
            classified_posts = db.query(ChannelPost).filter(ChannelPost.is_classified == True).count()
            
            log.info(f"Users: {user_count}")
            log.info(f"Preferences: {preference_count}")
            log.info(f"Deliveries: {delivery_count}")
            log.info(f"Channel Posts: {channel_post_count}")
            log.info(f"Classified Posts: {classified_posts}")
            log.info(f"Unclassified Posts: {channel_post_count - classified_posts}")
            
            log.info("")
            log.info("Recent Channel Posts:")
            log.info("-" * 30)
            
            recent_posts = db.query(ChannelPost).order_by(ChannelPost.posted_at.desc()).limit(5).all()
            for post in recent_posts:
                status = "✅ Classified" if post.is_classified else "⏳ Pending"
                log.info(f"ID: {post.channel_msg_id} | {status} | {post.posted_at.strftime('%Y-%m-%d %H:%M')}")
                if post.job_function:
                    log.info(f"  Role: {post.job_function}")
                if post.company_name:
                    log.info(f"  Company: {post.company_name}")
                log.info("")
            
            # Show classification statistics
            if classified_posts > 0:
                log.info("Classification Statistics:")
                log.info("-" * 30)
                
                # Employment types
                employment_types = db.query(ChannelPost.employment_type).filter(
                    ChannelPost.employment_type.isnot(None)
                ).distinct().all()
                log.info(f"Employment Types: {[t[0] for t in employment_types]}")
                
                # Job functions
                job_functions = db.query(ChannelPost.job_function).filter(
                    ChannelPost.job_function.isnot(None)
                ).distinct().all()
                log.info(f"Job Functions: {[f[0] for f in job_functions]}")
                
                # Work locations
                work_locations = db.query(ChannelPost.work_location).filter(
                    ChannelPost.work_location.isnot(None)
                ).distinct().all()
                log.info(f"Work Locations: {[l[0] for l in work_locations]}")
                
                # Companies
                companies = db.query(ChannelPost.company_name).filter(
                    ChannelPost.company_name.isnot(None)
                ).distinct().limit(10).all()
                log.info(f"Companies: {[c[0] for c in companies]}")
            
    except Exception as e:
        log.error(f"❌ Error getting database info: {e}")
        raise

if __name__ == "__main__":
    show_database_info()


