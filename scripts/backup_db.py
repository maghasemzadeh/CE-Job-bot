#!/usr/bin/env python3
"""
Database backup script for CE Job Bot
Usage: python scripts/backup_db.py [backup_name]
"""

import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def backup_database(backup_name=None):
    """Create a backup of the database"""
    try:
        db_path = project_root / "cejob.db"
        backups_dir = project_root / "backups"
        
        # Create backups directory if it doesn't exist
        backups_dir.mkdir(exist_ok=True)
        
        # Generate backup filename
        if backup_name:
            backup_filename = f"cejob_backup_{backup_name}.db"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"cejob_backup_{timestamp}.db"
        
        backup_path = backups_dir / backup_filename
        
        # Check if database exists
        if not db_path.exists():
            log.error(f"❌ Database file not found: {db_path}")
            return
        
        # Create backup
        shutil.copy2(db_path, backup_path)
        
        log.info(f"✅ Database backed up successfully!")
        log.info(f"Backup location: {backup_path}")
        log.info(f"Backup size: {backup_path.stat().st_size} bytes")
        
    except Exception as e:
        log.error(f"❌ Error creating backup: {e}")
        raise

def list_backups():
    """List all available backups"""
    try:
        backups_dir = project_root / "backups"
        
        if not backups_dir.exists():
            log.info("No backups directory found.")
            return
        
        backup_files = list(backups_dir.glob("cejob_backup_*.db"))
        
        if not backup_files:
            log.info("No backups found.")
            return
        
        log.info("Available backups:")
        log.info("=" * 50)
        
        for backup_file in sorted(backup_files):
            size = backup_file.stat().st_size
            modified = datetime.fromtimestamp(backup_file.stat().st_mtime)
            log.info(f"{backup_file.name}")
            log.info(f"  Size: {size} bytes")
            log.info(f"  Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            log.info("")
        
    except Exception as e:
        log.error(f"❌ Error listing backups: {e}")
        raise

def restore_database(backup_name):
    """Restore database from backup"""
    try:
        db_path = project_root / "cejob.db"
        backups_dir = project_root / "backups"
        backup_path = backups_dir / f"cejob_backup_{backup_name}.db"
        
        # Check if backup exists
        if not backup_path.exists():
            log.error(f"❌ Backup not found: {backup_path}")
            return
        
        log.warning("⚠️  WARNING: This will replace the current database!")
        log.warning("⚠️  Make sure you have a current backup if needed.")
        
        # Ask for confirmation
        response = input("Are you sure you want to restore from backup? (yes/no): ")
        if response.lower() != 'yes':
            log.info("Database restore cancelled.")
            return
        
        # Create backup of current database
        if db_path.exists():
            current_backup = backups_dir / f"cejob_backup_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(db_path, current_backup)
            log.info(f"Current database backed up to: {current_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, db_path)
        
        log.info(f"✅ Database restored successfully from: {backup_name}")
        
    except Exception as e:
        log.error(f"❌ Error restoring database: {e}")
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/backup_db.py [backup_name]     - Create backup")
        print("  python scripts/backup_db.py list              - List backups")
        print("  python scripts/backup_db.py restore <name>    - Restore from backup")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_backups()
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Please specify backup name to restore")
            sys.exit(1)
        restore_database(sys.argv[2])
    else:
        # Create backup with given name
        backup_database(command)

if __name__ == "__main__":
    main()


