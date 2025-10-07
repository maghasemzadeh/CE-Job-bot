#!/usr/bin/env python3
"""
Migration runner script for CE Job Bot
Usage: python scripts/migrate.py [up|down|status]
"""

import sys
import os
import glob
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.db import SessionLocal
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MigrationRunner:
    def __init__(self):
        self.migrations_dir = project_root / "migrations"
        self.migrations = self._load_migrations()
    
    def _load_migrations(self):
        """Load all migration files"""
        migrations = []
        migration_files = glob.glob(str(self.migrations_dir / "*.py"))
        migration_files = [f for f in migration_files if not f.endswith("__init__.py")]
        
        for file_path in sorted(migration_files):
            migration_name = Path(file_path).stem
            if migration_name.startswith(("0001_", "0002_", "0003_")):
                migrations.append((migration_name, file_path))
        
        return migrations
    
    def _get_applied_migrations(self):
        """Get list of applied migrations from database"""
        try:
            with SessionLocal() as db:
                # Check if migrations table exists
                result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"))
                if not result.fetchone():
                    # Create migrations table if it doesn't exist
                    db.execute(text("""
                        CREATE TABLE migrations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT UNIQUE NOT NULL,
                            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """))
                    db.commit()
                
                # Get applied migrations
                result = db.execute(text("SELECT name FROM migrations ORDER BY id"))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            log.error(f"Error getting applied migrations: {e}")
            return []
    
    def _mark_migration_applied(self, migration_name):
        """Mark a migration as applied"""
        try:
            with SessionLocal() as db:
                db.execute(text("INSERT INTO migrations (name) VALUES (:name)"), {"name": migration_name})
                db.commit()
        except Exception as e:
            log.error(f"Error marking migration as applied: {e}")
    
    def _unmark_migration_applied(self, migration_name):
        """Unmark a migration as applied"""
        try:
            with SessionLocal() as db:
                db.execute(text("DELETE FROM migrations WHERE name = :name"), {"name": migration_name})
                db.commit()
        except Exception as e:
            log.error(f"Error unmarking migration: {e}")
    
    def _run_migration(self, migration_name, file_path, direction="up"):
        """Run a single migration"""
        try:
            # Load migration module
            spec = importlib.util.spec_from_file_location(migration_name, file_path)
            migration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration_module)
            
            # Run migration
            if direction == "up":
                migration_module.up()
                self._mark_migration_applied(migration_name)
                log.info(f"✅ Migration {migration_name} applied successfully")
            else:
                migration_module.down()
                self._unmark_migration_applied(migration_name)
                log.info(f"✅ Migration {migration_name} rolled back successfully")
                
        except Exception as e:
            log.error(f"❌ Error running migration {migration_name}: {e}")
            raise
    
    def status(self):
        """Show migration status"""
        applied = self._get_applied_migrations()
        
        log.info("Migration Status:")
        log.info("=" * 50)
        
        for migration_name, file_path in self.migrations:
            status = "✅ APPLIED" if migration_name in applied else "⏳ PENDING"
            log.info(f"{migration_name}: {status}")
        
        log.info("=" * 50)
        log.info(f"Total migrations: {len(self.migrations)}")
        log.info(f"Applied: {len(applied)}")
        log.info(f"Pending: {len(self.migrations) - len(applied)}")
    
    def up(self):
        """Apply all pending migrations"""
        applied = self._get_applied_migrations()
        
        for migration_name, file_path in self.migrations:
            if migration_name not in applied:
                log.info(f"Applying migration: {migration_name}")
                self._run_migration(migration_name, file_path, "up")
            else:
                log.info(f"Skipping already applied migration: {migration_name}")
    
    def down(self, target_migration=None):
        """Rollback migrations"""
        applied = self._get_applied_migrations()
        
        if target_migration:
            # Rollback to specific migration
            target_index = None
            for i, (migration_name, _) in enumerate(self.migrations):
                if migration_name == target_migration:
                    target_index = i
                    break
            
            if target_index is None:
                log.error(f"Migration {target_migration} not found")
                return
            
            # Rollback migrations after target
            for migration_name, file_path in reversed(self.migrations[target_index + 1:]):
                if migration_name in applied:
                    log.info(f"Rolling back migration: {migration_name}")
                    self._run_migration(migration_name, file_path, "down")
        else:
            # Rollback last migration
            for migration_name, file_path in reversed(self.migrations):
                if migration_name in applied:
                    log.info(f"Rolling back last migration: {migration_name}")
                    self._run_migration(migration_name, file_path, "down")
                    break

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate.py [up|down|status] [target_migration]")
        sys.exit(1)
    
    command = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else None
    
    runner = MigrationRunner()
    
    if command == "status":
        runner.status()
    elif command == "up":
        runner.up()
    elif command == "down":
        runner.down(target)
    else:
        print("Invalid command. Use: up, down, or status")
        sys.exit(1)

if __name__ == "__main__":
    main()
