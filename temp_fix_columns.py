#!/usr/bin/env python3
"""
Temporary script to:
- Rename channel_posts.job_function -> channel_posts.position
- Drop channel_posts.job_spec

Idempotent: checks column existence before applying changes.

Usage:
  python scripts/temp_fix_columns.py
"""

from sqlalchemy import text
from app.db import engine, SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def column_exists(db, table: str, column: str) -> bool:
    dialect = engine.dialect.name
    if dialect == "sqlite":
        rows = db.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return any(r[1] == column for r in rows)
    else:
        # Works for Postgres and many others
        q = text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
            LIMIT 1
            """
        )
        return db.execute(q, {"table": table, "column": column}).first() is not None


def rename_column_sqlite(db, table: str, old: str, new: str):
    try:
        db.execute(text(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}"))
        log.info(f"SQLite: Renamed {table}.{old} -> {new}")
    except Exception as e:
        log.error(f"SQLite: Failed to rename column {old} -> {new}: {e}")
        raise


def drop_column_sqlite(db, table: str, column: str):
    try:
        db.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        log.info(f"SQLite: Dropped column {table}.{column}")
    except Exception as e:
        log.error(
            f"SQLite: Failed to drop column {column}. Your SQLite may be < 3.35 (no DROP COLUMN). Error: {e}"
        )
        raise


def rename_column_pg(db, table: str, old: str, new: str):
    db.execute(text(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}"))
    log.info(f"Postgres: Renamed {table}.{old} -> {new}")


def drop_column_pg(db, table: str, column: str):
    db.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}"))
    log.info(f"Postgres: Dropped column if exists {table}.{column}")


def main():
    dialect = engine.dialect.name
    log.info(f"Using dialect: {dialect}")

    with SessionLocal() as db:
        # Rename job_function -> position
        has_job_function = column_exists(db, "channel_posts", "job_function")
        has_position = column_exists(db, "channel_posts", "position")

        if has_job_function and not has_position:
            log.info("Renaming channel_posts.job_function -> position ...")
            if dialect == "sqlite":
                rename_column_sqlite(db, "channel_posts", "job_function", "position")
            else:
                rename_column_pg(db, "channel_posts", "job_function", "position")
        else:
            log.info("No rename needed (either already renamed or source column missing)")

        # Drop job_specialization
        has_job_spec = column_exists(db, "channel_posts", "job_specialization")
        if has_job_spec:
            log.info("Dropping channel_posts.job_specialization ...")
            if dialect == "sqlite":
                drop_column_sqlite(db, "channel_posts", "job_specialization")
            else:
                drop_column_pg(db, "channel_posts", "job_specialization")
        else:
            log.info("No drop needed (job_specialization not present)")

        db.commit()
        log.info("✅ Temp column fix complete")


if __name__ == "__main__":
    main()


