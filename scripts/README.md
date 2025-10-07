# CE Job Bot Scripts

This directory contains utility scripts for managing the CE Job Bot database and system.

## Available Scripts

### Database Management

#### `migrate.py` - Migration System
Django-style migration system for database schema changes.

```bash
# Check migration status
python scripts/migrate.py status

# Apply all pending migrations
python scripts/migrate.py up

# Rollback last migration
python scripts/migrate.py down

# Rollback to specific migration
python scripts/migrate.py down 0001_initial_database_setup
```

#### `init_db.py` - Database Initialization
Initialize the database with all required tables.

```bash
python scripts/init_db.py
```

#### `reset_db.py` - Database Reset
⚠️ **WARNING**: This will delete all data!

```bash
python scripts/reset_db.py
```

#### `backup_db.py` - Database Backup & Restore
Create and manage database backups.

```bash
# Create backup with timestamp
python scripts/backup_db.py

# Create backup with custom name
python scripts/backup_db.py my_backup

# List all backups
python scripts/backup_db.py list

# Restore from backup
python scripts/backup_db.py restore my_backup
```

#### `db_info.py` - Database Information
Show database statistics and information.

```bash
python scripts/db_info.py
```

## Migration System

The migration system follows Django-style conventions:

- **Numbered migrations**: `0001_`, `0002_`, etc.
- **Descriptive names**: Clear description of what the migration does
- **Up/Down methods**: Each migration can be applied or rolled back
- **Tracking**: Applied migrations are tracked in the database

### Migration Files

Located in the `migrations/` directory:

1. **`0001_initial_database_setup.py`** - Creates initial tables (users, preferences, deliveries, channel_posts)
2. **`0002_add_classification_fields.py`** - Adds classification fields to channel_posts
3. **`0003_add_classification_data_json.py`** - Adds JSON field for complete classification data

### Creating New Migrations

1. Create a new file in `migrations/` with the next number: `0004_description.py`
2. Follow the template structure with `up()` and `down()` functions
3. Test the migration with `python scripts/migrate.py up`

## Usage Examples

### Initial Setup
```bash
# Initialize database for the first time
python scripts/init_db.py

# Check status
python scripts/migrate.py status
```

### Daily Operations
```bash
# Check database info
python scripts/db_info.py

# Create backup before major changes
python scripts/backup_db.py before_changes
```

### Development
```bash
# Reset database during development
python scripts/reset_db.py

# Apply new migrations
python scripts/migrate.py up
```

## File Structure

```
scripts/
├── __init__.py
├── README.md
├── migrate.py          # Migration runner
├── init_db.py          # Database initialization
├── reset_db.py         # Database reset
├── backup_db.py        # Backup/restore
└── db_info.py          # Database information

migrations/
├── __init__.py
├── 0001_initial_database_setup.py
├── 0002_add_classification_fields.py
└── 0003_add_classification_data_json.py
```

## Notes

- All scripts require the virtual environment to be activated
- Scripts automatically add the project root to the Python path
- Database operations are logged for debugging
- Backup files are stored in the `backups/` directory


