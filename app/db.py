from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from app.config import settings
from app.models import Base
import logging
import time

log = logging.getLogger(__name__)

def init_db():
    """Initialize database with comprehensive logging"""
    log.info(f"Initializing database connection to: {settings.DATABASE_URL}")
    start_time = time.time()
    
    try:
        Base.metadata.create_all(engine)
        init_time = time.time() - start_time
        log.info(f"✅ Database initialization completed in {init_time:.2f}s")
        
        # Test database connection
        test_connection()
        
    except Exception as e:
        init_time = time.time() - start_time
        log.error(f"❌ Database initialization failed after {init_time:.2f}s: {e}")
        raise

def test_connection():
    """Test database connection with detailed logging"""
    log.info("Testing database connection...")
    start_time = time.time()
    
    try:
        with SessionLocal() as db:
            # Simple query to test connection
            result = db.execute("SELECT 1").fetchone()
            connection_time = time.time() - start_time
            log.info(f"✅ Database connection test passed in {connection_time:.2f}s")
            log.info(f"Database URL: {settings.DATABASE_URL}")
            log.info(f"Connection result: {result}")
            
    except SQLAlchemyError as e:
        connection_time = time.time() - start_time
        log.error(f"❌ Database connection test failed after {connection_time:.2f}s: {e}")
        log.error(f"SQLAlchemy error type: {type(e).__name__}")
        raise
    except Exception as e:
        connection_time = time.time() - start_time
        log.error(f"❌ Database connection test failed after {connection_time:.2f}s: {e}")
        raise

# Initialize engine with comprehensive logging
log.info(f"Creating database engine for: {settings.DATABASE_URL}")
try:
    engine = create_engine(
        settings.DATABASE_URL, 
        pool_pre_ping=True,
        echo=False,  # Set to True for SQL query logging
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600
    )
    log.info("✅ Database engine created successfully")
except Exception as e:
    log.error(f"❌ Failed to create database engine: {e}")
    raise

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
log.info("✅ SessionLocal configured successfully")
