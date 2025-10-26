#!/usr/bin/env python3
"""
Comprehensive connection testing script for CE Job Bot
Tests all external connections and provides detailed diagnostics
"""

import logging
import time
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('connection_test.log', encoding='utf-8')
    ]
)

log = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    log.info("Testing imports...")
    try:
        from app.config import settings
        from app.db import SessionLocal, test_connection
        from app.classification import JobClassifier
        from app.matcher import PreferenceMatcher, KeywordExtractor
        log.info("✅ All imports successful")
        return True
    except Exception as e:
        log.error(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    log.info("Testing configuration...")
    try:
        from app.config import settings
        
        log.info(f"Configuration loaded:")
        log.info(f"  - Database URL: {settings.DATABASE_URL}")
        log.info(f"  - LLM Base URL: {settings.LLM_BASE_URL}")
        log.info(f"  - LLM Model: {settings.LLM_MODEL_NAME}")
        log.info(f"  - Embedding Model: {settings.EMBED_MODEL}")
        log.info(f"  - Similarity Threshold: {settings.SIM_THRESHOLD}")
        log.info(f"  - Admin IDs: {settings.ADMIN_IDS}")
        log.info(f"  - Channel ID: {settings.CHANNEL_ID}")
        log.info(f"  - Bot Token: {'Configured' if settings.TELEGRAM_BOT_TOKEN else 'Not configured'}")
        log.info(f"  - OpenAI API Key: {'Configured' if settings.OPENAI_API_KEY else 'Not configured'}")
        
        # Check for missing critical settings
        missing = []
        if not settings.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not settings.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not settings.ADMIN_IDS:
            missing.append("ADMIN_IDS")
            
        if missing:
            log.warning(f"⚠️ Missing configuration: {missing}")
        else:
            log.info("✅ All critical configuration present")
            
        return len(missing) == 0
        
    except Exception as e:
        log.error(f"❌ Configuration test failed: {e}")
        return False

def test_database():
    """Test database connection"""
    log.info("Testing database connection...")
    try:
        from app.db import test_connection, SessionLocal
        
        # Test basic connection
        test_connection()
        
        # Test session creation
        with SessionLocal() as db:
            result = db.execute("SELECT 1 as test").fetchone()
            log.info(f"✅ Database connection successful: {result}")
            
        return True
        
    except Exception as e:
        log.error(f"❌ Database test failed: {e}")
        return False

def test_telegram():
    """Test Telegram bot API connectivity"""
    log.info("Testing Telegram bot API...")
    try:
        from app.config import settings
        import requests
        
        test_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                log.info(f"✅ Telegram bot API working: @{bot_info['result'].get('username')}")
                return True
            else:
                log.error(f"❌ Telegram API error: {bot_info}")
                return False
        else:
            log.error(f"❌ Telegram API returned status {response.status_code}")
            return False
            
    except Exception as e:
        log.error(f"❌ Telegram test failed: {e}")
        return False

def test_llm_api():
    """Test LLM API connectivity"""
    log.info("Testing LLM API connectivity...")
    try:
        from app.config import settings
        import requests
        
        test_url = f"{settings.LLM_BASE_URL.rstrip('/')}/models"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            log.info(f"✅ LLM API working: {len(models.get('data', []))} models available")
            return True
        else:
            log.error(f"❌ LLM API returned status {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        log.error(f"❌ LLM API test failed: {e}")
        return False

def test_classification():
    """Test job classification functionality"""
    log.info("Testing job classification...")
    try:
        from app.classification import JobClassifier
        
        classifier = JobClassifier()
        
        # Test with sample text
        test_text = "We are looking for a Python developer with 3+ years experience in Django and React."
        result = classifier.classify_job(test_text)
        
        log.info(f"✅ Classification test successful: {result.model_dump()}")
        return True
        
    except Exception as e:
        log.error(f"❌ Classification test failed: {e}")
        return False

def test_embeddings():
    """Test embedding functionality"""
    log.info("Testing embedding functionality...")
    try:
        from app.matcher import PreferenceMatcher
        
        matcher = PreferenceMatcher("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 0.62)
        
        # Test with sample data
        test_posts = ["Python developer needed", "React frontend developer", "DevOps engineer"]
        test_ids = [1, 2, 3]
        
        matcher.rebuild_index(test_posts, test_ids)
        
        # Test search
        results = matcher.search_users_by_embedding("Python developer")
        
        log.info(f"✅ Embedding test successful: {len(results)} matches found")
        return True
        
    except Exception as e:
        log.error(f"❌ Embedding test failed: {e}")
        return False

def test_keyword_extraction():
    """Test keyword extraction functionality"""
    log.info("Testing keyword extraction...")
    try:
        from app.matcher import KeywordExtractor
        
        extractor = KeywordExtractor("gpt-4o-mini")
        
        # Test with sample text
        test_text = "We need a Python developer with Django, React, and PostgreSQL experience."
        keywords = extractor.extract_keywords(test_text)
        
        log.info(f"✅ Keyword extraction test successful: {keywords}")
        return True
        
    except Exception as e:
        log.error(f"❌ Keyword extraction test failed: {e}")
        return False

def main():
    """Run all connection tests"""
    log.info("=" * 60)
    log.info("🔍 CE Job Bot Connection Test Suite")
    log.info("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Telegram", test_telegram),
        ("LLM API", test_llm_api),
        ("Classification", test_classification),
        ("Embeddings", test_embeddings),
        ("Keyword Extraction", test_keyword_extraction),
    ]
    
    for test_name, test_func in tests:
        log.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            log.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    log.info("\n" + "=" * 60)
    log.info("📊 Test Results Summary")
    log.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        log.info(f"{test_name}: {status}")
    
    log.info(f"\nOverall: {passed}/{total} tests passed in {total_time:.2f}s")
    
    if passed == total:
        log.info("🎉 All tests passed! System is ready.")
    else:
        log.warning(f"⚠️ {total - passed} tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
