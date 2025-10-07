#!/usr/bin/env python3
"""
OpenAI API Test Script for CE Job Bot
Usage: python scripts/test_openai.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__)
sys.path.insert(0, str(project_root))

print(os.listdir())
from app.config import Settings, import_config

import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_openai_connection():
    """Test OpenAI API connection with different methods"""
    
    # Load environment variables
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        log.info(f"✅ Loaded .env file from: {env_path}")
    else:
        log.warning(f"⚠️  No .env file found at: {env_path}")
        log.info("Loading environment variables from system...")
    
    # Check if API key is available
    api_key = import_config("OPENAI_API_KEY")
    if not api_key:
        log.error("❌ OPENAI_API_KEY not found in environment variables")
        log.info("Please set OPENAI_API_KEY in your .env file or environment")
        return False
    
    log.info(f"✅ Found OpenAI API key: {api_key[:10]}...{api_key[-4:]}")
    
    # Test 1: Direct OpenAI API call
    try:
        log.info("Testing direct OpenAI API call...")
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'API connection successful'"}
            ],
            max_tokens=50,
            temperature=0
        )
        
        log.info(f"✅ Direct OpenAI API call successful!")
        log.info(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        log.error(f"❌ Direct OpenAI API call failed: {e}")
        return False
    
    # Test 2: LangChain OpenAI call
    try:
        log.info("Testing LangChain OpenAI call...")
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0
        )
        
        response = llm.invoke("Hello! Please respond with 'LangChain connection successful'")
        
        log.info(f"✅ LangChain OpenAI call successful!")
        log.info(f"Response: {response.content}")
        
    except Exception as e:
        log.error(f"❌ LangChain OpenAI call failed: {e}")
        return False
    
    # Test 3: Test the classification system
    try:
        log.info("Testing job classification system...")
        from app.classification import job_classifier
        
        test_job_text = """
        We are looking for a Senior Python Developer to join our team.
        Requirements: Python, Django, PostgreSQL, 5+ years experience
        Location: Remote
        Type: Full-time
        Benefits: Health insurance, stock options
        """
        
        classification = job_classifier.classify_job(test_job_text)
        
        log.info(f"✅ Job classification test successful!")
        log.info(f"Employment Type: {classification.employment_type}")
        log.info(f"Job Function: {classification.job_function}")
        log.info(f"Seniority Level: {classification.seniority_level}")
        log.info(f"Work Location: {classification.work_location}")
        
    except Exception as e:
        log.error(f"❌ Job classification test failed: {e}")
        return False
    
    log.info("🎉 All OpenAI tests passed successfully!")
    return True

def check_environment():
    """Check environment setup"""
    log.info("Environment Check:")
    log.info("=" * 50)
    
    # Check .env file
    env_path = project_root / ".env"
    if env_path.exists():
        log.info(f"✅ .env file exists: {env_path}")
        with open(env_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    key = line.split('=')[0]
                    log.info(f"  - {key}: {'*' * 10}")
    else:
        log.warning(f"⚠️  .env file not found: {env_path}")
        log.info("Create a .env file with your API keys:")
        log.info("OPENAI_API_KEY=your_openai_api_key_here")
        log.info("TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here")
        log.info("CHANNEL_ID=your_channel_id_here")
    
    # Check environment variables
    api_key = import_config("OPENAI_API_KEY")
    if api_key:
        log.info(f"✅ OPENAI_API_KEY found in environment")
    else:
        log.warning("⚠️  OPENAI_API_KEY not found in environment")
    
    # Check required packages
    try:
        import openai
        log.info(f"✅ openai package installed: {openai.__version__}")
    except ImportError:
        log.error("❌ openai package not installed")
    
    try:
        import langchain_openai
        log.info("✅ langchain_openai package installed")
    except ImportError:
        log.error("❌ langchain_openai package not installed")
    
    try:
        from dotenv import load_dotenv
        log.info("✅ python-dotenv package installed")
    except ImportError:
        log.error("❌ python-dotenv package not installed")

def main():
    log.info("OpenAI API Test for CE Job Bot")
    log.info("=" * 50)
    
    check_environment()
    log.info("")
    
    if test_openai_connection():
        log.info("✅ All tests passed! OpenAI integration is working correctly.")
    else:
        log.error("❌ Some tests failed. Please check your API key and configuration.")
        log.info("")
        log.info("Troubleshooting steps:")
        log.info("1. Verify your OpenAI API key is correct")
        log.info("2. Check if you have sufficient API credits")
        log.info("3. Ensure your .env file is in the project root")
        log.info("4. Make sure all required packages are installed")

if __name__ == "__main__":
    main()


