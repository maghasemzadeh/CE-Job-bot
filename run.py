import asyncio
from app.bot import build_app
import logging
import time
import sys
from app.config import settings

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)

log = logging.getLogger(__name__)

def main():
    """Main application entry point with comprehensive logging"""
    log.info("=" * 60)
    log.info("🚀 Starting CE Job Bot Application")
    log.info("=" * 60)
    
    # Log configuration
    log.info(f"Configuration:")
    log.info(f"  - Database URL: {settings.DATABASE_URL}")
    log.info(f"  - LLM Base URL: {settings.LLM_BASE_URL}")
    log.info(f"  - LLM Model: {settings.LLM_MODEL_NAME}")
    log.info(f"  - Embedding Model: {settings.EMBED_MODEL}")
    log.info(f"  - Similarity Threshold: {settings.SIM_THRESHOLD}")
    log.info(f"  - Admin IDs: {settings.ADMIN_IDS}")
    log.info(f"  - Channel ID: {settings.CHANNEL_ID}")
    log.info(f"  - Bot Token: {'Configured' if settings.TELEGRAM_BOT_TOKEN else 'Not configured'}")
    log.info(f"  - OpenAI API Key: {'Configured' if settings.OPENAI_API_KEY else 'Not configured'}")
    
    start_time = time.time()
    
    try:
        log.info("Building application...")
        app = build_app()
        
        build_time = time.time() - start_time
        log.info(f"✅ Application built successfully in {build_time:.2f}s")
        
        log.info("Starting bot polling...")
        log.info("=" * 60)
        
        app.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        log.info("🛑 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        total_time = time.time() - start_time
        log.error(f"❌ Application failed after {total_time:.2f}s: {e}")
        log.error(f"Error type: {type(e).__name__}")
        log.error(f"Error details: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
