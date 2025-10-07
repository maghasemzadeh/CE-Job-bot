import asyncio
from app.bot import build_app
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    app = build_app()
    app.run_polling(drop_pending_updates=True)
