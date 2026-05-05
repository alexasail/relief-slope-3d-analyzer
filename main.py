"""
Точка входа ReliefAnalyzer: запуск Telegram-бота.
Модули: analyzer.py, bot.py, parse.py, main.py (paper.md).
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_bot():
    from bot import main as bot_main
    await bot_main()


def main():
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.error("Задайте TELEGRAM_BOT_TOKEN в .env")
        sys.exit(1)
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")


if __name__ == "__main__":
    main()
