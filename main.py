# main.py
import os
import asyncio
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram import Bot, Dispatcher
from handlers.decorators import ThrottlingMiddleware
from handlers.token_handlers import setup_token_handlers
from db import init_pg
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("WEBAPP_PORT", 8080))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
DB_URL = os.getenv("DB_URL")


async def init_db():
    if not DB_URL:
        raise RuntimeError("DB_URL environment variable is not set")
    await init_pg(DB_URL)


async def set_webhook(bot: Bot):
    """
    Устанавливаем вебхук и сбрасываем старые апдейты
    """
    await bot.set_webhook(f"https://your-domain.com{WEBHOOK_PATH}", drop_pending_updates=True)
    print("Webhook set and old updates dropped.")


async def local_debug_via_polling():
    """
    Локальный режим с polling. Старые апдейты игнорируются.
    """
    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    # dp.message.middleware(ThrottlingMiddleware())
    setup_token_handlers(dp, bot)
    dp.startup.register(init_db)

    # Удаляем возможный старый webhook
    await bot.delete_webhook(drop_pending_updates=True)
    print('Bot started (polling mode). Old updates dropped.')

    # polling с skip_updates=True, workers=1 для последовательной обработки
    await dp.start_polling(bot, skip_updates=True, workers=1)


def production_version_via_webhook():
    """
    Продакшн режим с webhook.
    """
    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.message.middleware(ThrottlingMiddleware())
    setup_token_handlers(dp, bot)
    dp.startup.register(init_db)
    dp.startup.register(lambda: set_webhook(bot))

    app = web.Application()

    webhook_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)
    webhook_handler.register(app, path=WEBHOOK_PATH)

    setup_application(app, dp, bot=bot)

    print(f"Running webhook server on {WEBAPP_HOST}:{WEBAPP_PORT}")
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)


if __name__ == "__main__":
    asyncio.run(local_debug_via_polling())
    # production_version_via_webhook()
