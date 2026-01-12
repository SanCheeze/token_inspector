# scripts/enrich_trades.py

import asyncio
import os
from db.db import init_db_pool, enrich_wallets_trades_with_price
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")


async def main():
    pool = await init_db_pool(DB_URL)
    await enrich_wallets_trades_with_price(pool)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
