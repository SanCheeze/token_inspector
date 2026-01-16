# scripts/enrich_trades.py

import asyncio
import os
from db import init_pg, get_pool, enrich_wallets_trades_with_price
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")


async def main():
    await init_pg(DB_URL)
    pool = get_pool()
    try:
        await enrich_wallets_trades_with_price(pool)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
