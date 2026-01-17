# scripts/backfill_token_supply.py

import asyncio
import logging
import os
from decimal import Decimal

from dotenv import load_dotenv

from db import get_pool, init_pg, load_tokens_missing_supply, update_token_supply
from scripts.token_utils import fetch_token_metadata


load_dotenv()

LOGGER = logging.getLogger(__name__)
DB_URL = os.getenv("DB_URL")
MAX_CONCURRENCY = int(os.getenv("SUPPLY_BACKFILL_CONCURRENCY", "20"))
MAX_RETRIES = int(os.getenv("SUPPLY_BACKFILL_RETRIES", "3"))
BACKOFF_SECONDS = float(os.getenv("SUPPLY_BACKFILL_BACKOFF", "1.5"))


async def _fetch_supply_with_retry(token: str) -> Decimal | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            metadata = await fetch_token_metadata(token)
            return metadata.get("supply")
        except Exception as exc:
            wait_time = BACKOFF_SECONDS * (2 ** (attempt - 1))
            LOGGER.warning(
                "Ошибка получения supply для %s (attempt %s/%s): %s",
                token,
                attempt,
                MAX_RETRIES,
                exc,
            )
            await asyncio.sleep(wait_time)
    return None


async def backfill_supply():
    pool = get_pool()
    tokens = await load_tokens_missing_supply(pool)
    if not tokens:
        LOGGER.info("Нет токенов без supply.")
        return

    LOGGER.info("Найдено токенов без supply: %s", len(tokens))
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _worker(token: str):
        async with semaphore:
            supply = await _fetch_supply_with_retry(token)
            if supply is None:
                LOGGER.warning("Supply не найден для %s", token)
                return
            await update_token_supply(pool, token, supply)
            LOGGER.info("Supply обновлён для %s: %s", token, supply)

    await asyncio.gather(*[_worker(token) for token in tokens])


async def main():
    await init_pg(DB_URL)
    try:
        await backfill_supply()
    finally:
        pool = get_pool()
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
