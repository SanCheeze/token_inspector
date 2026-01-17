# scripts/token_analyser.py

import asyncio
import os
import pandas as pd
from db import (
    get_pool,
    init_pg,
    save_wallets,
    load_wallets,
    save_token_metadata,
    token_exists,
    refresh_token_trades,
    load_token_trades,
)
from .token_utils import analyze_wallets_fifo, fetch_token_metadata
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")


async def inspect_token(token_mint: str):
    """
    Основная функция анализа токена:
    - загружает все транзакции токена через load_all_defi_activity
    - сохраняет метаданные токена
    - анализирует кошельки
    - сохраняет результат в базу wallets
    """

    pool = get_pool()

    # -------------------
    # Сохраняем метаданные токена (если ещё нет)
    # -------------------
    exists = await token_exists(pool, token_mint)

    if not exists:
        metadata = await fetch_token_metadata(token_mint)

        if not metadata:
            metadata = {
                "token": token_mint,
                "symbol": None,
                "content": [],
                "supply": None,
            }
        await save_token_metadata(pool, metadata)


    # -------------------
    # Проверяем, есть ли кошельки в базе
    # -------------------
    existing_wallets = await load_wallets(pool, token_mint)
    if not existing_wallets.empty:
        return {"message": f"Данные по токену {token_mint} уже есть в базе."}

    # -------------------
    # Загружаем/обновляем трейды токена
    # -------------------
    await refresh_token_trades(pool, token_mint)
    trades = await load_token_trades(pool, token_mint)
    if not trades:
        return {"message": f"Транзакции для токена {token_mint} не найдены."}

    df = pd.DataFrame(trades)
    df["Human Time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["From"] = df["from"]
    df["Token1"] = df["token1"]
    df["Amount1"] = df["amount1"]
    df["Token2"] = df["token2"]
    df["Amount2"] = df["amount2"]
    df["Value"] = df["value"]
    print(f"Всего строк транзакций: {len(df)}")

    # -------------------
    # Анализируем кошельки
    # -------------------
    wallets_df = analyze_wallets_fifo(df)

    # -------------------
    # Сохраняем в базу
    # -------------------
    await save_wallets(pool, token_mint, wallets_df)

    return {"message": f"Анализ токена {token_mint} завершён, {len(wallets_df)} кошельков сохранено."}


if __name__ == "__main__":
    import sys
    token = sys.argv[1] if len(sys.argv) > 1 else "QxSK4nJG2TQYoJoTyjjhcePAy1vgE9HbD6inTKWpump"

    async def _main():
        await init_pg(DB_URL)
        try:
            await inspect_token(token)
        finally:
            pool = get_pool()
            await pool.close()

    asyncio.run(_main())
