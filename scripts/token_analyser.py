# scripts/token_analyser.py

import os
import asyncio
import pandas as pd
from db.db import init_db_pool, init_tables, save_wallets, load_wallets, save_token_metadata, token_exists
from .token_utils import load_all_defi_activity, analyze_wallets_fifo, fetch_token_metadata
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

    pool = await init_db_pool(DB_URL)
    await init_tables(pool)

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
                "content": []
            }
        await save_token_metadata(pool, metadata)


    # -------------------
    # Проверяем, есть ли кошельки в базе
    # -------------------
    existing_wallets = await load_wallets(pool, token_mint)
    if not existing_wallets.empty:
        await pool.close()
        return {"message": f"Данные по токену {token_mint} уже есть в базе."}

    # -------------------
    # Загружаем все транзакции токена
    # -------------------
    output_file = f"data/{token_mint}_all_defi_activity.csv"
    df = await load_all_defi_activity(token_mint, output_file=output_file, save_dir="pages")
    print(f"Всего строк транзакций: {len(df)}")

    if df.empty:
        await pool.close()
        return {"message": f"Транзакции для токена {token_mint} не найдены."}

    # -------------------
    # Анализируем кошельки
    # -------------------
    wallets_df = analyze_wallets_fifo(df)

    # -------------------
    # Сохраняем в базу
    # -------------------
    await save_wallets(pool, token_mint, wallets_df)

    await pool.close()
    return {"message": f"Анализ токена {token_mint} завершён, {len(wallets_df)} кошельков сохранено."}


if __name__ == "__main__":
    import sys
    token = sys.argv[1] if len(sys.argv) > 1 else "QxSK4nJG2TQYoJoTyjjhcePAy1vgE9HbD6inTKWpump"
    asyncio.run(inspect_token(token))
