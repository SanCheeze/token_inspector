# handlers/token_handlers.py

import io
import os
import asyncpg
from aiogram import Router, types
from aiogram.filters import Command
from datetime import datetime, timezone, timedelta

from db.db import (
    init_db_pool,
    wallets_exist_for_token,
    insider_buyers,
    filter_wallets_by_min_buy,
    filter_most_active_wallets,
    get_related_wallets
)

from scripts.token_analyser import inspect_token
from scripts.token_utils import get_wallet_history

from dotenv import load_dotenv

load_dotenv()


router = Router()
bot = None
DB_URL = os.getenv("DB_URL")


def setup_token_handlers(dp, in_bot):
    global bot
    bot = in_bot
    dp.include_router(router)


async def _ensure_token_data(pool, token: str, message: types.Message) -> None:
    exists = await wallets_exist_for_token(pool, token)
    if not exists:
        await message.answer(f"Данных по {token} нет — загружаю...")
        await inspect_token(token)


async def _get_db_pool() -> "asyncpg.Pool":
    if not DB_URL:
        raise RuntimeError("DB_URL environment variable is not set")
    return await init_db_pool(DB_URL)


# ------------------- /start -------------------
# @throttled(rate=1)
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я твой приватный бот для анализа токенов.\n\n"
        "Команды:\n"
        "/analyze <token_mint> — анализ токена\n"
        "/related — список общих кошельков между токенами\n"
        "/min_buy <token_mint> <usd> — кошельки, купившие от суммы\n"
        "/most_active_traders <token_mint> — самые активные трейдеры\n"
        "/wallet_history <wallet> — история токенов кошелька\n"
        "/insider_buyers — кошельки, после входа которых токен сделал 2х"
    )


# ------------------- /analyze -------------------
# @throttled(rate=1)
@router.message(Command("analyze"))
async def cmd_analyze(message: types.Message):
    text = message.text.replace("/analyze", "").strip()

    if not text:
        return await message.answer(
            "Отправь токены в формате:\n"
            "/analyze\nTOKEN1\nTOKEN2\nTOKEN3"
        )

    tokens = [t.strip() for t in text.replace(",", "\n").split("\n") if t.strip()]

    await message.answer(f"Получено токенов: {len(tokens)}\nНачинаю анализ...")

    results = []

    # ---- Последовательно ----
    for token in tokens:
        await message.answer(f"🔍 Анализирую {token}...")
        try:
            result = await inspect_token(token)
            results.append(f"• {token}: {result.get('message', 'OK')}")
        except Exception as e:
            results.append(f"• {token}: ❌ Ошибка — {e}")

    response = "\n".join(results)
    await message.answer(f"Готово!\n\n{response}")


# ------------------- /related -------------------
@router.message(Command("related"))
async def cmd_related(message: types.Message):
    """
    Форматы:
    /related
    TOKEN1
    TOKEN2
    ...

    /related <token_mint>
    """
    text = message.text.strip()
    lines = text.split("\n")

    # один токен или список
    token_list = [line.strip() for line in lines[1:] if line.strip()] \
        if len(lines) > 1 else message.text.strip().split()[1:]

    if not token_list:
        return await message.answer(
            "Использование:\n/related <token>\n\n/related\n<token_1>\n<token_2>..."
        )

    pool = await _get_db_pool()

    # Проверяем наличие данных
    for token in token_list:
        await _ensure_token_data(pool, token, message)

    # Получаем данные
    related = await get_related_wallets(pool, token_list)
    await pool.close()

    if not related:
        return await message.answer("Связанных кошельков не найдено.")

    # сортировка по count уже выполнена в get_related_wallets
    wallets_sorted = [f'{r["wallet"]}: {r["count"]}' for r in related]

    # TXT: только кошельки через \n
    txt_bytes = "\n".join(wallets_sorted).encode()
    txt_file = types.BufferedInputFile(
        txt_bytes,
        filename="related_wallets.txt"
    )

    await message.answer_document(txt_file, caption=f"Связанные кошельки {len(related)}")


# ------------------- /min_buy <token> <usd> -------------------
# @throttled(rate=1)
@router.message(Command("min_buy"))
async def cmd_min_buy(message: types.Message):
    args = message.text.strip().split()

    if len(args) != 3:
        return await message.answer("Использование:\n/min_buy <token_mint> <min_usd>")

    token_mint = args[1]
    try:
        min_usd = float(args[2])
    except:
        return await message.answer("min_usd должно быть числом.")

    await message.answer(f"Проверяем токен {token_mint}...")

    pool = await _get_db_pool()

    await _ensure_token_data(pool, token_mint, message)

    wallets = await filter_wallets_by_min_buy(pool, token_mint, min_usd)
    await pool.close()

    if not wallets:
        return await message.answer(f"Нет кошельков с покупками ≥ {min_usd}$")

    txt = "\n".join(wallets)
    txt_file = types.BufferedInputFile(
        txt.encode(),
        filename=f"min_buy_{token_mint}_{min_usd}.txt"
    )

    await message.answer_document(txt_file, caption=f"Кошельки с покупкой ≥ {min_usd}$")


# ------------------- /most_active_traders <token> -------------------
# @throttled(rate=1)
@router.message(Command("most_active_traders"))
async def cmd_most_active_traders(message: types.Message):
    args = message.text.strip().split()

    if len(args) != 2:
        return await message.answer("Использование:\n/most_active_traders <token_mint>")

    token_mint = args[1]

    await message.answer(f"Проверяем токен {token_mint}...")

    pool = await _get_db_pool()

    await _ensure_token_data(pool, token_mint, message)

    # минимум для попадания в активные трейдеры
    MIN_TRADES = 5

    wallets = await filter_most_active_wallets(pool, token_mint, MIN_TRADES)
    await pool.close()

    if not wallets:
        return await message.answer("Нет активных трейдеров.")

    txt = "\n".join(wallets)
    txt_file = types.BufferedInputFile(
        txt.encode(),
        filename=f"most_active_traders_{token_mint}.txt"
    )

    await message.answer_document(
        txt_file,
        caption=f"Кошельки с ≥ {MIN_TRADES} сделок"
    )


# ------------------- /wallet_history <wallet> [days] -------------------
@router.message(Command("wallet_history"))
async def cmd_wallet_history(message: types.Message):
    args = message.text.strip().split()

    if len(args) < 2:
        return await message.answer("Использование:\n/wallet_history <wallet_address> [days]")

    wallet = args[1].strip()
    last_n_days = 1  # default: 1 день

    if len(args) >= 3:
        try:
            last_n_days = int(args[2])
        except:
            return await message.answer("days должно быть числом.")

    await message.answer(f"Получаю токены кошелька {wallet} за последние {last_n_days} день(дней)...")

    # timestamp для ограничения по времени (UTC)
    stop_after_ts = int((datetime.now(timezone.utc) - timedelta(days=last_n_days)).timestamp())

    try:
        # вызываем готовую функцию
        tokens = await get_wallet_history(wallet, stop_after_ts)
    except Exception as e:
        return await message.answer(f"❌ Ошибка при получении данных: {e}")

    if not tokens:
        return await message.answer("Нет купленных токенов за указанный период.")

    # Формируем txt файл
    txt_bytes = "\n".join(tokens).encode()
    txt_file = types.BufferedInputFile(
        txt_bytes,
        filename=f"wallet_history_{wallet}.txt"
    )

    await message.answer_document(
        txt_file,
        caption=f"Купленные токены кошелька {wallet} (найдено {len(tokens)})"
    )


# ------------------- /insider_buyers -------------------
@router.message(Command("insider_buyers"))
async def cmd_insider_buyers(message: types.Message):
    """
    Форматы:
    /insider_buyers

    /insider_buyers
    TOKEN1
    TOKEN2
    TOKEN3
    """

    lines = message.text.strip().split("\n")
    tokens = [l.strip() for l in lines[1:] if l.strip()]

    await message.answer(
        "Считаю insider buyers..."
        + (f"\nТокенов: {len(tokens)}" if tokens else "\nПо всем токенам")
    )

    pool = await _get_db_pool()

    try:
        df = await insider_buyers(pool, tokens=tokens or None)
    finally:
        await pool.close()

    if df.empty:
        return await message.answer("Подходящих кошельков не найдено.")

    # CSV в памяти
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    csv_bytes = buffer.getvalue().encode()
    csv_file = types.BufferedInputFile(
        csv_bytes,
        filename="insider_buyers.csv"
    )

    await message.answer_document(
        csv_file,
        caption=f"Insider buyers: {len(df)} кошельков"
    )
