# handlers/token_handlers.py

import io
import asyncio
from aiogram import Router, types
from aiogram.filters import Command
from datetime import datetime, timezone, timedelta

from db import (
    get_pool,
    wallets_exist_for_token,
    insider_buyers,
    filter_wallets_by_min_buy,
    filter_most_active_wallets,
    get_related_wallets
)

from scripts.token_analyser import inspect_token
from scripts.token_utils import get_wallet_history

router = Router()
bot = None
BASE58_CHARS = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def setup_token_handlers(dp, in_bot):
    global bot
    bot = in_bot
    dp.include_router(router)


async def _ensure_token_data(pool, token: str, message: types.Message) -> None:
    exists = await wallets_exist_for_token(pool, token)
    if not exists:
        await message.answer(f"Данных по {token} нет — загружаю...")
        await inspect_token(token)


def _is_probably_wallet(address: str) -> bool:
    if not (32 <= len(address) <= 44):
        return False
    return all(char in BASE58_CHARS for char in address)


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

    pool = get_pool()

    # Проверяем наличие данных
    for token in token_list:
        await _ensure_token_data(pool, token, message)

    # Получаем данные
    related = await get_related_wallets(pool, token_list)
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

    pool = get_pool()

    await _ensure_token_data(pool, token_mint, message)

    wallets = await filter_wallets_by_min_buy(pool, token_mint, min_usd)
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

    pool = get_pool()

    await _ensure_token_data(pool, token_mint, message)

    # минимум для попадания в активные трейдеры
    MIN_TRADES = 5

    wallets = await filter_most_active_wallets(pool, token_mint, MIN_TRADES)
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
    lines = message.text.strip().split("\n")
    first_line = lines[0].strip()
    first_parts = first_line.split()
    last_n_days = 1

    single_mode = False
    if len(first_parts) >= 2 and (len(first_parts) >= 3 or not first_parts[1].isdigit()):
        single_mode = True

    if single_mode:
        wallet = first_parts[1].strip()
        if len(first_parts) >= 3:
            try:
                last_n_days = int(first_parts[2])
            except ValueError:
                return await message.answer("days должно быть числом.")

        if not _is_probably_wallet(wallet):
            return await message.answer("Некорректный адрес кошелька.")

        await message.answer(
            f"Получаю токены кошелька {wallet} за последние {last_n_days} день(дней)..."
        )

        stop_after_ts = int((datetime.now(timezone.utc) - timedelta(days=last_n_days)).timestamp())

        try:
            tokens = await get_wallet_history(wallet, stop_after_ts)
        except Exception as e:
            return await message.answer(f"❌ Ошибка при получении данных: {e}")

        if not tokens:
            return await message.answer("Нет купленных токенов за указанный период.")

        txt_bytes = "\n".join(tokens).encode()
        txt_file = types.BufferedInputFile(
            txt_bytes,
            filename=f"wallet_history_{wallet}.txt"
        )

        return await message.answer_document(
            txt_file,
            caption=f"Купленные токены кошелька {wallet} (найдено {len(tokens)})"
        )

    if len(first_parts) >= 2:
        try:
            last_n_days = int(first_parts[1])
        except ValueError:
            return await message.answer("days должно быть числом.")

    wallets = [line.strip() for line in lines[1:] if line.strip()]

    if not wallets:
        return await message.answer(
            "Использование:\n/wallet_history <wallet_address> [days]\n\n"
            "/wallet_history [days]\n<wallet_1>\n<wallet_2>"
        )

    unique_wallets = list(dict.fromkeys(wallets))
    valid_wallets = []
    bad_wallets = []
    for wallet in unique_wallets:
        if _is_probably_wallet(wallet):
            valid_wallets.append(wallet)
        else:
            bad_wallets.append(wallet)

    if not valid_wallets:
        return await message.answer("Нет валидных кошельков для проверки.")

    if len(unique_wallets) > 20:
        return await message.answer("Слишком много кошельков. Максимум: 20")

    await message.answer(
        f"Получаю токены для {len(valid_wallets)} кошельков за последние {last_n_days} день(дней)..."
    )

    stop_after_ts = int((datetime.now(timezone.utc) - timedelta(days=last_n_days)).timestamp())
    semaphore = asyncio.Semaphore(4)

    async def fetch_wallet_history(wallet: str):
        async with semaphore:
            return await get_wallet_history(wallet, stop_after_ts)

    results = await asyncio.gather(
        *(fetch_wallet_history(wallet) for wallet in valid_wallets),
        return_exceptions=True
    )

    tokens = []
    seen_tokens = set()
    failed_wallets = list(bad_wallets)
    wallets_ok = 0

    for wallet, result in zip(valid_wallets, results):
        if isinstance(result, Exception):
            failed_wallets.append(wallet)
            continue

        wallets_ok += 1
        for token in result:
            if token not in seen_tokens:
                seen_tokens.add(token)
                tokens.append(token)

    if wallets_ok == 0:
        return await message.answer("❌ Ошибка при получении данных по всем кошелькам.")

    if not tokens:
        return await message.answer("Нет купленных токенов за указанный период.")

    total_wallets = len(valid_wallets) + len(bad_wallets)
    txt_bytes = "\n".join(tokens).encode()
    filename = f"wallet_history_{total_wallets}_wallets.txt"
    txt_file = types.BufferedInputFile(
        txt_bytes,
        filename=filename
    )

    await message.answer_document(
        txt_file,
        caption=(
            f"Уникальные токены (найдено {len(tokens)}) | "
            f"кошельков: {total_wallets} | ошибок: {len(failed_wallets)}"
        )
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

    pool = get_pool()
    df = await insider_buyers(pool, tokens=tokens or None)

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
