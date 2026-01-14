# db/db.py

import json
import asyncpg
import pandas as pd
from typing import Dict, Any, List, Optional

from scripts.trades_utils import add_price_to_trades



# ============================================================
#  ИНИЦИАЛИЗАЦИЯ БАЗЫ
# ============================================================

WALLETS_COLUMNS = [
    "id", "token", "wallet", "pnl", "num_buys", "num_sells",
    "trades", "created_at"
]


async def init_db_pool(db_url: str) -> asyncpg.pool.Pool:
    return await asyncpg.create_pool(dsn=db_url)


async def init_tables(pool: asyncpg.pool.Pool):
    async with pool.acquire() as conn:

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id SERIAL PRIMARY KEY,
            token TEXT UNIQUE NOT NULL,
            symbol TEXT,
            content JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS wallets (
            id SERIAL PRIMARY KEY,
            token TEXT NOT NULL,
            wallet TEXT NOT NULL,
            pnl DOUBLE PRECISION,
            num_buys INTEGER,
            num_sells INTEGER,
            trades JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(token, wallet)
        );
        """)

        print("Tables 'tokens' and 'wallets' are ready.")


# ============================================================
#  TOKENS TABLE
# ============================================================

async def token_exists(pool, token: str) -> bool:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM tokens WHERE token=$1);",
            token
        )


async def save_token_metadata(pool: asyncpg.pool.Pool, metadata: dict):
    query = """
    INSERT INTO tokens (token, symbol, content)
    VALUES ($1, $2, $3)
    ON CONFLICT (token) DO UPDATE
    SET symbol = EXCLUDED.symbol,
        content = EXCLUDED.content
    """
    async with pool.acquire() as conn:
        await conn.execute(query, metadata["token"], metadata["symbol"], json.dumps(metadata["content"]))


async def load_token_metadata(pool, token: str) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM tokens WHERE token=$1;", token)
        return dict(row) if row else None


# ============================================================
#  WALLETS TABLE
# ============================================================

async def wallets_exist_for_token(pool, token: str) -> bool:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM wallets WHERE token=$1);",
            token
        )


async def save_wallets(pool, token: str, wallets_df: pd.DataFrame):
    """
    Сохраняем кошельки.
    trades — JSONB, поэтому нужно передавать как строку JSON.
    """
    records = []

    for _, row in wallets_df.iterrows():
        wallet_value = row["wallet"]
        if pd.isna(wallet_value):
            wallet_value = ""  # или None, если колонка допускает NULL

        trades_value = row.get("trades", [])
        # если trades - строка JSON, загружаем её, иначе оставляем список
        if isinstance(trades_value, str):
            try:
                trades_value = json.loads(trades_value)
            except json.JSONDecodeError:
                trades_value = []
        elif not isinstance(trades_value, list):
            trades_value = []

        record = (
            token,
            str(wallet_value),
            float(row["pnl"]) if pd.notna(row.get("pnl")) else None,
            int(row["num_buys"]) if pd.notna(row.get("num_buys")) else None,
            int(row["num_sells"]) if pd.notna(row.get("num_sells")) else None,
            json.dumps(trades_value)  # <- ВАЖНО! JSONB через строку
        )
        records.append(record)

    async with pool.acquire() as conn:
        await conn.executemany("""
            INSERT INTO wallets (
                token, wallet, pnl, num_buys, num_sells, trades
            )
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (token, wallet) DO UPDATE SET
                pnl = EXCLUDED.pnl,
                num_buys = EXCLUDED.num_buys,
                num_sells = EXCLUDED.num_sells,
                trades = EXCLUDED.trades;
        """, records)

    print(f"Saved {len(records)} wallets for token {token}.")


async def load_wallets(pool, token: str | None = None) -> pd.DataFrame:
    """
    Загружаем кошельки как pandas DataFrame.
    Если token=None — загружаем все кошельки.
    """
    async with pool.acquire() as conn:
        if token is None:
            rows = await conn.fetch("SELECT * FROM wallets;")
        else:
            rows = await conn.fetch(
                "SELECT * FROM wallets WHERE token=$1;",
                token
            )

    if not rows:
        return pd.DataFrame(columns=WALLETS_COLUMNS)

    # asyncpg.Record → dict
    df = pd.DataFrame([dict(r) for r in rows])

    # сортируем столбцы
    df = df.reindex(columns=WALLETS_COLUMNS)

    # trades уже Python list (JSONB)
    df["trades"] = df["trades"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x or [])
    )

    return df


# ============================================================
#  FILTERS
# ============================================================

async def get_all_wallets_list(pool, token: str) -> List[str]:
    df = await load_wallets(pool, token)
    return df["wallet"].dropna().tolist()


async def filter_wallets_by_min_buy(pool, token: str, min_usd: float) -> List[str]:
    df = await load_wallets(pool, token)
    result = []

    for _, row in df.iterrows():
        trades = row["trades"] or []
        for t in trades:
            if t.get("side") == "buy" and t.get("usd_value", 0) >= min_usd:
                result.append(row["wallet"])
                break

    return result


async def filter_most_active_wallets(pool, token: str, min_trades: int) -> List[str]:
    df = await load_wallets(pool, token)
    return [
        row["wallet"]
        for _, row in df.iterrows()
        if len(row["trades"] or []) >= min_trades
    ]


async def get_related_wallets(pool, token_list):
    """
    Универсальная логика поиска связанных кошельков.

    Если token_list содержит 1 токен:
        - находим кошельки этого токена
        - ищем, где они встречаются в других токенах
        - считаем количество таких повторений

    Если token_list содержит 2+ токенов:
        - ищем пересечения среди этих токенов (как старая /related)
    """

    from collections import defaultdict, Counter

    # -----------------------------
    # ЗАГРУЖАЕМ ВСЕ КОШЕЛЬКИ ИЗ ВХОДНЫХ ТОКЕНОВ
    # -----------------------------
    rows = await pool.fetch("""
        SELECT wallet, token
        FROM wallets
        WHERE token = ANY($1::text[])
    """, token_list)

    if not rows:
        return []  # пусто

    df = [(r["wallet"], r["token"]) for r in rows]

    # -------------------------------------
    # СЛУЧАЙ 1 — ОДИН ВХОДНОЙ ТОКЕН
    # -------------------------------------
    if len(token_list) == 1:
        base_token = token_list[0]

        # кошельки токена
        base_wallets = {w for w, t in df}

        # ищем эти кошельки в других токенах
        rows_other = await pool.fetch("""
            SELECT wallet, token
            FROM wallets
            WHERE wallet = ANY($1::text[])
              AND token != $2
        """, list(base_wallets), base_token)

        counter = defaultdict(list)
        for r in rows_other:
            counter[r["wallet"]].append(r["token"])

        # превращаем в удобный список
        result = [
            {
                "wallet": w,
                "count": len(tokens),
                "tokens": sorted(set(tokens))
            }
            for w, tokens in counter.items()
        ]

        # сортировка по убыванию count
        result.sort(key=lambda x: x["count"], reverse=True)

        return result

    # -------------------------------------
    # СЛУЧАЙ 2 — НЕСКОЛЬКО ТОКЕНОВ
    # (как старая /related)
    # -------------------------------------

    grouped = defaultdict(set)
    for wallet, token in df:
        grouped[wallet].add(token)

    result = []
    for wallet, tokens in grouped.items():
        if len(tokens) > 1:
            result.append({
                "wallet": wallet,
                "count": len(tokens),
                "tokens": sorted(tokens)
            })

    # сортировка
    result.sort(key=lambda x: x["count"], reverse=True)

    return result


# ============================================================
#  ADD PRICES TO TRADES
# ============================================================

async def enrich_wallets_trades_with_price(pool: asyncpg.pool.Pool):
    """
    Проходит по всей таблице wallets
    и добавляет поле price в каждый трейд
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, trades
            FROM wallets
        """)

    if not rows:
        print("No wallets found.")
        return

    updates = []

    for r in rows:
        wallet_id = r["id"]
        trades = r["trades"]

        new_trades = add_price_to_trades(trades)

        updates.append((
            json.dumps(new_trades),
            wallet_id
        ))

    async with pool.acquire() as conn:
        await conn.executemany("""
            UPDATE wallets
            SET trades = $1
            WHERE id = $2
        """, updates)

    print(f"Updated trades with price for {len(updates)} wallets.")


# ============================================================
#  INSIDER BUYERS
# ============================================================

async def insider_buyers(pool, tokens: list[str] | None = None) -> pd.DataFrame:
    """
    Считает, сколько раз токен сделал x2 после входа кошелька.

    tokens:
        None  -> использовать ВСЮ таблицу wallets
        list  -> использовать только записи по этим токенам
    """

    # -------------------------------------------------
    # ЗАГРУЗКА ДАННЫХ
    # -------------------------------------------------
    if not tokens:
        df = await load_wallets(pool, token=None)
    else:
        dfs = []
        for token in tokens:
            df_token = await load_wallets(pool, token=token)
            if not df_token.empty:
                dfs.append(df_token)

        if not dfs:
            return pd.DataFrame(columns=["wallet", "count"])

        df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=["wallet", "count"])

    # -------------------------------------------------
    # EXPLODE trades
    # -------------------------------------------------
    df = df.explode("trades").dropna(subset=["trades"])

    trades_df = pd.json_normalize(df["trades"])
    df = pd.concat(
        [df[["wallet", "token"]].reset_index(drop=True), trades_df],
        axis=1
    )

    # -------------------------------------------------
    # ФИЛЬТР ВАЛИДНЫХ ТРЕЙДОВ
    # -------------------------------------------------
    if "price" not in df.columns:
        if {"usd_value", "base_amount"}.issubset(df.columns):
            base_amount = df["base_amount"].replace(0, pd.NA)
            df["price"] = df["usd_value"] / base_amount
        else:
            return pd.DataFrame(columns=["wallet", "count"])
    elif {"usd_value", "base_amount"}.issubset(df.columns):
        base_amount = df["base_amount"].replace(0, pd.NA)
        derived_price = df["usd_value"] / base_amount
        df["price"] = df["price"].fillna(derived_price)

    if not {"ts", "side", "price"}.issubset(df.columns):
        return pd.DataFrame(columns=["wallet", "count"])

    df = df[
        (df["price"].notna()) &
        (df["price"] > 0) &
        (df["ts"].notna()) &
        (df["side"] == "buy")
    ]

    if df.empty:
        return pd.DataFrame(columns=["wallet", "count"])

    df = df.sort_values("ts")

    # -------------------------------------------------
    # ПЕРВЫЙ BUY (entry)
    # -------------------------------------------------
    first_buys = (
        df
        .groupby(["wallet", "token"], as_index=False)
        .first()
        .rename(columns={"price": "entry_price", "ts": "entry_ts"})
        [["wallet", "token", "entry_price", "entry_ts"]]
    )

    # -------------------------------------------------
    # МАКС ЦЕНА ПОСЛЕ ВХОДА
    # -------------------------------------------------
    max_price = (
        df
        .merge(first_buys, on=["wallet", "token"], how="inner")
        .query("ts >= entry_ts")
        .groupby(["wallet", "token"], as_index=False)["price"]
        .max()
        .rename(columns={"price": "max_price"})
    )

    # -------------------------------------------------
    # УСЛОВИЕ X2
    # -------------------------------------------------
    x2 = max_price.merge(first_buys, on=["wallet", "token"])
    x2 = x2[x2["max_price"] >= x2["entry_price"] * 2]

    # -------------------------------------------------
    # COUNT ПО КОШЕЛЬКУ
    # -------------------------------------------------
    result = (
        x2
        .groupby("wallet")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return result
