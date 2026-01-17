import json
from typing import List, Optional

import asyncpg
import pandas as pd

from scripts.trades_utils import add_price_to_trades
from scripts.token_utils import load_token_trades_solscan
from .pg import get_pool

WALLETS_COLUMNS = [
    "id", "token", "wallet", "pnl", "num_buys", "num_sells",
    "trades", "created_at"
]

TOKEN_TRADES_COLUMNS = [
    "signature",
    "ts",
    "action",
    "from",
    "token1",
    "amount1",
    "token1_decimals",
    "token2",
    "amount2",
    "token2_decimals",
    "value",
    "platforms",
    "sources",
]


def _ensure_pool(pool: asyncpg.Pool | None) -> asyncpg.Pool:
    return pool or get_pool()


# ============================================================
#  TOKENS TABLE
# ============================================================

async def token_exists(pool, token: str) -> bool:
    pool = _ensure_pool(pool)
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM tokens WHERE token=$1);",
            token
        )


async def save_token_metadata(pool: asyncpg.pool.Pool, metadata: dict):
    pool = _ensure_pool(pool)
    query = """
    INSERT INTO tokens (token, symbol, content, supply)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (token) DO UPDATE
    SET symbol = EXCLUDED.symbol,
        content = EXCLUDED.content,
        supply = COALESCE(EXCLUDED.supply, tokens.supply)
    """
    async with pool.acquire() as conn:
        await conn.execute(
            query,
            metadata["token"],
            metadata["symbol"],
            json.dumps(metadata["content"]),
            metadata.get("supply"),
        )


def resolve_supply_update(existing_supply: int | None, incoming_supply: int | None) -> int | None:
    return incoming_supply if incoming_supply is not None else existing_supply


async def update_token_supply(pool: asyncpg.pool.Pool, token: str, supply: int | None):
    pool = _ensure_pool(pool)
    if supply is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE tokens SET supply=$2 WHERE token=$1;",
            token,
            supply,
        )


async def load_tokens_missing_supply(pool: asyncpg.pool.Pool) -> list[str]:
    pool = _ensure_pool(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT token FROM tokens WHERE supply IS NULL;")
    return [row["token"] for row in rows]


async def load_token_metadata(pool, token: str) -> Optional[dict]:
    pool = _ensure_pool(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM tokens WHERE token=$1;", token)
        return dict(row) if row else None


def _normalize_json_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _serialize_token_trades(df: pd.DataFrame) -> list[dict]:
    trades = []
    for _, row in df.iterrows():
        trade = {
            "signature": _normalize_json_value(row.get("Signature")),
            "ts": _normalize_json_value(row.get("Human Time TS")),
            "action": _normalize_json_value(row.get("Action")),
            "from": _normalize_json_value(row.get("From")),
            "token1": _normalize_json_value(row.get("Token1")),
            "amount1": _normalize_json_value(row.get("Amount1")),
            "token1_decimals": _normalize_json_value(row.get("TokenDecimals1")),
            "token2": _normalize_json_value(row.get("Token2")),
            "amount2": _normalize_json_value(row.get("Amount2")),
            "token2_decimals": _normalize_json_value(row.get("TokenDecimals2")),
            "value": _normalize_json_value(row.get("Value")),
            "platforms": _normalize_json_value(row.get("Platforms")),
            "sources": _normalize_json_value(row.get("Sources")),
        }
        trades.append(trade)
    return trades


async def save_token_trades(pool, token: str, trades: list[dict]):
    pool = _ensure_pool(pool)
    query = """
    INSERT INTO tokens (token, trades, trades_updated_at)
    VALUES ($1, $2, NOW())
    ON CONFLICT (token) DO UPDATE
    SET trades = EXCLUDED.trades,
        trades_updated_at = NOW()
    """
    async with pool.acquire() as conn:
        await conn.execute(query, token, json.dumps(trades))


async def load_token_trades(pool, token: str) -> list[dict]:
    pool = _ensure_pool(pool)
    async with pool.acquire() as conn:
        trades = await conn.fetchval("SELECT trades FROM tokens WHERE token=$1;", token)
    if trades is None:
        return []
    if isinstance(trades, str):
        return json.loads(trades)
    return trades


async def token_trades_exist(pool, token: str) -> bool:
    pool = _ensure_pool(pool)
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1
                FROM tokens
                WHERE token=$1
                  AND trades IS NOT NULL
                  AND jsonb_array_length(trades) > 0
            );
            """,
            token,
        )


async def refresh_token_trades(
    pool,
    token: str,
    from_time: int | None = None,
    force: bool = False,
) -> list[dict]:
    pool = _ensure_pool(pool)
    if not force and await token_trades_exist(pool, token):
        return await load_token_trades(pool, token)

    df = await load_token_trades_solscan(token, from_time=from_time)
    if df.empty:
        await save_token_trades(pool, token, [])
        return []

    trades = _serialize_token_trades(df)
    await save_token_trades(pool, token, trades)
    return trades


# ============================================================
#  WALLETS TABLE
# ============================================================

async def wallets_exist_for_token(pool, token: str) -> bool:
    pool = _ensure_pool(pool)
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
    pool = _ensure_pool(pool)
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
    pool = _ensure_pool(pool)
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
    def normalize_trades(value):
        if isinstance(value, str):
            return json.loads(value)
        if value is None:
            return []
        return value

    df["trades"] = df["trades"].apply(normalize_trades)

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


async def get_related_wallets(pool, token_list) -> list[dict]:
    """
    Универсальная логика поиска связанных кошельков.

    Если token_list содержит 1 токен:
        - находим кошельки этого токена
        - ищем, где они встречаются в других токенах
        - считаем количество таких повторений

    Если token_list содержит 2+ токенов:
        - ищем пересечения среди этих токенов (как старая /related)
    """

    from collections import defaultdict

    pool = _ensure_pool(pool)

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

async def enrich_wallets_trades_with_price(pool: asyncpg.Pool | None):
    """
    Проходит по всей таблице wallets
    и добавляет поле price в каждый трейд
    """

    pool = _ensure_pool(pool)

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

    pool = _ensure_pool(pool)

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
