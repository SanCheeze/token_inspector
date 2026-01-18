from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import asyncpg
import pandas as pd

from ml.config import SUPPLY_COLUMN, TOKEN_COLUMN, TRADES_COLUMN


async def load_tokens_df(
    pool: asyncpg.Pool,
    limit: int | None = None,
    where_sql: str | None = None,
) -> pd.DataFrame:
    conditions = [
        f"{SUPPLY_COLUMN} IS NOT NULL",
        f"{SUPPLY_COLUMN} > 0",
        f"{TRADES_COLUMN} IS NOT NULL",
        f"jsonb_array_length({TRADES_COLUMN}) > 0",
    ]
    if where_sql:
        conditions.append(f"({where_sql})")
    where_clause = " AND ".join(conditions)

    rows: list[dict[str, Any]] = []
    last_id = 0
    fetched = 0
    batch_size = 1000

    while True:
        remaining = None
        if limit is not None:
            remaining = max(limit - fetched, 0)
            if remaining == 0:
                break
        batch_limit = min(batch_size, remaining) if remaining is not None else batch_size

        query = f"""
            SELECT id, {TOKEN_COLUMN} AS token, {SUPPLY_COLUMN} AS supply, {TRADES_COLUMN} AS trades
            FROM tokens
            WHERE id > $1 AND {where_clause}
            ORDER BY id
            LIMIT $2
        """
        batch = await pool.fetch(query, last_id, batch_limit)
        if not batch:
            break
        for record in batch:
            rows.append(dict(record))
        last_id = batch[-1]["id"]
        fetched += len(batch)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop(columns=["id"], errors="ignore")
    return df


async def load_bundle_wallets(
    pool: asyncpg.Pool, bundle_name: str | None = None
) -> set[str] | None:
    file_path = Path("ml/data/bundle_wallets.txt")
    if not file_path.exists():
        return None

    wallets: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        wallet = line.strip()
        if wallet:
            wallets.add(wallet)
    if bundle_name:
        _ = bundle_name
    return wallets if wallets else None


async def load_token_by_mint(
    pool: asyncpg.Pool, token_mint: str
) -> dict[str, Any] | None:
    query = f"""
        SELECT {TOKEN_COLUMN} AS token, {SUPPLY_COLUMN} AS supply, {TRADES_COLUMN} AS trades
        FROM tokens
        WHERE {TOKEN_COLUMN} = $1
          AND {SUPPLY_COLUMN} IS NOT NULL
          AND {SUPPLY_COLUMN} > 0
          AND {TRADES_COLUMN} IS NOT NULL
          AND jsonb_array_length({TRADES_COLUMN}) > 0
        LIMIT 1
    """
    record = await pool.fetchrow(query, token_mint)
    return dict(record) if record else None


def parse_trades_payload(raw_trades: Any) -> list[dict[str, Any]]:
    if raw_trades is None:
        return []
    if isinstance(raw_trades, str):
        try:
            parsed = json.loads(raw_trades)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(raw_trades, list):
        return raw_trades
    return []
