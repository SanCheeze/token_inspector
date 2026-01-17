from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import asyncpg
import pandas as pd

from ml.config import (
    CREATED_TS_COLUMN,
    FIRST_TRADE_TS_COLUMN,
    MAX_MARKET_CAP_COLUMN,
    ML_PRED_COLUMN,
    TOKEN_COLUMN,
    TOKENS_TABLE,
    TRADES_COLUMN,
)

logger = logging.getLogger(__name__)


async def load_tokens(
    pool: asyncpg.Pool,
    limit: int | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    limit_value = limit or 1000
    available_columns = await _get_columns(pool, TOKENS_TABLE)
    select_columns = [_select_alias(TOKEN_COLUMN, available_columns)]
    select_columns.append(_select_alias(TRADES_COLUMN, available_columns))
    token_filter_column = _resolve_filter_column(TOKEN_COLUMN, available_columns)
    trades_filter_column = _resolve_filter_column(TRADES_COLUMN, available_columns)
    if MAX_MARKET_CAP_COLUMN in available_columns:
        select_columns.append(MAX_MARKET_CAP_COLUMN)
    if "supply" in available_columns:
        select_columns.append("supply")
    for optional in (CREATED_TS_COLUMN, FIRST_TRADE_TS_COLUMN):
        if optional in available_columns:
            select_columns.append(optional)
    order_column = CREATED_TS_COLUMN if CREATED_TS_COLUMN in available_columns else "id"
    where_clauses = ["($1::text IS NULL OR {col} = $1)".format(col=token_filter_column)]
    if "supply" in available_columns:
        where_clauses.append("supply IS NOT NULL")
    if trades_filter_column:
        where_clauses.append(f"{trades_filter_column} IS NOT NULL")
        where_clauses.append(f"jsonb_array_length({trades_filter_column}) > 0")
    query = f"""
        SELECT {", ".join(select_columns)}
        FROM {TOKENS_TABLE}
        WHERE {" AND ".join(where_clauses)}
        ORDER BY {order_column} DESC NULLS LAST
        LIMIT $2;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, token, limit_value)

    records = [dict(row) for row in rows]
    df = pd.DataFrame(records)
    if df.empty:
        columns = [TOKEN_COLUMN, TRADES_COLUMN]
        if MAX_MARKET_CAP_COLUMN in available_columns:
            columns.append(MAX_MARKET_CAP_COLUMN)
        if "supply" in available_columns:
            columns.append("supply")
        return pd.DataFrame(columns=columns)

    df[TRADES_COLUMN] = df[TRADES_COLUMN].apply(_parse_trades)
    return df


async def _get_columns(pool: asyncpg.Pool, table_name: str) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = $1;
            """,
            table_name,
        )
    return {row["column_name"] for row in rows}


def _parse_trades(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            logger.warning("Failed to parse trades JSON payload")
            return []
    return list(raw) if isinstance(raw, (tuple, set)) else []


def _select_alias(column: str, available_columns: set[str]) -> str:
    if column in available_columns:
        return column
    if column != "token" and "token" in available_columns and column == TOKEN_COLUMN:
        return f"token AS {TOKEN_COLUMN}"
    if column != "trades" and "trades" in available_columns and column == TRADES_COLUMN:
        return f"trades AS {TRADES_COLUMN}"
    return column


def _resolve_filter_column(column: str, available_columns: set[str]) -> str | None:
    if column in available_columns:
        return column
    if column != "token" and "token" in available_columns and column == TOKEN_COLUMN:
        return "token"
    if column != "trades" and "trades" in available_columns and column == TRADES_COLUMN:
        return "trades"
    return None


async def load_bundle_wallets(pool: asyncpg.Pool, bundle_name: str | None = None) -> set[str]:
    file_path = Path("ml/data/bundle_wallets.txt")
    if not file_path.exists():
        logger.info("Bundle wallets file not found, returning empty set")
        return set()
    wallets = {
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if bundle_name:
        logger.info("Bundle name '%s' ignored for file-based wallets", bundle_name)
    return wallets


async def save_token_prediction(
    pool: asyncpg.Pool,
    token: str,
    model_version: str,
    pred_mcap: float,
    pred_log: float,
    features: dict[str, float] | None = None,
) -> bool:
    async with pool.acquire() as conn:
        column_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = $1 AND column_name = $2
            );
            """,
            TOKENS_TABLE,
            ML_PRED_COLUMN,
        )
        if not column_exists:
            logger.info("Column %s not found; skipping prediction save", ML_PRED_COLUMN)
            return False

        payload = json.dumps(
            {
                "model_version": model_version,
                "pred_mcap": pred_mcap,
                "pred_log": pred_log,
                "features": features,
            }
        )
        await conn.execute(
            f"UPDATE {TOKENS_TABLE} SET {ML_PRED_COLUMN} = $2::jsonb WHERE {TOKEN_COLUMN} = $1;",
            token,
            payload,
        )
    return True
