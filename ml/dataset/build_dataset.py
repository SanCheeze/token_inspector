from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import asyncpg

from db.pg import get_pool, init_pg
from ml.config import (
    CREATED_TS_COLUMN,
    FIRST_TRADE_TS_COLUMN,
    MAX_MARKET_CAP_COLUMN,
    TARGET_COLUMN,
    TOKEN_COLUMN,
    TRADES_COLUMN,
)
from ml.features.extractor import extract_features
from ml.features_v1 import build_features
from ml.schema import FEATURE_NAMES
from ml.dataset.db import load_bundle_wallets, load_tokens
from ml.targets import build_target
from ml.trade_adapter import normalize_trade

logger = logging.getLogger(__name__)


def build_dataset(
    df_tokens: pd.DataFrame,
    bundle_wallets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_rows: list[dict[str, float]] = []
    target_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    skipped_empty = 0
    skipped_window = 0
    skipped_target = 0

    for _, row in df_tokens.iterrows():
        trades = _parse_trades(row.get(TRADES_COLUMN))
        trades = [normalize_trade(trade) for trade in trades]
        if not trades:
            skipped_empty += 1
            continue

        token_mint = row.get(TOKEN_COLUMN)
        if not token_mint:
            skipped_empty += 1
            continue

        t0 = _resolve_t0(row, trades)
        features = extract_features(trades, token_mint, t0, bundle_wallets=bundle_wallets)
        if features.get("f_trades_total", 0.0) == 0.0:
            skipped_window += 1
            continue

        supply = _coerce_decimal(row.get("supply"))
        max_market_cap = _compute_max_usd_mcap(trades, token_mint, t0, supply)
        if max_market_cap is None or float(max_market_cap) <= 0:
            skipped_target += 1
            continue

        feature_rows.append(features)
        target_rows.append(float(np.log1p(float(max_market_cap))))
        meta_rows.append(
            {
                TOKEN_COLUMN: token_mint,
                "t0": t0,
                "trades_total": features.get("f_trades_total", 0.0),
                MAX_MARKET_CAP_COLUMN: float(max_market_cap),
            }
        )

    logger.info(
        "Tokens loaded: %s | empty trades: %s | empty window: %s | missing target: %s",
        len(df_tokens),
        skipped_empty,
        skipped_window,
        skipped_target,
    )
    logger.info("Dataset size: %s", len(feature_rows))

    features_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    target_series = pd.Series(target_rows, name=TARGET_COLUMN)
    meta_df = pd.DataFrame(meta_rows)

    return features_df, target_series, meta_df


async def build_dataset_from_db(
    pool: asyncpg.Pool,
    limit: int,
    bundle_name: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df_tokens = await load_tokens(pool, limit=limit)
    bundle_wallets = await load_bundle_wallets(pool, bundle_name)
    return build_dataset(df_tokens, bundle_wallets=bundle_wallets)


async def _fetch_tokens_batch_v1(
    pool: asyncpg.Pool,
    last_id: int,
    batch_size: int,
) -> list[dict]:
    query = """
        SELECT id, token, supply, trades
        FROM tokens
        WHERE id > $1
          AND supply IS NOT NULL
          AND trades IS NOT NULL
          AND jsonb_array_length(trades) > 0
        ORDER BY id ASC
        LIMIT $2;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, last_id, batch_size)
    return [dict(row) for row in rows]


def _load_bundle_wallets_v1(path: str | None) -> set[str] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Bundle wallets file not found: {path}")
    return {
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _split_trade_windows_v1(trades: list[dict]) -> tuple[int, list[dict], list[dict]]:
    trades_with_ts = []
    for trade in trades:
        normalized = normalize_trade(trade)
        ts = normalized.get("ts")
        if ts is None:
            continue
        trades_with_ts.append((ts, trade))

    if not trades_with_ts:
        return 0, [], []

    trades_with_ts.sort(key=lambda item: item[0])
    t0 = trades_with_ts[0][0]
    end_0_3 = t0 + 180
    end_3_5 = t0 + 300

    trades_0_3 = [trade for ts, trade in trades_with_ts if t0 <= ts < end_0_3]
    trades_3_5 = [trade for ts, trade in trades_with_ts if end_0_3 <= ts < end_3_5]

    return t0, trades_0_3, trades_3_5


def _build_rows_v1(
    tokens: list[dict],
    bundle_wallets: set[str] | None,
) -> list[dict]:
    rows: list[dict] = []
    for token_row in tokens:
        token_mint = token_row.get("token")
        if not token_mint:
            continue
        supply = _coerce_decimal(token_row.get("supply"))
        if supply is None or supply <= 0:
            continue
        trades = _parse_trades(token_row.get("trades"))
        if not trades:
            continue

        _, trades_0_3, trades_3_5 = _split_trade_windows_v1(trades)
        if not trades_0_3 or not trades_3_5:
            continue

        target = build_target(trades_3_5, token_mint, supply)
        if target is None:
            continue

        features = build_features(trades_0_3, token_mint, supply, bundle_wallets)

        row = {
            "mint": token_mint,
            "target_usd_mcap_3_5m": float(target),
            **features,
        }
        rows.append(row)
    return rows


async def build_dataset_v1(
    dsn: str,
    out_path: str,
    limit: int | None = None,
    batch_size: int = 500,
    bundle_wallets_path: str | None = None,
) -> Path:
    await init_pg(dsn)
    pool = get_pool()

    bundle_wallets = _load_bundle_wallets_v1(bundle_wallets_path)
    rows: list[dict] = []

    last_id = 0
    fetched = 0

    while True:
        if limit is not None and fetched >= limit:
            break
        current_batch_size = batch_size
        if limit is not None:
            current_batch_size = min(batch_size, limit - fetched)
        batch = await _fetch_tokens_batch_v1(pool, last_id, current_batch_size)
        if not batch:
            break
        last_id = batch[-1]["id"]
        batch_rows = _build_rows_v1(batch, bundle_wallets)
        rows.extend(batch_rows)
        fetched += len(batch)

    df = pd.DataFrame(rows)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    return out_file


def parse_args_v1() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from tokens table.")
    parser.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="Postgres DSN")
    parser.add_argument(
        "--out",
        default="data/datasets/ds_v1.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit tokens processed")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument(
        "--bundle-wallets",
        dest="bundle_wallets",
        default=None,
        help="Path to bundle wallets file",
    )
    return parser.parse_args()


def main_v1() -> None:
    args = parse_args_v1()
    if not args.dsn:
        raise SystemExit("--dsn or DATABASE_URL must be provided")
    asyncio.run(
        build_dataset_v1(
            dsn=args.dsn,
            out_path=args.out,
            limit=args.limit,
            batch_size=args.batch_size,
            bundle_wallets_path=args.bundle_wallets,
        )
    )


def _resolve_t0(row: pd.Series, trades: list[dict]) -> int:
    created_ts = (
        _coerce_ts(row.get(CREATED_TS_COLUMN)) if CREATED_TS_COLUMN in row.index else None
    )
    if created_ts is not None:
        return created_ts
    first_trade_ts = (
        _coerce_ts(row.get(FIRST_TRADE_TS_COLUMN)) if FIRST_TRADE_TS_COLUMN in row.index else None
    )
    if first_trade_ts is not None:
        return first_trade_ts
    ts_values = [trade.get("ts") for trade in trades if trade.get("ts") is not None]
    return int(min(ts_values)) if ts_values else 0


def _coerce_ts(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    if hasattr(value, "timestamp"):
        return int(value.timestamp())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _coerce_decimal(value: Any) -> Decimal | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _extract_token_amount(trade: dict, token_mint: str) -> Decimal | None:
    if trade.get("token1") == token_mint:
        amount_raw = trade.get("amount1")
        decimals = trade.get("token1_decimals")
    elif trade.get("token2") == token_mint:
        amount_raw = trade.get("amount2")
        decimals = trade.get("token2_decimals")
    else:
        return None

    if amount_raw in (None, 0):
        return None
    if decimals is None:
        return None
    try:
        amount = Decimal(str(amount_raw))
        denom = Decimal(10) ** int(decimals)
        token_amount = amount / denom
    except (InvalidOperation, TypeError, ValueError):
        return None
    return token_amount if token_amount > 0 else None


def _compute_max_usd_mcap(
    trades: list[dict],
    token_mint: str,
    t0: int,
    supply: Decimal | None,
) -> Decimal | None:
    if supply is None or supply <= 0:
        return None
    window_start = t0 + 180
    window_end = t0 + 300
    max_mcap: Decimal | None = None

    for trade in trades:
        ts = trade.get("ts")
        if ts is None:
            continue
        try:
            ts_value = int(ts)
        except (TypeError, ValueError):
            continue
        if ts_value < window_start or ts_value > window_end:
            continue

        token_amount = _extract_token_amount(trade, token_mint)
        if token_amount is None:
            continue

        usd_value = _coerce_decimal(trade.get("value"))
        if usd_value is None or usd_value <= 0:
            continue

        usd_price = usd_value / token_amount
        usd_mcap = usd_price * supply
        if max_mcap is None or usd_mcap > max_mcap:
            max_mcap = usd_mcap

    return max_mcap
