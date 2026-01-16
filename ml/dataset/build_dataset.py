from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import asyncpg

from ml.config import (
    CREATED_TS_COLUMN,
    FIRST_TRADE_TS_COLUMN,
    MAX_MARKET_CAP_COLUMN,
    TARGET_COLUMN,
    TOKEN_COLUMN,
    TRADES_COLUMN,
)
from ml.features.extractor import extract_features
from ml.schema import FEATURE_NAMES
from ml.dataset.db import load_bundle_wallets, load_tokens

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

        max_market_cap = row.get(MAX_MARKET_CAP_COLUMN)
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


def _resolve_t0(row: pd.Series, trades: list[dict]) -> int:
    created_ts = _coerce_ts(row.get(CREATED_TS_COLUMN)) if CREATED_TS_COLUMN in row else None
    if created_ts is not None:
        return created_ts
    first_trade_ts = _coerce_ts(row.get(FIRST_TRADE_TS_COLUMN)) if FIRST_TRADE_TS_COLUMN in row else None
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
