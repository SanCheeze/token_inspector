from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

from ml.config import TARGET_COLUMN, TOKEN_COLUMN, TRADES_COLUMN
from ml.dataset.db import load_bundle_wallets, load_tokens
from ml.features_v1 import build_features
from ml.schema import FEATURE_NAMES
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
        if not trades:
            skipped_empty += 1
            continue

        token_mint = row.get(TOKEN_COLUMN)
        if not token_mint:
            skipped_empty += 1
            continue

        supply = _coerce_decimal(row.get("supply"))
        if supply is None or supply <= 0:
            skipped_empty += 1
            continue

        t0, trades_0_3, trades_3_5 = _split_trade_windows(trades)
        if not trades_0_3 or not trades_3_5:
            skipped_window += 1
            continue

        target = build_target(trades_3_5, token_mint, supply)
        if target is None or float(target) <= 0:
            skipped_target += 1
            continue

        features = build_features(trades_0_3, token_mint, supply, bundle_wallets)
        feature_rows.append(features)
        target_rows.append(float(np.log1p(float(target))))
        meta_rows.append(
            {
                TOKEN_COLUMN: token_mint,
                "t0": t0,
                "trades_total": float(features.get("f_trades_total", 0.0)),
                "target_usd_mcap_3_5m": float(target),
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


def _split_trade_windows(trades: list[dict]) -> tuple[int, list[dict], list[dict]]:
    trades_with_ts: list[tuple[int, dict]] = []
    for trade in trades:
        normalized = normalize_trade(trade)
        ts = normalized.get("ts")
        if ts is None:
            continue
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            continue
        trades_with_ts.append((ts_int, normalized))

    if not trades_with_ts:
        return 0, [], []

    trades_with_ts.sort(key=lambda item: item[0])
    t0 = trades_with_ts[0][0]
    end_0_3 = t0 + 180
    end_3_5 = t0 + 300

    trades_0_3 = [trade for ts, trade in trades_with_ts if t0 <= ts < end_0_3]
    trades_3_5 = [trade for ts, trade in trades_with_ts if end_0_3 <= ts < end_3_5]

    return t0, trades_0_3, trades_3_5


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
