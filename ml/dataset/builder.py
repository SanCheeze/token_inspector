from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml.config import (
    RAW_TARGET_NAME,
    TARGET_NAME,
    TOKEN_COLUMN,
    TRADES_COLUMN,
)
from ml.db import load_bundle_wallets, load_tokens_df, parse_trades_payload
from ml.features.marketcap import calc_usd_mcap
from ml.features.trade_normalize import normalize_trade
from ml.features.v1 import build_features_v1, split_windows


def build_target(
    trades_3_5: list[dict[str, Any]], token_mint: str, supply: float
) -> float | None:
    mcap_values = [
        value
        for trade in trades_3_5
        if (value := calc_usd_mcap(trade, token_mint, supply)) is not None
    ]
    if not mcap_values:
        return None
    return float(max(mcap_values))


def _normalize_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        normalized.append(normalize_trade(trade))
    return normalized


def _parse_supply(raw_supply: Any) -> float | None:
    try:
        supply_value = float(raw_supply)
    except (TypeError, ValueError):
        return None
    if supply_value <= 0:
        return None
    return supply_value


def _load_dsn(dsn: str | None) -> str:
    if dsn:
        return dsn
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
    dsn_env = os.getenv("DB_URL")
    if not dsn_env:
        raise RuntimeError("DB_URL environment variable is not set")
    return dsn_env


async def build_dataset_from_db(
    dsn: str | None,
    limit: int | None = None,
    bundle_name: str | None = None,
) -> pd.DataFrame:
    pool = await asyncpg.create_pool(dsn=_load_dsn(dsn))
    try:
        df_tokens = await load_tokens_df(pool, limit=limit)
        bundle_wallets = await load_bundle_wallets(pool, bundle_name)
    finally:
        await pool.close()

    rows: list[dict[str, Any]] = []
    for _, row in df_tokens.iterrows():
        token_mint = row.get(TOKEN_COLUMN)
        if not token_mint:
            continue
        supply = _parse_supply(row.get("supply"))
        if supply is None:
            continue
        raw_trades = row.get(TRADES_COLUMN)
        trades = parse_trades_payload(raw_trades)
        if not trades:
            continue
        normalized_trades = _normalize_trades(trades)
        _, trades_0_3, trades_3_5 = split_windows(normalized_trades)
        if not trades_3_5:
            continue
        target_raw = build_target(trades_3_5, token_mint, supply)
        if target_raw is None:
            continue
        features = build_features_v1(trades_0_3, token_mint, supply, bundle_wallets)
        row_data = {
            "mint": token_mint,
            **features,
            RAW_TARGET_NAME: float(target_raw),
            TARGET_NAME: float(np.log1p(target_raw)),
        }
        rows.append(row_data)

    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, path: str, fmt: str | None = None) -> None:
    if fmt is None:
        if path.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "parquet"
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")
