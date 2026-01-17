from __future__ import annotations

import asyncpg
import pandas as pd

from ml.config import TOKEN_COLUMN, TRADES_COLUMN
from ml.db import load_bundle_wallets, load_tokens_df, parse_trades_payload
from ml.features.trade_normalize import normalize_trade
from ml.features.v1 import build_features_v1, split_windows
from ml.dataset.schema import FEATURE_COLUMNS_V1


def load_model(model_path: str):
    import joblib

    return joblib.load(model_path)


def _normalize_trades(raw_trades):
    trades = parse_trades_payload(raw_trades)
    normalized = [normalize_trade(trade) for trade in trades if isinstance(trade, dict)]
    return normalized


def _load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def predict_from_dataset(model_path: str, data_path: str) -> pd.DataFrame:
    model = load_model(model_path)
    df = _load_dataset(data_path)
    if "mint" not in df.columns:
        raise ValueError("Dataset must include 'mint' column")
    X = df[FEATURE_COLUMNS_V1]
    preds = model.predict(X)
    return pd.DataFrame({"mint": df["mint"], "y_pred": preds})


async def predict_from_db(
    dsn: str,
    limit: int | None = None,
    bundle_name: str | None = None,
) -> pd.DataFrame:
    pool = await asyncpg.create_pool(dsn=dsn)
    try:
        df_tokens = await load_tokens_df(pool, limit=limit)
        bundle_wallets = await load_bundle_wallets(pool, bundle_name)
    finally:
        await pool.close()

    rows = []
    for _, row in df_tokens.iterrows():
        token_mint = row.get(TOKEN_COLUMN)
        supply = row.get("supply")
        if not token_mint or supply is None:
            continue
        try:
            supply_value = float(supply)
        except (TypeError, ValueError):
            continue
        trades = _normalize_trades(row.get(TRADES_COLUMN))
        if not trades:
            continue
        _, trades_0_3, _ = split_windows(trades)
        features = build_features_v1(trades_0_3, token_mint, supply_value, bundle_wallets)
        rows.append({"mint": token_mint, **features})

    return pd.DataFrame(rows)


def save_predictions(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
