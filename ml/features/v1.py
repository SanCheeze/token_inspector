from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from ml.config import FEATURE_WINDOW_SEC, TARGET_START_OFFSET_SEC, TARGET_WINDOW_SEC
from ml.features.marketcap import calc_usd_mcap

SOL_MINT = "So11111111111111111111111111111111111111112"


def split_windows(trades: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    trades = [trade for trade in trades if trade.get("ts") is not None]
    if not trades:
        return 0, [], []
    trades_sorted = sorted(trades, key=lambda item: item["ts"])
    t0 = trades_sorted[0]["ts"]
    t_feature_end = t0 + FEATURE_WINDOW_SEC
    t_target_end = t0 + TARGET_START_OFFSET_SEC + TARGET_WINDOW_SEC

    trades_0_3 = [trade for trade in trades_sorted if trade["ts"] < t_feature_end]
    trades_3_5 = [
        trade
        for trade in trades_sorted
        if t_feature_end <= trade["ts"] < t_target_end
    ]
    return t0, trades_0_3, trades_3_5


def _trade_side(trade: dict[str, Any], token_mint: str) -> str | None:
    side = trade.get("side")
    if isinstance(side, str):
        side = side.lower()
    if side in {"buy", "sell"}:
        return side
    token1 = str(trade.get("token1") or "")
    token2 = str(trade.get("token2") or "")
    if SOL_MINT in token2:
        return "buy"
    if SOL_MINT in token1:
        return "sell"
    amount = trade.get("amount1") if trade.get("token1") == token_mint else trade.get("amount2")
    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return None
    if amount_value > 0:
        return "buy"
    if amount_value < 0:
        return "sell"
    return None


def build_features_v1(
    trades_0_3: list[dict[str, Any]],
    token_mint: str,
    supply: float,
    bundle_wallets: set[str] | None,
) -> dict[str, float]:
    bundle_wallets = bundle_wallets or set()
    total_trades = len(trades_0_3)
    values = [trade.get("value") for trade in trades_0_3 if trade.get("value") is not None]
    values_float = [float(v) for v in values]

    volume_total = float(sum(values_float))
    unique_wallets = {trade.get("from") for trade in trades_0_3 if trade.get("from")}

    buy_trades = []
    sell_trades = []
    for trade in trades_0_3:
        side = _trade_side(trade, token_mint)
        if side == "buy":
            buy_trades.append(trade)
        elif side == "sell":
            sell_trades.append(trade)

    volume_buy = sum(float(trade.get("value") or 0) for trade in buy_trades)
    volume_sell = sum(float(trade.get("value") or 0) for trade in sell_trades)
    buy_sell_ratio = volume_buy / volume_sell if volume_sell else float(volume_buy)
    net_flow = volume_buy - volume_sell

    avg_trade = float(statistics.fmean(values_float)) if values_float else 0.0
    median_trade = float(np.median(values_float)) if values_float else 0.0
    max_trade = float(max(values_float)) if values_float else 0.0

    volume_min0 = 0.0
    volume_min1 = 0.0
    volume_min2 = 0.0
    if trades_0_3:
        t0 = min(trade["ts"] for trade in trades_0_3 if trade.get("ts") is not None)
        for trade in trades_0_3:
            ts = trade.get("ts")
            if ts is None or trade.get("value") is None:
                continue
            minute_bucket = int((ts - t0) // 60)
            if minute_bucket == 0:
                volume_min0 += float(trade.get("value"))
            elif minute_bucket == 1:
                volume_min1 += float(trade.get("value"))
            elif minute_bucket == 2:
                volume_min2 += float(trade.get("value"))

    volume_accel_1_0 = volume_min1 - volume_min0
    volume_accel_2_1 = volume_min2 - volume_min1

    mcap_values = [
        value
        for trade in trades_0_3
        if (value := calc_usd_mcap(trade, token_mint, supply)) is not None
    ]
    max_mcap = float(max(mcap_values)) if mcap_values else 0.0

    bundle_trades = [trade for trade in trades_0_3 if str(trade.get("from")) in bundle_wallets]
    bundle_wallets_unique = {trade.get("from") for trade in bundle_trades if trade.get("from")}
    bundle_volume = sum(float(trade.get("value") or 0) for trade in bundle_trades)
    bundle_volume_share = bundle_volume / volume_total if volume_total else 0.0
    bundle_buy = sum(
        float(trade.get("value") or 0)
        for trade in bundle_trades
        if _trade_side(trade, token_mint) == "buy"
    )
    bundle_sell = sum(
        float(trade.get("value") or 0)
        for trade in bundle_trades
        if _trade_side(trade, token_mint) == "sell"
    )
    bundle_net_flow = bundle_buy - bundle_sell

    return {
        "f_trades_total": float(total_trades),
        "f_volume_total_usd": float(volume_total),
        "f_unique_wallets": float(len(unique_wallets)),
        "f_trades_buy": float(len(buy_trades)),
        "f_trades_sell": float(len(sell_trades)),
        "f_volume_buy_usd": float(volume_buy),
        "f_volume_sell_usd": float(volume_sell),
        "f_buy_sell_ratio": float(buy_sell_ratio),
        "f_net_flow_usd": float(net_flow),
        "f_avg_trade_usd": float(avg_trade),
        "f_median_trade_usd": float(median_trade),
        "f_max_trade_usd": float(max_trade),
        "f_volume_min0_usd": float(volume_min0),
        "f_volume_min1_usd": float(volume_min1),
        "f_volume_min2_usd": float(volume_min2),
        "f_volume_accel_1_0": float(volume_accel_1_0),
        "f_volume_accel_2_1": float(volume_accel_2_1),
        "f_max_usd_mcap_0_3m": float(max_mcap),
        "f_bundle_wallets_participated": float(len(bundle_wallets_unique)),
        "f_bundle_volume_usd": float(bundle_volume),
        "f_bundle_volume_share": float(bundle_volume_share),
        "f_bundle_net_flow_usd": float(bundle_net_flow),
    }
