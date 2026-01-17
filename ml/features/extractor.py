from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ml.config import (
    VOLUME_ACCEL_SPLIT_SEC,
    VOLUME_THRESHOLDS_USD,
    WINDOW_MINUTES,
    WINDOW_SEC,
)
from ml.features.gini import gini
from ml.schema import FEATURE_NAMES
from ml.trade_adapter import normalize_trade


@dataclass(frozen=True)
class NormalizedTrade:
    ts: int
    usd_value: float
    side: int
    token_amount: float
    wallet: str
    platform: str | None
    sources: str | None


def _parse_platforms(platforms: str | None) -> list[str]:
    if not platforms:
        return []
    return [p.strip() for p in platforms.split("|") if p.strip()]


def _normalize_trades(trades: Iterable[dict], token_mint: str) -> list[NormalizedTrade]:
    normalized: list[NormalizedTrade] = []
    for trade in trades:
        normalized_trade = normalize_trade(trade)
        token1 = normalized_trade.get("token1")
        token2 = normalized_trade.get("token2")
        if token1 != token_mint and token2 != token_mint:
            continue
        usd_value = normalized_trade.get("value")
        if usd_value is None or float(usd_value) <= 0:
            continue
        ts_raw = normalized_trade.get("ts")
        if ts_raw is None:
            continue
        ts = int(ts_raw)
        side = 1 if token2 == token_mint else -1
        if side > 0:
            decimals = normalized_trade.get("token2_decimals") or 0
            amount_raw = normalized_trade.get("amount2") or 0
        else:
            decimals = normalized_trade.get("token1_decimals") or 0
            amount_raw = normalized_trade.get("amount1") or 0
        token_amount = float(amount_raw) / (10 ** int(decimals)) if decimals is not None else 0.0
        platform = normalized_trade.get("platforms")
        sources = normalized_trade.get("sources")
        normalized.append(
            NormalizedTrade(
                ts=ts,
                usd_value=float(usd_value),
                side=side,
                token_amount=token_amount,
                wallet=str(normalized_trade.get("from") or ""),
                platform=platform,
                sources=sources,
            )
        )
    return normalized


def extract_features(
    trades: list[dict],
    token_mint: str,
    t0: int,
    bundle_wallets: set[str] | None = None,
) -> dict[str, float]:
    """Extract model features for a token within the first 5 minutes."""

    window_end = t0 + WINDOW_SEC
    normalized = _normalize_trades(trades, token_mint)
    filtered = [trade for trade in normalized if t0 <= trade.ts <= window_end]

    trades_total = len(filtered)
    buy_trades = [trade for trade in filtered if trade.side > 0]
    sell_trades = [trade for trade in filtered if trade.side < 0]
    trades_buy = len(buy_trades)
    trades_sell = len(sell_trades)

    volume_total = sum(trade.usd_value for trade in filtered)
    volume_buy = sum(trade.usd_value for trade in buy_trades)
    volume_sell = sum(trade.usd_value for trade in sell_trades)

    trade_values = np.array([trade.usd_value for trade in filtered], dtype=float)
    avg_trade = float(trade_values.mean()) if trades_total else 0.0
    median_trade = float(np.median(trade_values)) if trades_total else 0.0
    max_trade = float(trade_values.max()) if trades_total else 0.0

    wallet_volumes: dict[str, float] = {}
    for trade in filtered:
        wallet_volumes[trade.wallet] = wallet_volumes.get(trade.wallet, 0.0) + trade.usd_value

    unique_wallets = len(wallet_volumes)
    sorted_wallet_volumes = sorted(wallet_volumes.values(), reverse=True)
    top1_share = (sorted_wallet_volumes[0] / volume_total) if sorted_wallet_volumes and volume_total else 0.0
    top3_share = (
        sum(sorted_wallet_volumes[:3]) / volume_total if sorted_wallet_volumes and volume_total else 0.0
    )
    gini_value = gini(list(wallet_volumes.values()))

    trade_times = sorted(trade.ts for trade in filtered)
    first_trade_ts = trade_times[0] if trade_times else t0
    time_to_first_trade = max(0.0, float(first_trade_ts - t0))
    time_to_10_trades = WINDOW_SEC
    if len(trade_times) >= 10:
        time_to_10_trades = float(trade_times[9] - t0)

    time_to_100usd = WINDOW_SEC
    time_to_1000usd = WINDOW_SEC
    thresholds_hit = [False, False]
    cumulative = 0.0
    for trade in sorted(filtered, key=lambda x: x.ts):
        cumulative += trade.usd_value
        for idx, threshold in enumerate(VOLUME_THRESHOLDS_USD):
            if not thresholds_hit[idx] and cumulative >= threshold:
                if idx == 0:
                    time_to_100usd = float(trade.ts - t0)
                else:
                    time_to_1000usd = float(trade.ts - t0)
                thresholds_hit[idx] = True
        if all(thresholds_hit):
            break

    trades_per_min = trades_total / WINDOW_MINUTES
    volume_per_min = volume_total / WINDOW_MINUTES
    volume_first = sum(
        trade.usd_value for trade in filtered if t0 <= trade.ts < t0 + VOLUME_ACCEL_SPLIT_SEC
    )
    volume_second = sum(
        trade.usd_value for trade in filtered if t0 + VOLUME_ACCEL_SPLIT_SEC <= trade.ts <= window_end
    )
    volume_accel = (volume_second + 1.0) / (volume_first + 1.0)

    platforms: dict[str, float] = {}
    platforms_seen: set[str] = set()
    has_aggregator = 0.0
    for trade in filtered:
        platform_entries = _parse_platforms(trade.platform)
        if platform_entries:
            platforms_seen.update(platform_entries)
            primary = platform_entries[0]
            platforms[primary] = platforms.get(primary, 0.0) + trade.usd_value
        if trade.sources and "|" in trade.sources:
            has_aggregator = 1.0

    platforms_count = float(len(platforms_seen))
    top_platform_share = 0.0
    if platforms and volume_total:
        top_platform_share = max(platforms.values()) / volume_total

    bundle_wallets = bundle_wallets or set()
    bundle_trades = [trade for trade in filtered if trade.wallet in bundle_wallets]
    bundle_wallets_unique = {trade.wallet for trade in bundle_trades}
    bundle_volume = sum(trade.usd_value for trade in bundle_trades)
    bundle_net_flow = sum(trade.usd_value * trade.side for trade in bundle_trades)
    if bundle_trades:
        first_by_wallet: dict[str, int] = {}
        for trade in bundle_trades:
            current = first_by_wallet.get(trade.wallet)
            if current is None or trade.ts < current:
                first_by_wallet[trade.wallet] = trade.ts
        entry_times = [ts - t0 for ts in first_by_wallet.values()]
        bundle_first_entry = float(min(entry_times))
        bundle_entry_std = float(np.std(entry_times))
    else:
        bundle_first_entry = float(WINDOW_SEC)
        bundle_entry_std = 0.0

    bundle_participation_ratio = (
        len(bundle_wallets_unique) / unique_wallets if unique_wallets else 0.0
    )
    bundle_volume_share = bundle_volume / volume_total if volume_total else 0.0

    feature_values = {
        "f_trades_total": float(trades_total),
        "f_trades_buy": float(trades_buy),
        "f_trades_sell": float(trades_sell),
        "f_buy_sell_ratio": (trades_buy + 1.0) / (trades_sell + 1.0),
        "f_volume_total_usd": float(volume_total),
        "f_volume_buy_usd": float(volume_buy),
        "f_volume_sell_usd": float(volume_sell),
        "f_net_flow_usd": float(volume_buy - volume_sell),
        "f_avg_trade_usd": avg_trade,
        "f_median_trade_usd": median_trade,
        "f_max_trade_usd": max_trade,
        "f_unique_wallets": float(unique_wallets),
        "f_top1_wallet_vol_share": float(top1_share),
        "f_top3_wallet_vol_share": float(top3_share),
        "f_wallet_vol_gini": float(gini_value),
        "f_time_to_first_trade_sec": time_to_first_trade,
        "f_time_to_10_trades_sec": float(time_to_10_trades),
        "f_time_to_100usd_sec": float(time_to_100usd),
        "f_time_to_1000usd_sec": float(time_to_1000usd),
        "f_trades_per_min": float(trades_per_min),
        "f_volume_per_min": float(volume_per_min),
        "f_volume_accel": float(volume_accel),
        "f_platforms_count": float(platforms_count),
        "f_top_platform_share": float(top_platform_share),
        "f_has_aggregator_route": float(has_aggregator),
        "f_bundle_wallets_participated": float(len(bundle_wallets_unique)),
        "f_bundle_participation_ratio": float(bundle_participation_ratio),
        "f_bundle_volume_share": float(bundle_volume_share),
        "f_bundle_net_flow_usd": float(bundle_net_flow),
        "f_bundle_first_entry_sec": float(bundle_first_entry),
        "f_bundle_entry_std_sec": float(bundle_entry_std),
    }

    return {name: feature_values.get(name, 0.0) for name in FEATURE_NAMES}
