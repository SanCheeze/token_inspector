from __future__ import annotations

from decimal import Decimal

from ml.marketcap import calc_usd_mcap
from ml.schema import FEATURE_NAMES
from ml.trade_adapter import normalize_trade


def build_features(
    trades_0_3m: list[dict],
    token_mint: str,
    supply: Decimal,
    bundle_wallets: set[str] | None,
) -> dict:
    bundle_wallets = bundle_wallets or set()

    normalized_trades = [normalize_trade(trade) for trade in trades_0_3m]
    trades = [trade for trade in normalized_trades if trade.get("ts") is not None]

    trades_total = len(trades)
    buy_trades = [trade for trade in trades if trade.get("token2") == token_mint]
    sell_trades = [trade for trade in trades if trade.get("token1") == token_mint]

    volume_total = sum((trade.get("value") or Decimal(0)) for trade in trades)
    volume_buy = sum((trade.get("value") or Decimal(0)) for trade in buy_trades)
    volume_sell = sum((trade.get("value") or Decimal(0)) for trade in sell_trades)

    trade_values = [trade.get("value") or Decimal(0) for trade in trades]
    max_trade = max(trade_values) if trade_values else Decimal(0)
    avg_trade = (volume_total / trades_total) if trades_total else Decimal(0)

    wallets = {str(trade.get("from")) for trade in trades if trade.get("from")}
    unique_wallets = len(wallets)

    t0 = min(trade.get("ts") for trade in trades) if trades else 0
    volume_min0 = _volume_in_window(trades, t0, t0 + 60)
    volume_min1 = _volume_in_window(trades, t0 + 60, t0 + 120)
    volume_min2 = _volume_in_window(trades, t0 + 120, t0 + 180)

    volume_accel_1_0 = (volume_min1 + Decimal(1)) / (volume_min0 + Decimal(1))
    volume_accel_2_1 = (volume_min2 + Decimal(1)) / (volume_min1 + Decimal(1))

    max_usd_mcap = _max_usd_mcap(trades, token_mint, supply)

    bundle_trades = [trade for trade in trades if str(trade.get("from")) in bundle_wallets]
    bundle_wallets_unique = {str(trade.get("from")) for trade in bundle_trades if trade.get("from")}
    bundle_volume = sum((trade.get("value") or Decimal(0)) for trade in bundle_trades)
    bundle_volume_share = (bundle_volume / volume_total) if volume_total else Decimal(0)

    bundle_buy = sum(
        (trade.get("value") or Decimal(0))
        for trade in bundle_trades
        if trade.get("token2") == token_mint
    )
    bundle_sell = sum(
        (trade.get("value") or Decimal(0))
        for trade in bundle_trades
        if trade.get("token1") == token_mint
    )
    bundle_net_flow = bundle_buy - bundle_sell

    computed = {
        "f_trades_total": float(trades_total),
        "f_volume_total_usd": float(volume_total),
        "f_avg_trade_usd": float(avg_trade),
        "f_max_trade_usd": float(max_trade),
        "f_volume_min0_usd": float(volume_min0),
        "f_volume_min1_usd": float(volume_min1),
        "f_volume_min2_usd": float(volume_min2),
        "f_unique_wallets": float(unique_wallets),
        "f_trades_buy": float(len(buy_trades)),
        "f_trades_sell": float(len(sell_trades)),
        "f_volume_buy_usd": float(volume_buy),
        "f_volume_sell_usd": float(volume_sell),
        "f_buy_sell_ratio": float((len(buy_trades) + 1) / (len(sell_trades) + 1)),
        "f_net_flow_usd": float(volume_buy - volume_sell),
        "f_volume_accel": float(volume_accel_2_1),
        "f_volume_accel_1_0": float(volume_accel_1_0),
        "f_volume_accel_2_1": float(volume_accel_2_1),
        "f_max_usd_mcap_0_3m": float(max_usd_mcap) if max_usd_mcap is not None else 0.0,
        "f_bundle_wallets_participated": float(len(bundle_wallets_unique)),
        "f_bundle_volume_usd": float(bundle_volume),
        "f_bundle_volume_share": float(bundle_volume_share),
        "f_bundle_net_flow_usd": float(bundle_net_flow),
    }
    return {name: computed.get(name, 0.0) for name in FEATURE_NAMES}


def _volume_in_window(trades: list[dict], start_ts: int, end_ts: int) -> Decimal:
    return sum(
        (trade.get("value") or Decimal(0))
        for trade in trades
        if trade.get("ts") is not None and start_ts <= trade.get("ts") < end_ts
    )


def _max_usd_mcap(trades: list[dict], token_mint: str, supply: Decimal) -> Decimal | None:
    max_mcap: Decimal | None = None
    for trade in trades:
        usd_mcap = calc_usd_mcap(trade, token_mint, supply)
        if usd_mcap is None:
            continue
        if max_mcap is None or usd_mcap > max_mcap:
            max_mcap = usd_mcap
    return max_mcap
