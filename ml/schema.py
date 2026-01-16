from __future__ import annotations

FEATURE_NAMES = [
    "f_trades_total",
    "f_trades_buy",
    "f_trades_sell",
    "f_buy_sell_ratio",
    "f_volume_total_usd",
    "f_volume_buy_usd",
    "f_volume_sell_usd",
    "f_net_flow_usd",
    "f_avg_trade_usd",
    "f_median_trade_usd",
    "f_max_trade_usd",
    "f_unique_wallets",
    "f_top1_wallet_vol_share",
    "f_top3_wallet_vol_share",
    "f_wallet_vol_gini",
    "f_time_to_first_trade_sec",
    "f_time_to_10_trades_sec",
    "f_time_to_100usd_sec",
    "f_time_to_1000usd_sec",
    "f_trades_per_min",
    "f_volume_per_min",
    "f_volume_accel",
    "f_platforms_count",
    "f_top_platform_share",
    "f_has_aggregator_route",
    "f_bundle_wallets_participated",
    "f_bundle_participation_ratio",
    "f_bundle_volume_share",
    "f_bundle_net_flow_usd",
    "f_bundle_first_entry_sec",
    "f_bundle_entry_std_sec",
]


def get_feature_names() -> list[str]:
    return list(FEATURE_NAMES)
