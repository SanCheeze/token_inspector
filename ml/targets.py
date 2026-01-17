from __future__ import annotations

from decimal import Decimal

from ml.marketcap import calc_usd_mcap


def build_target(
    trades_3_5m: list[dict],
    token_mint: str,
    supply: Decimal,
) -> Decimal | None:
    max_mcap: Decimal | None = None
    for trade in trades_3_5m:
        usd_mcap = calc_usd_mcap(trade, token_mint, supply)
        if usd_mcap is None:
            continue
        if max_mcap is None or usd_mcap > max_mcap:
            max_mcap = usd_mcap
    return max_mcap
