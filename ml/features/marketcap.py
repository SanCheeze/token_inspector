from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_amount(amount: Any, decimals: int | None) -> float | None:
    if amount is None:
        return None
    if isinstance(amount, (int, str)):
        value = _to_float(amount)
        if value is None:
            return None
        if decimals is not None:
            return value / (10**decimals)
        return value
    value = _to_float(amount)
    return value


def calc_token_amount(trade: dict[str, Any], token_mint: str) -> float | None:
    token1 = trade.get("token1")
    token2 = trade.get("token2")
    if token1 == token_mint:
        return _normalize_amount(trade.get("amount1"), trade.get("token1_decimals"))
    if token2 == token_mint:
        return _normalize_amount(trade.get("amount2"), trade.get("token2_decimals"))
    return None


def calc_usd_price(trade: dict[str, Any], token_mint: str) -> float | None:
    value = _to_float(trade.get("value"))
    if value is None:
        return None
    amount = calc_token_amount(trade, token_mint)
    if amount in (None, 0):
        return None
    return abs(value) / abs(amount)


def calc_usd_mcap(trade: dict[str, Any], token_mint: str, supply: float) -> float | None:
    price = calc_usd_price(trade, token_mint)
    if price is None or supply <= 0:
        return None
    return price * supply
