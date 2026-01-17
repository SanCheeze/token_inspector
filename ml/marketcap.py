from __future__ import annotations

from decimal import Decimal, InvalidOperation

from ml.trade_adapter import normalize_trade


def _coerce_decimal(value: Decimal | int | float | str | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def calc_token_amount(trade: dict, token_mint: str) -> Decimal | None:
    normalized = normalize_trade(trade)
    if normalized.get("token1") == token_mint:
        amount_raw = normalized.get("amount1")
        decimals = normalized.get("token1_decimals")
    elif normalized.get("token2") == token_mint:
        amount_raw = normalized.get("amount2")
        decimals = normalized.get("token2_decimals")
    else:
        return None

    if amount_raw in (None, 0):
        return None
    if decimals is None:
        return None

    try:
        amount = _coerce_decimal(amount_raw)
        if amount is None:
            return None
        denom = Decimal(10) ** int(decimals)
        token_amount = amount / denom
    except (InvalidOperation, TypeError, ValueError):
        return None
    return token_amount if token_amount > 0 else None


def calc_usd_price(trade: dict, token_mint: str) -> Decimal | None:
    token_amount = calc_token_amount(trade, token_mint)
    if token_amount is None or token_amount <= 0:
        return None
    normalized = normalize_trade(trade)
    usd_value = normalized.get("value")
    usd_value = _coerce_decimal(usd_value)
    if usd_value is None or usd_value <= 0:
        return None
    return usd_value / token_amount


def calc_usd_mcap(trade: dict, token_mint: str, supply: Decimal) -> Decimal | None:
    usd_price = calc_usd_price(trade, token_mint)
    if usd_price is None:
        return None
    if supply is None or supply <= 0:
        return None
    return usd_price * supply
