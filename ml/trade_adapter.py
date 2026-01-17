from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def _first_value(trade: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in trade and trade[key] is not None:
            return trade[key]
    return None


def _coerce_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_trade(trade: dict) -> dict:
    return {
        "ts": _coerce_int(
            _first_value(trade, ["ts", "timestamp", "blockTime", "block_time", "time"])
        ),
        "from": _first_value(
            trade,
            ["from", "from_address", "wallet", "owner", "maker", "taker"],
        ),
        "value": _coerce_decimal(
            _first_value(trade, ["value", "usd_value", "usdValue", "usd_value_usd"])
        ),
        "token1": _first_value(trade, ["token1", "Token1", "token_1", "tokenA", "tokenIn"]),
        "amount1": _coerce_decimal(
            _first_value(trade, ["amount1", "Amount1", "tokenAmount1", "amount_in"])
        ),
        "token1_decimals": _first_value(
            trade,
            [
                "token1_decimals",
                "tokenDecimals1",
                "TokenDecimals1",
                "token1Decimals",
            ],
        ),
        "token2": _first_value(trade, ["token2", "Token2", "token_2", "tokenB", "tokenOut"]),
        "amount2": _coerce_decimal(
            _first_value(trade, ["amount2", "Amount2", "tokenAmount2", "amount_out"])
        ),
        "token2_decimals": _first_value(
            trade,
            [
                "token2_decimals",
                "tokenDecimals2",
                "TokenDecimals2",
                "token2Decimals",
            ],
        ),
        "platforms": _first_value(trade, ["platforms", "Platforms", "platform", "Platform"]),
        "sources": _first_value(trade, ["sources", "Sources", "source", "Source"]),
    }
