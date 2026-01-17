from __future__ import annotations

from typing import Any


def _lower_keys(trade: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in trade.items()}


def _first(trade: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in trade:
            return trade[key]
    return None


def normalize_trade(trade: dict[str, Any]) -> dict[str, Any]:
    lower = _lower_keys(trade)

    ts = _first(
        lower,
        ["ts", "timestamp", "blocktime", "block_time", "time", "created_at"],
    )
    if isinstance(ts, str) and ts.isdigit():
        ts = int(ts)
    if isinstance(ts, float):
        ts = int(ts)
    if isinstance(ts, int) and ts > 1_000_000_000_000:
        ts = ts // 1000

    wallet = _first(lower, ["from", "owner", "wallet", "user", "trader"])  # type: ignore[assignment]

    value = _first(
        lower,
        [
            "value",
            "usd",
            "usd_value",
            "usdvalue",
            "amount_usd",
            "valueusd",
        ],
    )

    token1 = _first(lower, ["token1", "token_in", "mint_in", "tokena", "token_a"])
    token2 = _first(lower, ["token2", "token_out", "mint_out", "tokenb", "token_b"])

    amount1 = _first(
        lower,
        [
            "amount1",
            "amount_in",
            "amounta",
            "token1amount",
            "tokenaamount",
        ],
    )
    amount2 = _first(
        lower,
        [
            "amount2",
            "amount_out",
            "amountb",
            "token2amount",
            "tokenbamount",
        ],
    )

    token1_decimals = _first(
        lower,
        [
            "token1_decimals",
            "tokendecimals1",
            "token1decimals",
            "decimals1",
            "tokendecimalsa",
        ],
    )
    token2_decimals = _first(
        lower,
        [
            "token2_decimals",
            "tokendecimals2",
            "token2decimals",
            "decimals2",
            "tokendecimalsb",
        ],
    )

    side = _first(lower, ["side", "direction", "type"])
    is_buy = _first(lower, ["is_buy", "isbuy"])
    if isinstance(is_buy, bool):
        side = "buy" if is_buy else "sell"
    if isinstance(side, str):
        side = side.lower()

    return {
        "ts": int(ts) if ts is not None else None,
        "from": str(wallet) if wallet not in (None, "") else None,
        "value": float(value) if value not in (None, "") else None,
        "token1": str(token1) if token1 not in (None, "") else None,
        "token2": str(token2) if token2 not in (None, "") else None,
        "amount1": amount1,
        "amount2": amount2,
        "token1_decimals": int(token1_decimals) if token1_decimals not in (None, "") else None,
        "token2_decimals": int(token2_decimals) if token2_decimals not in (None, "") else None,
        "side": side,
    }
