import json

def add_price_to_trades(trades):
    """
    trades: list[dict] | str | None
    возвращает list[dict] с добавленным price
    """
    if not trades:
        return []

    if isinstance(trades, str):
        trades = json.loads(trades)

    out = []

    for t in trades:
        usd = t.get("usd_value")
        base = t.get("base_amount")

        price = None
        if usd is not None and base not in (None, 0):
            price = usd / base

        nt = dict(t)
        nt["price"] = price
        out.append(nt)

    return out
