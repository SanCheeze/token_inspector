from __future__ import annotations

from ml.features.extractor import extract_features
from ml.features.gini import gini
from ml.schema import FEATURE_NAMES


def test_extract_features_window_and_sides() -> None:
    token = "TokenMint"
    t0 = 1_700_000_000
    trades = [
        {
            "ts": t0,
            "from": "wallet1",
            "value": 100.0,
            "token1": "So11111111111111111111111111111111111111112",
            "token2": token,
            "amount1": 100,
            "amount2": 50,
            "token1_decimals": 6,
            "token2_decimals": 6,
            "platforms": "A",
            "sources": "A",
        },
        {
            "ts": t0 + 120,
            "from": "wallet2",
            "value": 40.0,
            "token1": token,
            "token2": "So11111111111111111111111111111111111111112",
            "amount1": 30,
            "amount2": 10,
            "token1_decimals": 6,
            "token2_decimals": 6,
            "platforms": "B",
            "sources": "A|B",
        },
        {
            "ts": t0 + 400,
            "from": "wallet3",
            "value": 100.0,
            "token1": token,
            "token2": "So11111111111111111111111111111111111111112",
            "amount1": 10,
            "amount2": 5,
            "token1_decimals": 6,
            "token2_decimals": 6,
            "platforms": "C",
            "sources": "C",
        },
    ]

    features = extract_features(trades, token, t0)

    assert features["f_trades_total"] == 2.0
    assert features["f_trades_buy"] == 1.0
    assert features["f_trades_sell"] == 1.0
    assert features["f_has_aggregator_route"] == 1.0
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_gini_edge_cases() -> None:
    assert gini([]) == 0.0
    assert gini([0.0, 0.0]) == 0.0
    assert gini([10.0]) == 0.0
