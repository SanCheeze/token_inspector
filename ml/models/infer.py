from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from ml.features.extractor import extract_features
from ml.schema import FEATURE_NAMES


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    model_type: str
    feature_names: list[str]
    config: dict[str, Any]


def load_model(path: str | Path) -> ModelBundle:
    path = Path(path)
    if path.is_dir():
        model_path = path / "model.bin"
        features_path = path / "feature_names.json"
        config_path = path / "config.json"
    else:
        model_path = path
        features_path = path.with_name("feature_names.json")
        config_path = path.with_name("config.json")

    with features_path.open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle).get("features", FEATURE_NAMES)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    model, model_type = _load_backend(model_path)
    return ModelBundle(model=model, model_type=model_type, feature_names=feature_names, config=config)


def predict_for_token(
    trades: list[dict],
    token_mint: str,
    model_path: str | Path,
    bundle_wallets: set[str] | None = None,
    t0: int | None = None,
    include_features: bool = False,
) -> dict[str, Any]:
    if t0 is None:
        t0 = min((trade.get("ts") for trade in trades if trade.get("ts") is not None), default=0)
    features = extract_features(trades, token_mint, t0, bundle_wallets=bundle_wallets)

    model_bundle = load_model(model_path)
    X = np.array([features[name] for name in model_bundle.feature_names], dtype=float).reshape(1, -1)
    pred_log = _predict_backend(model_bundle, X)
    pred_mcap = float(np.expm1(pred_log))
    result = {
        "pred_log_mcap": float(pred_log),
        "pred_mcap": pred_mcap,
    }
    if include_features:
        result["features_used"] = features
    return result


def _load_backend(path: Path) -> tuple[Any, str]:
    if find_spec("lightgbm") is not None:
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        return booster, "lightgbm"
    if find_spec("xgboost") is not None:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(str(path))
        return model, "xgboost"
    raise RuntimeError("Neither lightgbm nor xgboost is installed")


def _predict_backend(bundle: ModelBundle, X: np.ndarray) -> float:
    if bundle.model_type == "lightgbm":
        pred = bundle.model.predict(X)
    else:
        pred = bundle.model.predict(X)
    return float(pred[0])
