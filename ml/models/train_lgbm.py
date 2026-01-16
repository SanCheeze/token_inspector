from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from importlib.util import find_spec

from ml.config import MODEL_CONFIG, TARGET_COLUMN
from ml.dataset.splits import time_based_split
from ml.models.artifacts import create_artifact_paths, save_json
from ml.schema import FEATURE_NAMES

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _spearman(preds: np.ndarray, actual: np.ndarray) -> float:
    if preds.size == 0:
        return 0.0
    preds_rank = pd.Series(preds).rank().to_numpy()
    actual_rank = pd.Series(actual).rank().to_numpy()
    return float(np.corrcoef(preds_rank, actual_rank)[0, 1])


def _topk_hitrate(preds: np.ndarray, actual: np.ndarray, top_ratio: float = 0.1) -> float:
    if preds.size == 0:
        return 0.0
    k = max(1, int(np.ceil(preds.size * top_ratio)))
    pred_top_idx = np.argsort(preds)[-k:]
    actual_top_idx = np.argsort(actual)[-k:]
    hitrate = len(set(pred_top_idx).intersection(set(actual_top_idx))) / k
    return float(hitrate)


def train_model(
    dataset_path: str,
    meta_path: str | None,
    artifacts_root: str,
) -> Path:
    df = load_dataset(Path(dataset_path))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset missing target column: {TARGET_COLUMN}")

    meta_df = load_dataset(Path(meta_path)) if meta_path else None
    if meta_df is None:
        meta_df = pd.DataFrame({"t0": np.arange(len(df))})

    time_column = "t0" if "t0" in meta_df.columns else meta_df.columns[0]
    meta_df = meta_df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    if len(meta_df) != len(df):
        min_len = min(len(meta_df), len(df))
        meta_df = meta_df.iloc[:min_len].reset_index(drop=True)
        df = df.iloc[:min_len].reset_index(drop=True)

    train_meta, valid_meta, test_meta = time_based_split(meta_df, time_column)
    train_idx = train_meta.index
    valid_idx = valid_meta.index
    test_idx = test_meta.index

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_valid = X.loc[valid_idx]
    y_valid = y.loc[valid_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    model, model_type = _train_model_backend(X_train, y_train, X_valid, y_valid)

    valid_preds = _predict(model, X_valid, model_type)
    test_preds = _predict(model, X_test, model_type)

    valid_rmse = float(np.sqrt(np.mean((valid_preds - y_valid.to_numpy()) ** 2)))
    test_rmse = float(np.sqrt(np.mean((test_preds - y_test.to_numpy()) ** 2)))

    metrics = {
        "rmse_valid": valid_rmse,
        "rmse_test": test_rmse,
        "model_type": model_type,
    }

    if "max_market_cap" in meta_df.columns:
        actual_valid = meta_df.loc[valid_idx, "max_market_cap"].to_numpy(dtype=float)
        actual_test = meta_df.loc[test_idx, "max_market_cap"].to_numpy(dtype=float)
        metrics.update(
            {
                "spearman_valid": _spearman(valid_preds, actual_valid),
                "spearman_test": _spearman(test_preds, actual_test),
                "top10_hitrate_valid": _topk_hitrate(valid_preds, actual_valid),
                "top10_hitrate_test": _topk_hitrate(test_preds, actual_test),
            }
        )

    artifacts_dir = Path(artifacts_root) / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts = create_artifact_paths(artifacts_dir)

    _save_model(model, model_type, artifacts.model_path)

    save_json(artifacts.feature_names_path, {"features": FEATURE_NAMES})
    save_json(
        artifacts.config_path,
        {
            "window_sec": MODEL_CONFIG.window_sec,
            "target_transform": "log1p",
            "target_column": TARGET_COLUMN,
        },
    )
    save_json(artifacts.metrics_path, metrics)

    importance = _feature_importance(model, model_type, FEATURE_NAMES)
    importance.to_csv(artifacts.feature_importance_path, index=False)

    logger.info("Saved artifacts to %s", artifacts_dir)
    return artifacts_dir


def _train_model_backend(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
):
    if find_spec("lightgbm") is not None:
        import lightgbm as lgb

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
        }
        booster = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
        )
        return booster, "lightgbm"
    if find_spec("xgboost") is not None:
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=50,
        )
        return model, "xgboost"

    raise RuntimeError("Neither lightgbm nor xgboost is installed")


def _predict(model: Any, X: pd.DataFrame, model_type: str) -> np.ndarray:
    if model_type == "lightgbm":
        return model.predict(X, num_iteration=model.best_iteration)
    return model.predict(X)


def _save_model(model: Any, model_type: str, path: Path) -> None:
    if model_type == "lightgbm":
        model.save_model(str(path))
    else:
        model.save_model(str(path))


def _feature_importance(model: Any, model_type: str, features: list[str]) -> pd.DataFrame:
    if model_type == "lightgbm":
        values = model.feature_importance(importance_type="gain")
    else:
        values = model.feature_importances_
    return pd.DataFrame({"feature": features, "importance": values}).sort_values(
        "importance", ascending=False
    )
