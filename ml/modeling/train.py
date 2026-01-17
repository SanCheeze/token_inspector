from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import RAW_TARGET_NAME, RANDOM_SEED, TARGET_NAME
from ml.dataset.schema import FEATURE_COLUMNS_V1
from ml.modeling.evaluate import evaluate_predictions
from ml.modeling.pipelines import build_pipeline


def _load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def train_model(
    data_path: str,
    model_out: str,
    metrics_out: str,
    model_type: str = "hgb",
) -> dict[str, float]:
    df = _load_dataset(data_path)
    target_column = TARGET_NAME if TARGET_NAME in df.columns else RAW_TARGET_NAME
    if target_column not in df.columns:
        raise ValueError("Target column missing in dataset")

    X = df[FEATURE_COLUMNS_V1]
    y = df[target_column].to_numpy()

    if "created_at" in df.columns:
        df_sorted = df.sort_values("created_at")
        split_index = int(len(df_sorted) * 0.8)
        X_train = df_sorted[FEATURE_COLUMNS_V1].iloc[:split_index]
        X_test = df_sorted[FEATURE_COLUMNS_V1].iloc[split_index:]
        y_train = df_sorted[target_column].iloc[:split_index].to_numpy()
        y_test = df_sorted[target_column].iloc[split_index:].to_numpy()
    else:
        print("Warning: time-based split not possible, using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

    pipeline = build_pipeline(feature_columns=FEATURE_COLUMNS_V1, model_type=model_type)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(np.asarray(y_test), np.asarray(y_pred))

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(pipeline, model_out)

    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {**metrics, "target": target_column, "model_type": model_type}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    feature_list_path = metrics_path.with_name("feature_list.json")
    feature_list_path.write_text(
        json.dumps(list(FEATURE_COLUMNS_V1), indent=2), encoding="utf-8"
    )

    model_step = pipeline.named_steps.get("model")
    if hasattr(model_step, "feature_importances_"):
        importances = list(model_step.feature_importances_)
        importance_path = metrics_path.with_name("feature_importances.json")
        importance_path.write_text(
            json.dumps(
                {
                    feature: float(score)
                    for feature, score in zip(FEATURE_COLUMNS_V1, importances)
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return metrics
