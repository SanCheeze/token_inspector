from __future__ import annotations

from typing import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.dataset.schema import FEATURE_COLUMNS_V1


def build_pipeline(
    feature_columns: Iterable[str] | None = None,
    model_type: str = "hgb",
    scale: bool = False,
) -> Pipeline:
    feature_columns = list(feature_columns or FEATURE_COLUMNS_V1)

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(numeric_steps), feature_columns)],
        remainder="drop",
    )

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(random_state=42)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
