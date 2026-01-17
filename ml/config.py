from __future__ import annotations

TOKEN_COLUMN = "token"
TRADES_COLUMN = "trades"
SUPPLY_COLUMN = "supply"

FEATURE_WINDOW_SEC = 180
TARGET_WINDOW_SEC = 120
TARGET_START_OFFSET_SEC = 180

TARGET_NAME = "target_log1p_usd_mcap_3_5m"
RAW_TARGET_NAME = "target_usd_mcap_3_5m"

RANDOM_SEED = 42

DEFAULT_DATASET_PATH = "data/datasets/ds_v1.parquet"
DEFAULT_MODEL_PATH = "ml/artifacts/model.joblib"
