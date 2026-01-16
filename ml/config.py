from __future__ import annotations

from dataclasses import dataclass

WINDOW_SEC = 300
WINDOW_MINUTES = WINDOW_SEC / 60
VOLUME_ACCEL_SPLIT_SEC = 120

TRADES_COLUMN = "trades"
TOKEN_COLUMN = "token"
MAX_MARKET_CAP_COLUMN = "max_market_cap"
CREATED_TS_COLUMN = "created_at"
FIRST_TRADE_TS_COLUMN = "first_trade_ts"
TARGET_COLUMN = "target_log_mcap"

TOKENS_TABLE = "tokens"
ML_PRED_COLUMN = "ml_pred"

VOLUME_THRESHOLDS_USD = (100.0, 1000.0)
TRADES_COUNT_THRESHOLD = 10


@dataclass(frozen=True)
class ModelConfig:
    window_sec: int = WINDOW_SEC
    target_column: str = TARGET_COLUMN
    max_market_cap_column: str = MAX_MARKET_CAP_COLUMN


MODEL_CONFIG = ModelConfig()
