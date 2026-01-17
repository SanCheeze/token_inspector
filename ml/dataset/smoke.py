from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import pandas as pd

from ml.config import TOKEN_COLUMN
from ml.dataset.build_dataset import build_dataset_from_db
from ml.dataset.db_pool import get_pool, init_pg

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_db_url_from_env() -> str:
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not db_url:
        raise RuntimeError("DB_URL/DATABASE_URL/POSTGRES_DSN environment variable is not set")
    return db_url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke dataset build from DB")
    parser.add_argument("--limit", type=int, default=50, help="Tokens to load from DB")
    parser.add_argument("--out", default="data/datasets/smoke.parquet", help="Output dataset parquet")
    parser.add_argument("--meta-out", default="data/datasets/smoke_meta.parquet", help="Output meta parquet")
    return parser.parse_args()


def _save_df(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_obj, index=False)


async def _run(limit: int, out_path: str, meta_out_path: str) -> None:
    db_url = _get_db_url_from_env()
    await init_pg(db_url)
    pool = get_pool()
    try:
        features, target, meta = await build_dataset_from_db(pool, limit=limit)
    finally:
        await pool.close()

    if features.empty:
        raise RuntimeError("features_df is empty")
    if target.empty:
        raise RuntimeError("target_series is empty")
    if TOKEN_COLUMN not in meta.columns or "target_usd_mcap_3_5m" not in meta.columns:
        raise RuntimeError("meta_df missing required columns")

    dataset = pd.concat([features, target], axis=1)

    logger.info("features shape: %s", features.shape)
    logger.info("target shape: %s", target.shape)
    logger.info("meta shape: %s", meta.shape)
    logger.info("top-5 feature columns: %s", list(features.columns[:5]))
    logger.info(
        "target summary: min=%s max=%s nan=%s",
        float(target.min()),
        float(target.max()),
        int(target.isna().sum()),
    )

    _save_df(dataset, out_path)
    _save_df(meta, meta_out_path)
    logger.info("Saved dataset to %s", out_path)
    logger.info("Saved meta to %s", meta_out_path)


def main() -> None:
    _configure_logging()
    args = _parse_args()
    asyncio.run(_run(args.limit, args.out, args.meta_out))


if __name__ == "__main__":
    main()
