from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import pandas as pd

from ml.dataset.db_pool import get_pool, init_pg
from ml.dataset.build_dataset import build_dataset, build_dataset_from_db
from ml.models.infer import predict_for_token
from ml.models.train_lgbm import train_model

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML utilities for max market cap prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-dataset", help="Build dataset from local file")
    build_parser.add_argument("--input", required=True, help="Path to tokens parquet/csv")
    build_parser.add_argument("--out", required=True, help="Output dataset parquet/csv")
    build_parser.add_argument("--meta-out", required=True, help="Output meta parquet/csv")

    build_db_parser = subparsers.add_parser("build-dataset-db", help="Build dataset from database")
    build_db_parser.add_argument("--limit", type=int, default=5000)
    build_db_parser.add_argument("--out", required=True)
    build_db_parser.add_argument("--meta-out", required=True)
    build_db_parser.add_argument("--bundle", default=None)

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data", required=True, help="Dataset parquet/csv")
    train_parser.add_argument("--meta", required=False, help="Meta parquet/csv")
    train_parser.add_argument("--artifacts", required=True, help="Artifacts root folder")

    infer_parser = subparsers.add_parser("infer", help="Run inference for token trades")
    infer_parser.add_argument("--token", required=True)
    infer_parser.add_argument("--trades", required=True, help="Path to trades json")
    infer_parser.add_argument("--model", required=True, help="Path to model artifact directory")

    return parser.parse_args()


def _load_df(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if path_obj.suffix == ".parquet":
        return pd.read_parquet(path_obj)
    return pd.read_csv(path_obj)


def _save_df(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if path_obj.suffix == ".parquet":
        df.to_parquet(path_obj, index=False)
    else:
        df.to_csv(path_obj, index=False)


def _load_trades(path: str) -> list[dict]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _get_db_url_from_env() -> str:
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not db_url:
        raise RuntimeError("DB_URL/DATABASE_URL/POSTGRES_DSN environment variable is not set")
    return db_url


def main() -> None:
    _configure_logging()
    args = _parse_args()

    if args.command == "build-dataset":
        df_tokens = _load_df(args.input)
        features, target, meta = build_dataset(df_tokens)
        dataset = pd.concat([features, target], axis=1)
        _save_df(dataset, args.out)
        _save_df(meta, args.meta_out)
        logger.info("Saved dataset to %s", args.out)
        return

    if args.command == "build-dataset-db":
        db_url = _get_db_url_from_env()

        async def _run() -> None:
            await init_pg(db_url)
            pool = get_pool()
            try:
                features, target, meta = await build_dataset_from_db(
                    pool, limit=args.limit, bundle_name=args.bundle
                )
                dataset = pd.concat([features, target], axis=1)
                _save_df(dataset, args.out)
                _save_df(meta, args.meta_out)
                logger.info("Saved dataset to %s", args.out)
            finally:
                await pool.close()

        asyncio.run(_run())
        return

    if args.command == "train":
        artifacts_dir = train_model(args.data, args.meta, args.artifacts)
        logger.info("Training complete: %s", artifacts_dir)
        return

    if args.command == "infer":
        trades = _load_trades(args.trades)
        result = predict_for_token(trades, args.token, args.model, include_features=True)
        print(result)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
