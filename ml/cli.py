from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ml.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from ml.dataset.builder import build_dataset_from_db, save_dataset
from ml.modeling.predict import predict_from_dataset, predict_from_db, save_predictions
from ml.modeling.train import train_model


def _build_dataset(args: argparse.Namespace) -> None:
    async def _run():
        df = await build_dataset_from_db(
            dsn=args.dsn,
            limit=args.limit,
            bundle_name=args.bundle_name,
        )
        save_dataset(df, args.out)

    asyncio.run(_run())


def _train(args: argparse.Namespace) -> None:
    train_model(args.data, args.model_out, args.metrics_out, model_type=args.model_type)


def _predict(args: argparse.Namespace) -> None:
    if args.data:
        df = predict_from_dataset(args.model, args.data)
        save_predictions(df, args.out)
        return

    async def _run():
        df_features = await predict_from_db(
            dsn=args.dsn,
            limit=args.limit,
            bundle_name=args.bundle_name,
        )
        model = None
        import joblib

        model = joblib.load(args.model)
        preds = model.predict(df_features.drop(columns=["mint"]))
        df_out = df_features[["mint"]].copy()
        df_out["y_pred"] = preds
        save_predictions(df_out, args.out)

    asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-dataset", help="Build dataset from DB")
    build_parser.add_argument("--dsn", required=True, help="Database DSN")
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--bundle-name", default=None)
    build_parser.add_argument("--out", default=DEFAULT_DATASET_PATH)
    build_parser.set_defaults(func=_build_dataset)

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--metrics-out", default="ml/artifacts/metrics.json")
    train_parser.add_argument("--model-type", default="hgb", choices=["hgb", "rf"])
    train_parser.set_defaults(func=_train)

    predict_parser = subparsers.add_parser("predict", help="Predict using model")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--data", default=None)
    predict_parser.add_argument("--dsn", default=None)
    predict_parser.add_argument("--limit", type=int, default=None)
    predict_parser.add_argument("--bundle-name", default=None)
    predict_parser.add_argument("--out", default="data/preds.csv")
    predict_parser.set_defaults(func=_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "predict" and not args.data and not args.dsn:
        raise SystemExit("predict requires --data or --dsn")
    args.func(args)


if __name__ == "__main__":
    Path("ml/artifacts").mkdir(parents=True, exist_ok=True)
    main()
