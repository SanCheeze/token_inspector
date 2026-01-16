# Max Market Cap Prediction (5-minute window)

## Goal
Build a model that predicts a token's **max market cap** using only trades from the first 5 minutes after launch.

## Features
Extracted per token (see `ml/schema.py` for stable ordering):

- **Trades / Volume**: trade counts, buy/sell ratio, USD volume stats.
- **Wallets / concentration**: unique wallets, top wallet volume share, Gini.
- **Timing**: time-to-first/10 trades/100 USD/1000 USD, acceleration.
- **Platforms / routing**: platform counts, top platform share, aggregator route flag.
- **Bundle wallets** (optional): bundle share/participation metrics.

All features are computed **only within `[t0, t0 + 300s]`** to prevent leakage.

## Dataset
### DB mode
```bash
python -m ml.cli.ml_cli build-dataset-db --limit 5000 --out data/dataset.parquet --meta-out data/dataset_meta.parquet --bundle default
```

### File mode
```bash
python -m ml.cli.ml_cli build-dataset --input data/tokens.parquet --out data/dataset.parquet --meta-out data/dataset_meta.parquet
```

Notes:
- `trades` can be JSON or stringified JSON.
- `value` is treated as USD and trades with non-positive value are skipped.
- Bundle wallets default to `data/bundle_wallets.txt` if present (one wallet per line).
- If your schema uses different column names, update `ml/config.py` accordingly.

## Training
```bash
python -m ml.cli.ml_cli train --data data/dataset.parquet --meta data/dataset_meta.parquet --artifacts artifacts/ml_mcap_model
```

Artifacts:
```
artifacts/ml_mcap_model/<timestamp>/
  model.bin
  feature_names.json
  metrics.json
  feature_importance.csv
  config.json
```

## Inference
```bash
python -m ml.cli.ml_cli infer --token <mint> --trades data/trades.json --model artifacts/ml_mcap_model/<timestamp>
```

## Bundle wallets
If your DB does not have a bundle table, the ML module falls back to `data/bundle_wallets.txt`.
If you have a source table, update `ml/dataset/db.py` accordingly.

## Optional prediction storage
`ml/dataset/db.py` includes `save_token_prediction(...)` which writes to `tokens.ml_pred` if present.
If the column does not exist, the function returns without saving.
