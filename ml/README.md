# ML (sklearn)

## Датасет из DB
```bash
python -m ml.cli build-dataset --limit 5000 --out ml/data/ds_v1.parquet
```

## Датасет для предикта по одному токену
```bash
python -m ml.cli build-predict-dataset --token-mint <TOKEN_MINT> --out ml/data/predict_v1.parquet
```

## Обучение
```bash
python -m ml.cli train --data ml/data/ds_v1.parquet --model-out ml/artifacts/model.joblib
```

## Предсказание
```bash
python -m ml.cli predict --model ml/artifacts/model.joblib --data ml/data/ds_v1.parquet --out data/preds.csv
```

## Примечания
- Фичи строятся по трейдам в окне 0–3 минуты от T0, таргет — max USD market cap в окне 3–5 минут.
- `tokens.supply` используется как уже нормализованное значение.
- Bundle‑кошельки (опционально) берутся из `ml/data/bundle_wallets.txt`.
