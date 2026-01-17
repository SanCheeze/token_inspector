# ML: Max Market Cap Prediction (5-minute window)

## Цель
Построить модель, которая предсказывает **максимальную капитализацию токена** по сделкам из первых 5 минут после запуска.

## Признаки
Вычисляются на токен (фиксированный порядок — `ml/schema.py`):

- **Trades / Volume**: количество сделок, отношение buy/sell, статистики объёма в USD.
- **Wallets / concentration**: уникальные кошельки, доля объёма топ-кошелька, Gini.
- **Timing**: время до 1/10 сделок, 100/1000 USD, ускорение.
- **Platforms / routing**: количество платформ, доля топ-платформы, флаг роутинга через агрегатор.
- **Bundle wallets** (опционально): доля/участие bundle-кошельков.

Все признаки считаются **только в окне `[t0, t0 + 300s]`**, чтобы избежать leakage.

## Создание датасета
Датасеты **нужно сохранять внутри модуля `ml` в `ml/data/`**.

### Режим DB
```bash
python -m ml.cli.ml_cli build-dataset-db \
  --limit 5000 \
  --out ml/data/dataset.parquet \
  --meta-out ml/data/dataset_meta.parquet \
  --bundle default
```

### Режим файла
```bash
python -m ml.cli.ml_cli build-dataset \
  --input data/tokens.parquet \
  --out ml/data/dataset.parquet \
  --meta-out ml/data/dataset_meta.parquet
```

Примечания:
- `trades` может быть JSON или строковым JSON.
- `value` считается USD; сделки с не‑положительным значением пропускаются.
- Bundle‑кошельки берутся из `data/bundle_wallets.txt` (по одному адресу в строке), если файл существует.
- Если в вашей схеме другие имена колонок — настройте `ml/config.py`.

## Обучение модели
```bash
python -m ml.cli.ml_cli train \
  --data ml/data/dataset.parquet \
  --meta ml/data/dataset_meta.parquet \
  --artifacts artifacts/ml_mcap_model
```

Структура артефактов:
```
artifacts/ml_mcap_model/<timestamp>/
  model.bin
  feature_names.json
  metrics.json
  feature_importance.csv
  config.json
```

## Использование натренированной модели (инференс)
```bash
python -m ml.cli.ml_cli infer \
  --token <mint> \
  --trades data/trades.json \
  --model artifacts/ml_mcap_model/<timestamp>
```

## Bundle wallets
Если в БД нет таблицы с bundle‑кошельками, ML модуль использует `data/bundle_wallets.txt`.
Если таблица есть — обновите `ml/dataset/db.py`.

## Опциональное сохранение предсказаний
В `ml/dataset/db.py` есть `save_token_prediction(...)`, которая пишет в `tokens.ml_pred`.
Если колонки нет, функция завершится без сохранения.
