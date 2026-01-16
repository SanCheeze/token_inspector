from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    time_column: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy(), df.copy()
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    total = len(df_sorted)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
    valid_df = df_sorted.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = df_sorted.iloc[valid_end:].reset_index(drop=True)
    return train_df, valid_df, test_df
