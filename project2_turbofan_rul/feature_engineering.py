"""Shared preprocessing and feature-engineering functions for C-MAPSS FD001."""

from pathlib import Path
import numpy as np
import pandas as pd

COLUMN_NAMES = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

DROP_SENSORS = {
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19"
}

ROLLING_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17"
]

def load_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26]
    df.columns = COLUMN_NAMES
    return df

def add_train_rul(df, cap=125):
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df

def add_test_rul(df, rul_df, cap=125):
    """Assign true RUL to the final test cycle of each unit.

    The official RUL file gives the remaining cycles after the last observed
    cycle, so values are joined only to each unit's final test row.
    """
    df = df.copy()
    rul_values = np.asarray(rul_df).reshape(-1)
    final_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = np.nan
    final_mask = df["cycle"].eq(final_cycles)

    lookup = dict(zip(range(1, len(rul_values) + 1), rul_values))
    df.loc[final_mask, "rul"] = df.loc[final_mask, "unit_id"].map(lookup).clip(upper=cap)
    return df

def add_rolling_features(df, window=5):
    df = df.copy()
    for sensor in ROLLING_SENSORS:
        grouped = df.groupby("unit_id", group_keys=False)[sensor]
        df[f"{sensor}_roll_mean"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_roll_std"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=2).std().fillna(0.0)
        )
    return df

def clean_features(df):
    feature_df = df.drop(columns=["rul"], errors="ignore").copy()
    feature_df = feature_df.drop(columns=["unit_id"], errors="ignore")
    feature_df = feature_df.drop(columns=list(DROP_SENSORS), errors="ignore")
    return feature_df

def build_training_frame(path, cap=125):
    df = load_txt(path)
    df = add_train_rul(df, cap)
    df = df.sort_values(["unit_id", "cycle"])
    df = add_rolling_features(df)
    return df

def build_test_frame(test_path, rul_path, cap=125):
    df = load_txt(test_path)
    rul = pd.read_csv(rul_path, header=None).iloc[:, 0]
    df = df.sort_values(["unit_id", "cycle"])
    df = add_rolling_features(df)
    df = add_test_rul(df, rul, cap)
    return df
