"""Feature engineering and MinMax preprocessing for C-MAPSS FD001."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from feature_engineering import build_training_frame, build_test_frame, clean_features

TRAIN = "data/train_FD001.txt"
TEST = "data/test_FD001.txt"
RUL = "data/RUL_FD001.txt"
OUT = Path("processed")
SCALER_PATH = Path("models/minmax_scaler.pkl")

def main():
    OUT.mkdir(exist_ok=True)
    SCALER_PATH.parent.mkdir(exist_ok=True)

    train = build_training_frame(TRAIN)
    test = build_test_frame(TEST, RUL)

    train_x = clean_features(train)
    test_x = clean_features(test)

    target = train["rul"]

    scaler = MinMaxScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    train_out = pd.DataFrame(train_x_scaled, columns=train_x.columns)
    train_out["unit_id"] = train["unit_id"].values
    train_out["cycle"] = train["cycle"].values
    train_out["rul"] = target.values

    test_out = pd.DataFrame(test_x_scaled, columns=test_x.columns)
    test_out["unit_id"] = test["unit_id"].values
    test_out["cycle"] = test["cycle"].values
    test_out["rul"] = test["rul"].values

    train_out.to_csv(OUT / "train_processed.csv", index=False)
    test_out.to_csv(OUT / "test_processed.csv", index=False)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Train shape: {train_out.shape}")
    print(f"Test shape: {test_out.shape}")
    print(f"Features: {train_x.shape[1]}")
    print("Saved processed CSV files and scaler.")

if __name__ == "__main__":
    main()
