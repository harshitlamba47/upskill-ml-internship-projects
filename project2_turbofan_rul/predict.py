"""Generate final test-engine RUL estimates and maintenance alerts."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

from feature_engineering import build_test_frame, clean_features

TEST = "data/test_FD001.txt"
RUL = "data/RUL_FD001.txt"
MODEL_DIR = Path("models")
RESULTS = Path("results")
CAP = 125

class Regressor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

def alert(rul):
    if rul <= 30:
        return "Critical"
    if rul <= 60:
        return "Warning"
    return "Healthy"

def final_rows(df):
    return df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1).sort_values("unit_id")

def main():
    RESULTS.mkdir(exist_ok=True)

    test = build_test_frame(TEST, RUL)
    final = final_rows(test)
    X = clean_features(final)

    rf_pack = joblib.load(MODEL_DIR / "random_forest.pkl")
    gb_pack = joblib.load(MODEL_DIR / "gradient_boosting.pkl")

    rf = rf_pack["model"]
    gb = gb_pack["model"]
    features = rf_pack["features"]
    X_np = X[features].values.astype(np.float32)

    rf_pred = np.clip(rf.predict(X_np), 0, CAP)
    gb_pred = np.clip(gb.predict(X_np), 0, CAP)

    nn_pack = torch.load(MODEL_DIR / "neural_network.pt", map_location="cpu")
    nn_model = Regressor(nn_pack["input_dim"])
    nn_model.load_state_dict(nn_pack["state_dict"])
    nn_model.eval()

    with torch.no_grad():
        nn_pred = nn_model(torch.from_numpy(X_np)).squeeze(1).numpy()
    nn_pred = np.clip(nn_pred, 0, CAP)

    ensemble = np.clip((rf_pred + gb_pred + nn_pred) / 3.0, 0, CAP)
    actual = final["rul"].values.astype(float)

    out = pd.DataFrame({
        "engine_id": final["unit_id"].astype(int).values,
        "actual_rul": actual,
        "random_forest_rul": rf_pred,
        "gradient_boosting_rul": gb_pred,
        "neural_network_rul": nn_pred,
        "ensemble_rul": ensemble,
    })
    out["status"] = out["ensemble_rul"].apply(alert)

    out.to_csv(RESULTS / "engine_rul_predictions.csv", index=False)

    counts = out["status"].value_counts().reindex(
        ["Critical", "Warning", "Healthy"], fill_value=0
    )
    with (RESULTS / "maintenance_summary.json").open("w") as f:
        json.dump(counts.to_dict(), f, indent=2)

    plt.figure(figsize=(14, 6))
    plt.plot(out["engine_id"], actual, label="Actual RUL")
    plt.plot(out["engine_id"], ensemble, label="Ensemble Predicted RUL")
    plt.axhline(30, linestyle="--", label="Critical threshold")
    plt.axhline(60, linestyle="--", label="Warning threshold")
    plt.xlabel("Engine")
    plt.ylabel("RUL (cycles)")
    plt.title("Actual vs Ensemble Predicted RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "engine_rul_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, ensemble, alpha=0.7)
    plt.plot([0, CAP], [0, CAP], linestyle="--", label="Perfect prediction")
    plt.xlabel("Actual RUL")
    plt.ylabel("Ensemble predicted RUL")
    plt.title("Actual vs Ensemble RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "ensemble_scatter.png", dpi=160)
    plt.close()

    print("\nMaintenance summary")
    print(counts)
    print("\nSample predictions")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
