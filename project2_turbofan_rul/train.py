"""Train Random Forest, Gradient Boosting, and PyTorch Neural Network."""

from pathlib import Path
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA = Path("processed/train_processed.csv")
OUT = Path("models")
RESULTS = Path("results")

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

def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def main():
    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in {"unit_id", "cycle", "rul"}]

    # Engine-level validation split prevents nearby cycles from leaking across sets.
    units = df["unit_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(units)
    split = int(len(units) * 0.80)
    train_units, val_units = units[:split], units[split:]

    train_df = df[df.unit_id.isin(train_units)]
    val_df = df[df.unit_id.isin(val_units)]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["rul"].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["rul"].values.astype(np.float32)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=3,
            min_samples_leaf=5, random_state=SEED
        ),
    }

    summary = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_val), 0, 125)
        summary[name] = metrics(y_val, pred)
        filename = OUT / ("random_forest.pkl" if name == "Random Forest" else "gradient_boosting.pkl")
        joblib.dump({"model": model, "features": feature_cols}, filename)
        print(name, summary[name])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device)
    print("PyTorch device:", device)

    model = Regressor(len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True
    )

    for epoch in range(80):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_val).to(device)).squeeze(1).cpu().numpy()
    pred = np.clip(pred, 0, 125)

    summary["Neural Network"] = metrics(y_val, pred)
    torch.save(
        {"state_dict": model.state_dict(), "features": feature_cols, "input_dim": len(feature_cols)},
        OUT / "neural_network.pt",
    )

    with (RESULTS / "model_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    names = list(summary)
    rmses = [summary[n]["RMSE"] for n in names]

    plt.figure(figsize=(9, 5))
    plt.bar(names, rmses)
    plt.title("Validation RMSE by Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
