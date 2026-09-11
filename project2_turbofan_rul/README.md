# Turbofan Engine Remaining Useful Life (RUL) Prediction

A predictive-maintenance project using the **NASA C-MAPSS FD001** run-to-failure dataset.

The system performs:

- engine/sensor data exploration
- piecewise-linear RUL capping at 125 cycles
- sensor selection
- 5-cycle rolling mean and standard deviation features
- MinMax scaling
- Random Forest regression
- Gradient Boosting regression
- PyTorch neural-network regression
- ensemble prediction
- Critical / Warning / Healthy maintenance alerts
- final engine-level prediction reports and plots

## Dataset

Download the C-MAPSS FD001 files and place these three files inside `data/`:

```text
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

The loader expects the standard 26 columns: unit id, cycle, three operating settings, and 21 sensor channels.

## Folder structure

```text
project2_turbofan_rul/
├── data/
├── processed/
├── models/
├── results/
├── feature_engineering.py
├── explore.py
├── preprocess.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Installation

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run in VS Code

### 1. Explore

```bash
python explore.py
```

Creates:

```text
results/engine_life_distribution.png
results/sensor_readings.png
results/rul_distribution.png
results/sensor_rul_correlation.png
```

### 2. Preprocess

```bash
python preprocess.py
```

Creates processed CSV files and a reusable MinMaxScaler.

### 3. Train models

```bash
python train.py
```

Creates:

```text
models/random_forest.pkl
models/gradient_boosting.pkl
models/neural_network.pt
```

and a model comparison chart.

### 4. Generate predictions and alerts

```bash
python predict.py
```

Creates:

```text
results/engine_rul_predictions.csv
results/maintenance_summary.json
results/engine_rul_comparison.png
results/ensemble_scatter.png
```

Alert logic:

- `Critical` → predicted RUL ≤ 30 cycles
- `Warning` → predicted RUL 31–60 cycles
- `Healthy` → predicted RUL > 60 cycles

## Important evaluation note

The training script uses an **engine-level validation split**, which avoids putting cycles from the same engine in both training and validation. This is more appropriate for time-series predictive maintenance than randomly splitting individual rows.

Final test predictions use the last observed cycle of each test engine and compare them with `RUL_FD001.txt`.

## Internship project scope

This repository implements the workflow documented during Weeks 3–4: C-MAPSS exploration, rolling-feature engineering, regression model training, inference, ensemble prediction, and maintenance alert generation.

The reported internship results can be documented in the final report, but the exact numerical scores obtained on a local run depend on the dataset files, Python/package versions, random seed, and hardware.
