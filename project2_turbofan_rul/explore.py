"""Exploratory analysis for NASA C-MAPSS FD001."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import load_txt, add_train_rul

TRAIN = "data/train_FD001.txt"
RESULTS = Path("results")
SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

def main():
    RESULTS.mkdir(exist_ok=True)

    df = load_txt(TRAIN)
    train = add_train_rul(df)
    life = df.groupby("unit_id")["cycle"].max()

    plt.figure(figsize=(8, 5))
    plt.hist(life, bins=15)
    plt.title("Engine Life Distribution")
    plt.xlabel("Cycles to failure")
    plt.ylabel("Number of engines")
    plt.tight_layout()
    plt.savefig(RESULTS / "engine_life_distribution.png", dpi=160)
    plt.close()

    sample = train[train["unit_id"] <= 6]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, sensor in zip(axes.flat, SENSORS):
        for unit_id, group in sample.groupby("unit_id"):
            ax.plot(group["cycle"], group[sensor], alpha=0.7, label=f"Unit {unit_id}")
        ax.set_title(sensor)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Reading")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Representative Sensor Trends")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RESULTS / "sensor_readings.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.hist(train["rul"], bins=30)
    plt.title("Capped RUL Distribution")
    plt.xlabel("RUL cycles")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(RESULTS / "rul_distribution.png", dpi=160)
    plt.close()

    corr = train[SENSORS + ["rul"]].corr(numeric_only=True)["rul"].drop("rul").sort_values()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr.values, y=corr.index)
    plt.title("Sensor-to-RUL Correlation")
    plt.xlabel("Correlation")
    plt.ylabel("Sensor")
    plt.tight_layout()
    plt.savefig(RESULTS / "sensor_rul_correlation.png", dpi=160)
    plt.close()

    print(f"Rows: {len(df):,}")
    print(f"Engines: {df['unit_id'].nunique()}")
    print(f"Min life: {life.min()} | Max life: {life.max()} | Mean: {life.mean():.1f}")
    print("Missing values:", int(df.isna().sum().sum()))

if __name__ == "__main__":
    main()
