import torch
from torch.utils.data import Dataset

import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
from model.model import ResidualNet

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project"
)

METEO_PATH = ROOT / "output/data_preprocessing/clean_datasets/clean_nwp_dataset.nc"
DATA_PATH = ROOT / "output/data_preprocessing/clean_datasets/Hydro_Power.csv" 

PARAMS_DIR = ROOT / "process/stage1/parameters"

RESULTS_DIR = ROOT / "output/stage1"
RESULTS_CSV_PATH = RESULTS_DIR / "hydro_improved_forecast.csv"
RESULTS_PLOT_PATH = RESULTS_DIR / "hydro_improved_timeseries.png"


# ============================================================
# CROSS-FITTING FOLDS
#
# Must match the FOLDS definition used in the training script.
# Each fold's checkpoint is only ever applied to the days it
# was held out from during training, so every prediction this
# script produces is genuinely out-of-fold -- no train/val
# flag needed, the whole year comes out honest by construction.
# ============================================================

FOLDS = [
    {"name": "Q1", "val_start": pd.Timestamp("2025-01-01"), "val_end": pd.Timestamp("2025-03-31")},
    {"name": "Q2", "val_start": pd.Timestamp("2025-04-01"), "val_end": pd.Timestamp("2025-06-30")},
    {"name": "Q3", "val_start": pd.Timestamp("2025-07-01"), "val_end": pd.Timestamp("2025-09-30")},
    {"name": "Q4", "val_start": pd.Timestamp("2025-10-01"), "val_end": pd.Timestamp("2025-12-30")},
]


# ============================================================
# LOAD FORECAST + ACTUAL
#
# Only the raw forecast/actual series are needed here. Each
# fold's residual normalization (mean/std) comes from that
# fold's own checkpoint, not recomputed fresh in this script,
# so evaluation always matches what each model was trained on.
# ============================================================

COL_CFG = {
    "actual": ["actual_00", "actual_15", "actual_30", "actual_45"],
    "forecast": ["forecast_00", "forecast_15", "forecast_30", "forecast_45"],
}


def wide_to_15min(df, cols):
    values = df[cols].values.reshape(-1)
    index = pd.date_range(start=df.index[0], periods=len(values), freq="15min")
    return pd.Series(values, index=index)


def load_solar(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    forecast = wide_to_15min(df, COL_CFG["forecast"])
    actual = wide_to_15min(df, COL_CFG["actual"])
    return forecast, actual


forecast_series, actual_series = load_solar(DATA_PATH)
residual_series = actual_series - forecast_series

print("Forecast:", forecast_series.shape)
print("Actual:", actual_series.shape)


# ============================================================
# METEO LAZY LOAD
# ============================================================

ds = xr.open_dataset(METEO_PATH)
cube = ds["cube"]

print(cube)


# ============================================================
# AVAILABLE DAYS
# ============================================================

days = pd.date_range("2025-01-01", "2025-12-30", freq="D")

valid_days = []
for d in days:
    start = d
    end = d + pd.Timedelta(hours=23)
    if (
        start.to_datetime64() in cube.time.values
        and end.to_datetime64() in cube.time.values
    ):
        valid_days.append(d)

print("Days:", len(valid_days))


def make_windows(day_list):
    starts = []
    for d in day_list:
        starts.append(d)
        starts.append(d + pd.Timedelta(hours=12))
    return starts


# ============================================================
# DATASET
# ============================================================

class ResidualDataset(Dataset):

    def __init__(self, cube, window_starts, residual_norm, forecast, actual):
        self.cube = cube
        self.window_starts = window_starts
        self.residual_norm = residual_norm
        self.forecast = forecast
        self.actual = actual

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end = start + pd.Timedelta(hours=11)

        weather = self.cube.sel(time=slice(start, end)).values
        weather = torch.tensor(weather, dtype=torch.float32)

        mask = (
            (self.residual_norm.index >= start)
            & (self.residual_norm.index < start + pd.Timedelta(hours=12))
        )

        residual_slice = self.residual_norm.loc[mask]
        forecast_slice = self.forecast.loc[mask]
        actual_slice = self.actual.loc[mask]

        residual_norm = torch.tensor(residual_slice.values, dtype=torch.float32)
        forecast_raw = torch.tensor(forecast_slice.values, dtype=torch.float32)
        actual_raw = torch.tensor(actual_slice.values, dtype=torch.float32)

        return weather, residual_norm, forecast_raw, actual_raw


# ============================================================
# R^2 HELPER
# ============================================================

def r2_score(actual, pred):
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ============================================================
# OUT-OF-FOLD ROLLOUT
#
# For each fold, load ONLY that fold's checkpoint and run it
# ONLY on the days it was held out from during training. This
# reproduces, standalone, the same out-of-fold table the
# training script writes -- useful for re-plotting or
# re-checking metrics without retraining.
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

all_times, all_actual, all_forecast, all_improved, all_fold = [], [], [], [], []

for fold in FOLDS:

    fold_name = fold["name"]
    val_start, val_end = fold["val_start"], fold["val_end"]

    fold_model_path = PARAMS_DIR / f"hydro_model_fold_{fold_name}.pt"
    checkpoint = torch.load(fold_model_path, map_location=device, weights_only=False)

    RESIDUAL_MEAN = checkpoint["residual_mean"]
    RESIDUAL_STD = checkpoint["residual_std"]

    residual_series_norm = (residual_series - RESIDUAL_MEAN) / RESIDUAL_STD

    val_days = [d for d in valid_days if val_start <= d <= val_end]
    val_windows = make_windows(val_days)

    eval_dataset = ResidualDataset(cube, val_windows, residual_series_norm, forecast_series, actual_series)

    model = ResidualNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded fold {fold_name} model from: {fold_model_path}  "
          f"({len(val_days)} held-out days)")

    with torch.no_grad():
        for idx in range(len(eval_dataset)):

            weather, _, forecast_raw, actual_raw = eval_dataset[idx]
            start = val_windows[idx]

            times = pd.date_range(start=start, periods=48, freq="15min")

            pred_norm = model(weather.unsqueeze(0).to(device)).squeeze(0).cpu()
            pred_real = pred_norm * RESIDUAL_STD + RESIDUAL_MEAN
            improved = forecast_raw + pred_real

            all_times.extend(times)
            all_actual.extend(actual_raw.numpy())
            all_forecast.extend(forecast_raw.numpy())
            all_improved.extend(improved.numpy())
            all_fold.extend([fold_name] * len(times))


results = pd.DataFrame({
    "time": all_times,
    "actual": all_actual,
    "forecast": all_forecast,
    "improved_forecast": all_improved,  # out-of-fold improved forecast
    "fold": all_fold,                    # which held-out quarter produced this row
})

results = results.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULTS_CSV_PATH, index=False)


# ============================================================
# METRICS (whole year, all out-of-fold)
# ============================================================

baseline_mae = (results["actual"] - results["forecast"]).abs().mean()
model_mae = (results["actual"] - results["improved_forecast"]).abs().mean()

baseline_r2 = r2_score(results["actual"].values, results["forecast"].values)
model_r2 = r2_score(results["actual"].values, results["improved_forecast"].values)

print(f"\nBaseline MAE (forecast vs actual):        {baseline_mae:.3f}")
print(f"Model MAE (improved forecast vs actual):  {model_mae:.3f}")
print(f"MAE improvement:                          {baseline_mae - model_mae:+.3f}")
print(f"Baseline R2 (forecast vs actual):         {baseline_r2:.4f}")
print(f"Model R2 (improved forecast vs actual):   {model_r2:.4f}")


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(18, 6))

plt.plot(results["time"], results["actual"], label="Actual", linewidth=2)
plt.plot(results["time"], results["forecast"], label="Forecast", alpha=0.7)
plt.plot(results["time"], results["improved_forecast"], label="Improved forecast", alpha=0.9)

plt.legend()
plt.grid(True)

plt.xlabel("Time")
plt.ylabel("Power")

plt.tight_layout()
plt.savefig(RESULTS_PLOT_PATH, dpi=600)
plt.show()

print(f"\nWrote out-of-fold CSV to: {RESULTS_CSV_PATH}")
print(f"Wrote out-of-fold plot to: {RESULTS_PLOT_PATH}")