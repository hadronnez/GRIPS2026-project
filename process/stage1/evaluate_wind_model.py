import torch
from torch.utils.data import Dataset

import xarray as xr
import pandas as pd

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
SOLAR_PATH = ROOT / "output/data_preprocessing/clean_datasets/Wind_Power.csv"

PARAMS_DIR = ROOT / "process/stage1/parameters"
MODEL_PATH = PARAMS_DIR / "wind_model.pt"

RESULTS_DIR = ROOT / "output/stage1"


# ============================================================
# EVAL SCOPE FLAG
#
# False (default): evaluate only on the validation split
# (VAL_START..VAL_END), same as before.
# True: evaluate on every available day in the dataset
# (train + val combined), useful for a full-range sanity plot
# rather than a held-out metric. Flip this to switch modes.
# ============================================================

EVAL_ON_FULL_DATASET = True

RESULTS_CSV_PATH = RESULTS_DIR / (
    "wind_forecast.csv" if EVAL_ON_FULL_DATASET else "validation_wind_forecast.csv"
)
RESULTS_PLOT_PATH = RESULTS_DIR / (
    "wind_timeseries.png" if EVAL_ON_FULL_DATASET else "validation_wind_timeseries.png"
)


# ============================================================
# LOAD FORECAST + ACTUAL
#
# Only the raw forecast/actual series are needed here. The
# residual is only recomputed below so the dataset can hand
# back a normalized target tensor of the right shape; the
# normalization itself uses the mean/std stored in the
# checkpoint (not values recomputed fresh in this script), so
# evaluation always matches what the model was trained on.
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


forecast_series, actual_series = load_solar(SOLAR_PATH)

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


# ============================================================
# EVAL WINDOWS (chronological, by day)
#
# By default only the validation split is used. When
# EVAL_ON_FULL_DATASET is True, every valid day (train + val)
# is used instead.
# ============================================================

VAL_START = pd.Timestamp("2025-11-01")
VAL_END   = pd.Timestamp("2025-12-30")

val_days = [
    d for d in valid_days
    if VAL_START <= d <= VAL_END
]

eval_days = valid_days if EVAL_ON_FULL_DATASET else val_days

print("Val days:", len(val_days))
print("Eval days used this run:", len(eval_days), "(full dataset)" if EVAL_ON_FULL_DATASET else "(validation only)")


def make_windows(day_list):
    starts = []
    for d in day_list:
        starts.append(d)
        starts.append(d + pd.Timedelta(hours=12))
    return starts


eval_windows = make_windows(eval_days)

print("Eval windows:", len(eval_windows))


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

        return (
            weather,
            residual_norm,
            forecast_raw,
            actual_raw
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

RESIDUAL_MEAN = checkpoint["residual_mean"]
RESIDUAL_STD = checkpoint["residual_std"]

residual_series = actual_series - forecast_series
residual_series_norm = (residual_series - RESIDUAL_MEAN) / RESIDUAL_STD

eval_dataset = ResidualDataset(cube, eval_windows, residual_series_norm, forecast_series, actual_series)

model = ResidualNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded trained model parameters from: {MODEL_PATH}")


# ============================================================
# FULL ROLLOUT
#
# Reconstruct "forecast + predicted residual" for every
# window in eval_windows, in real units, and compare against
# the real baseline forecast and the ground truth.
# ============================================================

all_times = []
all_actual = []
all_forecast = []
all_improved = []

with torch.no_grad():

    for idx in range(len(eval_dataset)):

        weather, _, forecast_raw, actual_raw = eval_dataset[idx]

        start = eval_windows[idx]

        times = pd.date_range(
            start=start,
            periods=48,
            freq="15min"
        )

        pred_norm = model(
            weather.unsqueeze(0).to(device)
        ).squeeze(0).cpu()

        pred_real = pred_norm * RESIDUAL_STD + RESIDUAL_MEAN

        improved = forecast_raw + pred_real

        all_times.extend(times)
        all_actual.extend(actual_raw.numpy())
        all_forecast.extend(forecast_raw.numpy())
        all_improved.extend(improved.numpy())

results = pd.DataFrame({
    "time": all_times,
    "actual": all_actual,
    "forecast": all_forecast,
    "improved_forecast": all_improved,  # This is the improved forecast
})

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results.to_csv(
    RESULTS_CSV_PATH,
    index=False
)

baseline_mae = (results["actual"] - results["forecast"]).abs().mean()
model_mae = (results["actual"] - results["improved_forecast"]).abs().mean()

print(f"Baseline MAE (forecast vs actual): {baseline_mae:.3f}")
print(f"Model MAE (improved forecast vs actual): {model_mae:.3f}")
print(f"Improvement: {baseline_mae - model_mae:+.3f}")

plt.figure(figsize=(18, 6))

plt.plot(
    results["time"],
    results["actual"],
    label="Actual",
    linewidth=2,
)

plt.plot(
    results["time"],
    results["forecast"],
    label="Forecast",
    alpha=0.7,
)

plt.plot(
    results["time"],
    results["improved_forecast"],
    label="Improved forecast",
    alpha=0.9,
)

plt.legend()
plt.grid(True)

plt.xlabel("Time")
plt.ylabel("Power")

plt.tight_layout()

plt.savefig(RESULTS_PLOT_PATH, dpi=600)
plt.show()

print(f"Wrote {'wind-dataset' if EVAL_ON_FULL_DATASET else 'validation'} CSV to: {RESULTS_CSV_PATH}")
print(f"Wrote {'wind-dataset' if EVAL_ON_FULL_DATASET else 'validation'} plot to: {RESULTS_PLOT_PATH}")
