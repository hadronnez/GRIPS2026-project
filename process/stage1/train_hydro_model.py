import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import xarray as xr
import pandas as pd

import numpy as np
from pathlib import Path

from model.model import ResidualNet

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project"
)

METEO_PATH = ROOT / "output/data_preprocessing/clean_datasets/clean_nwp_dataset.nc"
SOLAR_PATH = ROOT / "output/data_preprocessing/clean_datasets/Hydro_Power.csv"

PARAMS_DIR = ROOT / "process/stage1/parameters"
MODEL_PATH = PARAMS_DIR / "hydro_model.pt"


# ============================================================
# LOAD FORECAST + ACTUAL, BUILD RESIDUAL TARGET
#
# The task changed: instead of modeling production directly,
# we now model the RESIDUAL = actual - forecast. The
# day-ahead forecast is the baseline; the weather cube (which
# presumably carries more accurate / higher-resolution
# information than what went into the day-ahead forecast) is
# used to predict the *correction* on top of that baseline.
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

residual_series = actual_series - forecast_series

print("Forecast:", forecast_series.shape)
print("Actual:", actual_series.shape)
print("Residual (actual - forecast):", residual_series.shape)
print("Residual describe:\n", residual_series.describe())


# ============================================================
# RESIDUAL NORMALIZATION (z-score)
#
# Unlike raw production, the residual is signed and roughly
# centered around 0, so a z-score (mean/std) is the natural
# normalization here rather than dividing by a max.
# ============================================================

RESIDUAL_MEAN = residual_series.mean()
RESIDUAL_STD = residual_series.std()

print("Residual mean:", RESIDUAL_MEAN, "std:", RESIDUAL_STD)

residual_series_norm = (residual_series - RESIDUAL_MEAN) / RESIDUAL_STD


# ============================================================
# METEO LAZY LOAD
# ============================================================

ds = xr.open_dataset(METEO_PATH)
cube = ds["cube"]

print(cube)


# ============================================================
# AVAILABLE DAYS
#
# Extended to a full year now that a year of data is
# available. Adjust the end date if your actual coverage is
# different -- this just needs to bracket whatever your .nc
# file actually contains; days outside the file's range are
# silently dropped by the valid_days check below anyway.
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
# TRAIN / VAL SPLIT (chronological, by day)
#
# Splitting by shuffled windows would leak information across
# the train/val boundary (the two 12h windows of the same day
# share a lot of weather correlation). Splitting by day, in
# time order, gives an honest read on whether the model
# generalizes to unseen days rather than just memorizing.
# ============================================================


TRAIN_START = pd.Timestamp("2025-01-02")
TRAIN_END   = pd.Timestamp("2025-10-31")

VAL_START = pd.Timestamp("2025-11-01")
VAL_END   = pd.Timestamp("2025-12-30")

train_days = [
    d for d in valid_days
    if TRAIN_START <= d <= TRAIN_END
]

val_days = [
    d for d in valid_days
    if VAL_START <= d <= VAL_END
]

print("Train days:", len(train_days), "Val days:", len(val_days))


def make_windows(day_list):
    starts = []
    for d in day_list:
        starts.append(d)
        starts.append(d + pd.Timedelta(hours=12))
    return starts


train_windows = make_windows(train_days)
val_windows = make_windows(val_days)

print("Train windows:", len(train_windows), "Val windows:", len(val_windows))


# ============================================================
# DATASET
#
# Returns, per window:
#   weather          (12,7,104,225) float32
#   residual_norm    (48,) float32   -- training target
#   forecast_raw     (48,) float32   -- baseline, real units
#   actual_raw       (48,) float32   -- ground truth, real units
#
# forecast_raw / actual_raw aren't used in the loss -- they're
# carried along so we can reconstruct "forecast + predicted
# residual" and compare it against the real baseline at eval
# time, in real units.
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


train_dataset = ResidualDataset(cube, train_windows, residual_series_norm, forecast_series, actual_series)
val_dataset = ResidualDataset(cube, val_windows, residual_series_norm, forecast_series, actual_series)

# Batch size raised 4 -> 8: now that the model is much
# cheaper per sample (see ResidualNet below), a bigger batch
# uses your CPU's vectorized ops more efficiently instead of
# paying per-call Python/tensor overhead more times.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# ============================================================
# LOSS
#
# Plain MSE on the normalized residual, plus a small temporal
# smoothness term (physical residuals shouldn't jump wildly
# between consecutive 15-min steps). No sparsity/L1 terms --
# those belonged to the old spatial-map formulation.
# ============================================================

TV_WEIGHT = 1e-3


def loss_function(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    tv_t = torch.abs(pred[:, 1:] - pred[:, :-1]).mean()
    return mse + TV_WEIGHT * tv_t


# ============================================================
# TRAIN
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResidualNet().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)

print("Number of parameters:", sum(p.numel() for p in model.parameters()))

N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    model.train()
    total = 0

    for i, (weather, residual_norm, _, _) in enumerate(train_loader):

        weather = weather.to(device)
        residual_norm = residual_norm.to(device)

        pred = model(weather)
        loss = loss_function(pred, residual_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_loss = total / len(train_loader)

    # --------------------------------------------------------
    # Validation: compare model MAE (forecast + predicted
    # residual vs actual) against the baseline MAE (forecast
    # alone vs actual). If the model isn't beating the
    # baseline on held-out days, it isn't adding value.
    # --------------------------------------------------------

    model.eval()

    baseline_abs_err = []
    model_abs_err = []

    with torch.no_grad():
        for weather, residual_norm, forecast_raw, actual_raw in val_loader:

            weather = weather.to(device)

            pred_norm = model(weather).cpu()
            pred_real = pred_norm * RESIDUAL_STD + RESIDUAL_MEAN

            corrected_forecast = forecast_raw + pred_real

            baseline_abs_err.append((actual_raw - forecast_raw).abs())
            model_abs_err.append((actual_raw - corrected_forecast).abs())

    baseline_mae = torch.cat(baseline_abs_err).mean().item() if baseline_abs_err else float("nan")
    model_mae = torch.cat(model_abs_err).mean().item() if model_abs_err else float("nan")

    print(
        f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
        f"baseline_MAE {baseline_mae:.3f}  model_MAE {model_mae:.3f}  "
        f"improvement {baseline_mae - model_mae:+.3f}"
    )


# ============================================================
# SAVE MODEL PARAMETERS
#
# Bundle the trained weights together with the residual
# normalization stats (mean/std) fit on this run's data. The
# evaluation script needs both to turn the model's normalized
# residual output back into real power units.
# ============================================================

PARAMS_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "residual_mean": RESIDUAL_MEAN,
        "residual_std": RESIDUAL_STD,
    },
    MODEL_PATH,
)

print(f"Saved trained model parameters to: {MODEL_PATH}")
