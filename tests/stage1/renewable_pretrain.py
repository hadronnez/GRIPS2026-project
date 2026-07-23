import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

from model.model import Model as S1NN

# ============================================================
# PATHS
#
# One data file per commodity -- same paths as `renewables` in
# configs.json. Pass --commodity solar|wind|hydro to pick which
# branch this run trains; everything else (windows, model,
# folds) is identical across the three.
# ============================================================

ROOT = Path(
    "C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project"
)

METEO_PATH = ROOT / "output/data_preprocessing/clean_datasets/clean_nwp_dataset.nc"

COMMODITY_DATA_PATHS = {
    "solar": ROOT / "output/data_preprocessing/clean_datasets/Photovoltaic.csv",
    "wind": ROOT / "output/data_preprocessing/clean_datasets/Wind_Power.csv",
    "hydro": ROOT / "output/data_preprocessing/clean_datasets/Hydro_Power.csv",
}

PARAMS_DIR = ROOT / "process/nn/pretrained_parameters/"

# Edit this if you're running via an IDE "Run" button that doesn't pass
# CLI args -- --commodity on the command line always overrides it.
DEFAULT_COMMODITY = "hydro"

parser = argparse.ArgumentParser(description="Stage 1 pretraining for one commodity")
parser.add_argument("--commodity", choices=list(COMMODITY_DATA_PATHS), default=DEFAULT_COMMODITY)
args = parser.parse_args()

COMMODITY = args.commodity
DATA_PATH = COMMODITY_DATA_PATHS[COMMODITY]

# ============================================================
# STAGE 1 CONFIG
#
# Same values as stage1 in configs.json -- kept inline here so this
# standalone pretraining script doesn't need to import the full
# common.py/architecture.py chain just to build one branch. If you
# ever change stage1 in configs.json, mirror the change here too,
# or better: load it from configs.json directly.
# ============================================================

STAGE1_CONFIG = SimpleNamespace(
    in_channels=7,
    encoder_channels=[16, 32],
    encoder_norm_groups=[4, 8],
    encoder_kernel_size=3,
    encoder_padding_mode="replicate",
    num_temporal_upsamples=2,   # 24h -> 48 -> 96 (quarter-hours), matches SLOTS_PER_DAY=96
    upsample_kernel_size=2,
    upsample_stride=2,
    head_hidden_channels=16,
    head_kernel_size=3,
    head_norm_groups=4,
    out_channels=1,
)

SLOTS_PER_DAY = 96
HOURS_PER_DAY = 24

# ============================================================
# LOAD FORECAST + ACTUAL, BUILD RESIDUAL TARGET
#
# We model the RESIDUAL = actual - forecast. The day-ahead
# forecast is the baseline; the weather cube is used to
# predict the *correction* on top of that baseline.
# ============================================================

COL_CFG = {
    "actual": ["actual_00", "actual_15", "actual_30", "actual_45"],
    "forecast": ["forecast_00", "forecast_15", "forecast_30", "forecast_45"],
}


def wide_to_15min(df, cols):
    values = df[cols].values.reshape(-1)
    index = pd.date_range(start=df.index[0], periods=len(values), freq="15min")
    return pd.Series(values, index=index)


def load_series(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    forecast = wide_to_15min(df, COL_CFG["forecast"])
    actual = wide_to_15min(df, COL_CFG["actual"])
    return forecast, actual


forecast_series, actual_series = load_series(DATA_PATH)
residual_series = actual_series - forecast_series

print(f"Commodity: {COMMODITY}")
print("Forecast:", forecast_series.shape)
print("Actual:", actual_series.shape)
print("Residual (actual - forecast):", residual_series.shape)
print("Residual describe:\n", residual_series.describe())


# ============================================================
# METEO LAZY LOAD
# ============================================================

ds = xr.open_dataset(METEO_PATH)
cube = ds["cube"]

print(cube)


# ============================================================
# AVAILABLE DAYS (days with full 24h coverage in the cube)
# ============================================================

days = pd.date_range("2025-01-01", "2025-12-30", freq="D")

valid_days = []
for d in days:
    start = d
    end = d + pd.Timedelta(hours=HOURS_PER_DAY - 1)
    if (
        start.to_datetime64() in cube.time.values
        and end.to_datetime64() in cube.time.values
    ):
        valid_days.append(d)

print("Days:", len(valid_days))


# ============================================================
# TRAIN / VAL SPLIT
#
# Single split, not cross-fold -- these weights are meant to be
# loaded as pretraining into the joint pipeline (model.stage1[c]),
# not to produce out-of-fold corrections. It MUST match
# date_ranges in configs.json exactly: if this pretraining's val
# days overlapped with days the joint pipeline also uses as val,
# the joint model would get warm-started on days it's later
# "validated" on -- an information leak that quietly inflates the
# joint pipeline's first validation numbers.
# ============================================================

TRAIN_RANGE = ("2025-01-02", "2025-10-31")
VAL_RANGE = ("2025-11-01", "2025-12-30")


def make_windows(day_list):
    """One window per day (00:00 -> 23:00), matching the main pipeline's
    full-day Stage 1 input."""
    return list(day_list)


# ============================================================
# DATASET
#
# Returns, per window (one full day):
#   weather          (24,7,104,225) float32
#   residual_norm    (96,) float32   -- training target
#   forecast_raw     (96,) float32   -- baseline, real units
#   actual_raw       (96,) float32   -- ground truth, real units
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
        end = start + pd.Timedelta(hours=HOURS_PER_DAY - 1)

        weather = self.cube.sel(time=slice(start, end)).values
        weather = torch.tensor(weather, dtype=torch.float32)

        mask = (
            (self.residual_norm.index >= start)
            & (self.residual_norm.index < start + pd.Timedelta(hours=HOURS_PER_DAY))
        )

        residual_slice = self.residual_norm.loc[mask]
        forecast_slice = self.forecast.loc[mask]
        actual_slice = self.actual.loc[mask]

        assert len(residual_slice) == SLOTS_PER_DAY, (
            f"{start.date()}: expected {SLOTS_PER_DAY} slots, got {len(residual_slice)}"
        )

        residual_norm = torch.tensor(residual_slice.values, dtype=torch.float32)
        forecast_raw = torch.tensor(forecast_slice.values, dtype=torch.float32)
        actual_raw = torch.tensor(actual_slice.values, dtype=torch.float32)

        return weather, residual_norm, forecast_raw, actual_raw


# ============================================================
# LOSS
#
# NOTE: this same TV (smoothness) penalty is applied regardless of
# commodity. Solar's diurnal curve and wind are both fairly smooth,
# but hydro can have sharp ramps (gate operations, scheduled
# releases) -- penalizing variation may actively fight real hydro
# behavior. If hydro validation R2 looks worse than baseline or
# predictions look over-smoothed, consider lowering/zeroing
# TV_WEIGHT specifically for the hydro run.
# ============================================================

TV_WEIGHT = 1e-3


def loss_function(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    tv_t = torch.abs(pred[:, 1:] - pred[:, :-1]).mean()
    return mse + TV_WEIGHT * tv_t


# ============================================================
# R^2 HELPER
# ============================================================

def r2_score(actual, pred):
    actual = actual.numpy() if torch.is_tensor(actual) else np.asarray(actual)
    pred = pred.numpy() if torch.is_tensor(pred) else np.asarray(pred)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ============================================================
# TRAIN + OUT-OF-FOLD INFERENCE, ONE FOLD AT A TIME
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS = 3

PARAMS_DIR.mkdir(parents=True, exist_ok=True)

train_start, train_end = pd.Timestamp(TRAIN_RANGE[0]), pd.Timestamp(TRAIN_RANGE[1])
val_start, val_end = pd.Timestamp(VAL_RANGE[0]), pd.Timestamp(VAL_RANGE[1])

train_days = [d for d in valid_days if train_start <= d <= train_end]
val_days = [d for d in valid_days if val_start <= d <= val_end]

print(f"\n=== [{COMMODITY}] pretraining  (train {train_start.date()} -> {train_end.date()}, "
      f"val {val_start.date()} -> {val_end.date()}) ===")
print("Train days:", len(train_days), "Val days:", len(val_days))

train_windows = make_windows(train_days)
val_windows = make_windows(val_days)

# --------------------------------------------------------
# Residual normalization stats fit on TRAIN days only -- same
# convention as fit_residual_stats() in common.py, so these
# stats are directly comparable/reusable with the joint pipeline.
# --------------------------------------------------------

train_mask = residual_series.index.normalize().isin(pd.DatetimeIndex(train_days))
RESIDUAL_MEAN = residual_series[train_mask].mean()
RESIDUAL_STD = residual_series[train_mask].std()
residual_series_norm = (residual_series - RESIDUAL_MEAN) / RESIDUAL_STD

train_dataset = ResidualDataset(cube, train_windows, residual_series_norm, forecast_series, actual_series)
val_dataset = ResidualDataset(cube, val_windows, residual_series_norm, forecast_series, actual_series)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = S1NN(STAGE1_CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)

print("Number of parameters:", sum(p.numel() for p in model.parameters()))

for epoch in range(N_EPOCHS):

    model.train()
    total = 0

    for weather, residual_norm, _, _ in train_loader:
        weather = weather.to(device)
        residual_norm = residual_norm.to(device)

        pred = model(weather)
        loss = loss_function(pred, residual_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_loss = total / len(train_loader)

    # ------------------------------------------------------------
    # R^2 over the WHOLE validation set at once (pooled across all
    # windows, not averaged per-batch): baseline forecast vs
    # actual, and reconstructed (forecast + predicted residual)
    # vs actual.
    # ------------------------------------------------------------

    model.eval()

    actual_chunks, forecast_chunks, corrected_chunks = [], [], []

    with torch.no_grad():
        for weather, residual_norm, forecast_raw, actual_raw in val_loader:
            weather = weather.to(device)

            pred_norm = model(weather).cpu()
            pred_real = pred_norm * RESIDUAL_STD + RESIDUAL_MEAN
            corrected_forecast = forecast_raw + pred_real

            actual_chunks.append(actual_raw)
            forecast_chunks.append(forecast_raw)
            corrected_chunks.append(corrected_forecast)

    if actual_chunks:
        actual_all = torch.cat(actual_chunks)
        forecast_all = torch.cat(forecast_chunks)
        corrected_all = torch.cat(corrected_chunks)

        r2_baseline = r2_score(actual_all, forecast_all)
        r2_model = r2_score(actual_all, corrected_all)
    else:
        r2_baseline = float("nan")
        r2_model = float("nan")

    print(
        f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
        f"R2_baseline(forecast) {r2_baseline:.4f}  "
        f"R2_model(corrected) {r2_model:.4f}"
    )

# --------------------------------------------------------
# Save the pretrained model + its normalization stats. This
# checkpoint's "model_state_dict" is what goes straight into
# configs.json's stage1_pretrained_weights[COMMODITY].
# --------------------------------------------------------

model_path = PARAMS_DIR / f"{COMMODITY}_pretrained.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "residual_mean": RESIDUAL_MEAN,
        "residual_std": RESIDUAL_STD,
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
    },
    model_path,
)
print(f"Saved pretrained model to: {model_path}")