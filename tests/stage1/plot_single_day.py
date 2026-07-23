from types import SimpleNamespace
import torch
from torch.utils.data import Dataset

import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
from model.model import Model

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project"
)

METEO_PATH = ROOT / "output/data_preprocessing/clean_datasets/clean_nwp_dataset.nc"
DATA_PATH = ROOT / "output/data_preprocessing/clean_datasets/Photovoltaic.csv"

PARAMS_DIR = ROOT / "tests/stage1/parameters"

RESULTS_DIR = ROOT / "tests/stage1_output"
RESIDUAL_PLOT_PATH = RESULTS_DIR / "solar_residual_single_day.png"
PRODUCTION_PLOT_PATH = RESULTS_DIR / "solar_production_single_day.png"

# Which day to plot. Must be a valid day (full 24h NWP coverage) --
# change this to look at a different day.
TARGET_DAY = pd.Timestamp("2025-06-15")


STAGE1_CONFIG = SimpleNamespace(
    in_channels=7,
    encoder_channels=[16, 32],
    encoder_norm_groups=[4, 8],
    encoder_kernel_size=3,
    encoder_padding_mode="replicate",
    num_temporal_upsamples=2,
    upsample_kernel_size=2,
    upsample_stride=2,
    head_hidden_channels=16,
    head_kernel_size=3,
    head_norm_groups=4,
    out_channels=1,
)
 
 
# ============================================================
# CROSS-FITTING FOLDS
#
# Must match the FOLDS definition used in the training script.
# TARGET_DAY's own fold is picked automatically below so the
# checkpoint used is always genuinely out-of-fold for that day.
# ============================================================
 
FOLDS = [
    {"name": "Q1", "val_start": pd.Timestamp("2025-01-01"), "val_end": pd.Timestamp("2025-03-31")},
    {"name": "Q2", "val_start": pd.Timestamp("2025-04-01"), "val_end": pd.Timestamp("2025-06-30")},
    {"name": "Q3", "val_start": pd.Timestamp("2025-07-01"), "val_end": pd.Timestamp("2025-09-30")},
    {"name": "Q4", "val_start": pd.Timestamp("2025-10-01"), "val_end": pd.Timestamp("2025-12-30")},
]
 
 
def fold_for_day(day: pd.Timestamp) -> str:
    for fold in FOLDS:
        if fold["val_start"] <= day <= fold["val_end"]:
            return fold["name"]
    raise ValueError(f"{day.date()} doesn't fall in any fold's val range")
 
 
# ============================================================
# LOAD FORECAST + ACTUAL
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
 
 
# ============================================================
# METEO LAZY LOAD
# ============================================================
 
ds = xr.open_dataset(METEO_PATH)
cube = ds["cube"]
 
 
# ============================================================
# VALIDITY CHECK FOR TARGET_DAY
# ============================================================
 
start_check = TARGET_DAY
end_check = TARGET_DAY + pd.Timedelta(hours=23)
if not (
    start_check.to_datetime64() in cube.time.values
    and end_check.to_datetime64() in cube.time.values
):
    raise ValueError(f"{TARGET_DAY.date()} doesn't have full 24h NWP coverage in the cube")
 
 
def make_windows(day):
    """The two 12h windows (00:00 and 12:00) that together cover one full day."""
    return [day, day + pd.Timedelta(hours=12)]
 
 
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
# LOAD THE ONE FOLD THAT HAS TARGET_DAY OUT-OF-FOLD, AND RUN IT
# ============================================================
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
fold_name = fold_for_day(TARGET_DAY)
fold_model_path = PARAMS_DIR / f"solar_model_fold_{fold_name}.pt"
checkpoint = torch.load(fold_model_path, map_location=device, weights_only=False)
 
RESIDUAL_MEAN = checkpoint["residual_mean"]
RESIDUAL_STD = checkpoint["residual_std"]
residual_series_norm = (residual_series - RESIDUAL_MEAN) / RESIDUAL_STD
 
windows = make_windows(TARGET_DAY)
eval_dataset = ResidualDataset(cube, windows, residual_series_norm, forecast_series, actual_series)
 
def remap_legacy_temporal_keys(state_dict):
    """Old checkpoints (trained before temporal_up1/temporal_up2 were merged
    into a temporal_ups ModuleList) use different key names for the same
    weights. Same tensors, just renamed -- remap instead of retraining."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("temporal_up1."):
            key = key.replace("temporal_up1.", "temporal_ups.0.")
        elif key.startswith("temporal_up2."):
            key = key.replace("temporal_up2.", "temporal_ups.1.")
        remapped[key] = value
    return remapped
 
 
model = Model(STAGE1_CONFIG).to(device)
model.load_state_dict(remap_legacy_temporal_keys(checkpoint["model_state_dict"]))
model.eval()
 
print(f"Target day {TARGET_DAY.date()} -> fold {fold_name} (out-of-fold), "
      f"checkpoint: {fold_model_path}")
 
times, actual_vals, forecast_vals, improved_vals, pred_residual_vals = [], [], [], [], []
 
with torch.no_grad():
    for idx in range(len(eval_dataset)):
        weather, _, forecast_raw, actual_raw = eval_dataset[idx]
        start = windows[idx]
 
        window_times = pd.date_range(start=start, periods=48, freq="15min")
 
        pred_norm = model(weather.unsqueeze(0).to(device)).squeeze(0).cpu()
        pred_real = pred_norm * RESIDUAL_STD + RESIDUAL_MEAN
        improved = forecast_raw + pred_real
 
        times.extend(window_times)
        actual_vals.extend(actual_raw.numpy())
        forecast_vals.extend(forecast_raw.numpy())
        improved_vals.extend(improved.numpy())
        pred_residual_vals.extend(pred_real.numpy())
 
day_df = pd.DataFrame({
    "time": times,
    "actual": actual_vals,
    "forecast": forecast_vals,
    "improved_forecast": improved_vals,
    "predicted_residual": pred_residual_vals,
}).sort_values("time").reset_index(drop=True)
 
day_df["residual"] = day_df["actual"] - day_df["forecast"]
 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
 
 

# ============================================================
# PLOT 1: residual time series for the day (actual - forecast)
# ============================================================
 
plt.figure(figsize=(8, 8))
plt.plot(day_df["time"], day_df["predicted_residual"], label="Predicted residual", linestyle="-", linewidth=3)
plt.plot(day_df["time"], day_df["residual"], label="Actual residual", linestyle="--", linewidth=3)
plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Residual (power)")
plt.title(f"Residual -- {TARGET_DAY.date()} (fold {fold_name})")
plt.tight_layout()
plt.savefig(RESIDUAL_PLOT_PATH, dpi=600)
plt.show()
 
 
# ============================================================
# PLOT 2: production for the day -- improved (solid) vs forecast (dashed)
# ============================================================
 
plt.figure(figsize=(8, 8))
plt.plot(day_df["time"], day_df["improved_forecast"], label="Predicted (improved) production", linestyle="-", linewidth=3)
plt.plot(day_df["time"], day_df["actual"], label="Actual production", linestyle="--", linewidth=3)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Power")
plt.title(f"Production -- {TARGET_DAY.date()} (fold {fold_name})")
plt.tight_layout()
plt.savefig(PRODUCTION_PLOT_PATH, dpi=600)
plt.show()
 
print(f"Wrote residual plot to: {RESIDUAL_PLOT_PATH}")
print(f"Wrote production plot to: {PRODUCTION_PLOT_PATH}")
