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
DATA_PATH = ROOT / "output/data_preprocessing/clean_datasets/Hydro_Power.csv" 

PARAMS_DIR = ROOT / "process/stage1/parameters"
OOF_CSV_PATH = ROOT / "output/data_preprocessing/clean_datasets/hydro_improved_forecast.csv"


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
    end = d + pd.Timedelta(hours=23)
    if (
        start.to_datetime64() in cube.time.values
        and end.to_datetime64() in cube.time.values
    ):
        valid_days.append(d)

print("Days:", len(valid_days))


# ============================================================
# CROSS-FITTING FOLDS
#
# 4 quarters, each held out once as validation. The train set
# for a fold is every valid day OUTSIDE that quarter (the
# other ~9 months) -- for Q2/Q3 that's two chronological
# chunks (e.g. holding out Apr-Jun trains on Jan-Mar +
# Jul-Dec). Rotating like this means every day of the year
# eventually gets an out-of-fold correction, produced by a
# model that never saw that day's target during its own
# training.
# ============================================================

FOLDS = [
    {"name": "Q1", "val_start": pd.Timestamp("2025-01-02"), "val_end": pd.Timestamp("2025-03-31")},
    {"name": "Q2", "val_start": pd.Timestamp("2025-04-01"), "val_end": pd.Timestamp("2025-06-30")},
    {"name": "Q3", "val_start": pd.Timestamp("2025-07-01"), "val_end": pd.Timestamp("2025-09-30")},
    {"name": "Q4", "val_start": pd.Timestamp("2025-10-01"), "val_end": pd.Timestamp("2025-12-30")},
]


def make_windows(day_list):
    starts = []
    for d in day_list:
        starts.append(d)
        starts.append(d + pd.Timedelta(hours=12))
    return starts


# ============================================================
# DATASET
#
# Returns, per window:
#   weather          (12,7,104,225) float32
#   residual_norm    (48,) float32   -- training target
#   forecast_raw     (48,) float32   -- baseline, real units
#   actual_raw       (48,) float32   -- ground truth, real units
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
# LOSS
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

oof_frames = []

for fold in FOLDS:

    fold_name = fold["name"]
    val_start, val_end = fold["val_start"], fold["val_end"]

    val_days = [d for d in valid_days if val_start <= d <= val_end]
    train_days = [d for d in valid_days if d not in val_days]

    print(f"\n=== Fold {fold_name}  (val {val_start.date()} -> {val_end.date()}) ===")
    print("Train days:", len(train_days), "Val days:", len(val_days))

    train_windows = make_windows(train_days)
    val_windows = make_windows(val_days)

    # --------------------------------------------------------
    # Residual normalization stats fit on THIS FOLD'S training
    # days only -- never on validation days, otherwise the val
    # fold leaks into its own correction stats.
    # --------------------------------------------------------

    train_mask = residual_series.index.normalize().isin(pd.DatetimeIndex(train_days))
    RESIDUAL_MEAN = residual_series[train_mask].mean()
    RESIDUAL_STD = residual_series[train_mask].std()
    residual_series_norm = (residual_series - RESIDUAL_MEAN) / RESIDUAL_STD

    train_dataset = ResidualDataset(cube, train_windows, residual_series_norm, forecast_series, actual_series)
    val_dataset = ResidualDataset(cube, val_windows, residual_series_norm, forecast_series, actual_series)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ResidualNet().to(device)
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
        # R^2 over the WHOLE validation fold at once (pooled across all
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
    # Save this fold's model + its own normalization stats
    # --------------------------------------------------------

    fold_model_path = PARAMS_DIR / f"hydro_model_fold_{fold_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "residual_mean": RESIDUAL_MEAN,
            "residual_std": RESIDUAL_STD,
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
        },
        fold_model_path,
    )
    print(f"Saved fold {fold_name} model to: {fold_model_path}")
