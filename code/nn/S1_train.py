import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import pandas as pd
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project"
)

METEO_PATH = ROOT / "output/clean_datasets/clean_meteodata_dataset.nc"
SOLAR_PATH = ROOT / "output/raw_datasets/Photovoltaic.csv"


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

        times = residual_slice.index

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
# MODEL
#
# Same computation as before, reordered for speed. This is
# NOT an approximation -- it produces mathematically identical
# output, just much cheaper to compute.
#
# WHY THE REORDER IS EXACT:
# temporal_up1/temporal_up2 are per-pixel affine maps (kernel
# size 1 in H,W -- no cross-pixel mixing) with weights shared
# across every spatial location. For any affine f applied
# identically to every pixel, mean_pixels(f(x)) == f(mean_pixels(x)).
# So pooling H,W away right after the encoder, THEN doing the
# temporal upsampling on the pooled (B,32,12) sequence, gives
# the exact same numbers as upsampling first at full spatial
# resolution and pooling afterward.
#
# The old order built and back-propped through a (B,32,48,104,225)
# tensor -- ~36M values per sample -- just to average it away
# at the end. That tensor, and the two ConvTranspose3d layers
# that produced it at full spatial resolution, were the
# overwhelming majority of the compute and memory cost. None
# of it changed the final answer.
#
# Only the encoder genuinely needs full spatial resolution --
# its 3x3 spatial kernel is the only place pixels actually mix
# with their neighbors, which is real information (e.g. local
# cloud patterns) that pooling first would destroy.
# ============================================================


class ResidualNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.weather_encoder = nn.Sequential(
            nn.Conv3d(7, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
        )

        # learn 12h -> 24h, and 24h -> 48 quarter-hours.
        # Now 1D: operates on the pooled (B,32,T) sequence, not
        # on (B,32,T,H,W). Same weights-per-timestep idea as
        # before, at a tiny fraction of the compute.
        self.temporal_up1 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2)
        self.temporal_up2 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2)

        # scalar-per-timestep head, operating on (B, 32, 48)
        self.head = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, weather):
        # incoming: B,T,C,H,W (T=12) -> Conv3D expects B,C,T,H,W
        weather = weather.permute(0, 2, 1, 3, 4)

        x = self.weather_encoder(weather)
        # B,32,12,H,W -- last point where spatial resolution matters

        x = x.mean(dim=(3, 4))
        # B,32,12 -- pool NOW, while the tensor is still small

        x = self.temporal_up1(x)
        x = self.temporal_up2(x)
        # B,32,48 -- upsampling on a tiny sequence, not a spatial grid

        residual_pred = self.head(x).squeeze(1)
        # B,48

        return residual_pred


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

model.eval()

all_times = []
all_actual = []
all_forecast = []
all_improved = []

with torch.no_grad():

    for idx in range(len(val_dataset)):

        weather, _, forecast_raw, actual_raw = val_dataset[idx]

        start = val_windows[idx]

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
    "improved_forecast": all_improved, #This is the improved forecast
})

results.to_csv(
    "validation_forecast.csv",
    index=False
)
plt.figure(figsize=(18,6))

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

plt.savefig("validation_timeseries.png", dpi=300)
plt.show()