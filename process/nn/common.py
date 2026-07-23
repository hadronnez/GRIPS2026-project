"""
Shared utilities for the Stage1/2/3 (S123NN) battery-dispatch pipeline.

This module holds everything that train.py, evaluate.py, and visualize.py
all need: config loading, path setup, data loading, the Dataset/collate
code, and small numeric helpers. Keeping it in one place means the three
stages can never quietly drift out of sync with each other.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from architecture import Model as S123NN 

# ============================================================
# LOGGING
# ============================================================

def info(msg: str) -> None:
    """Timestamped [INFO] print, flushed immediately so logs stream live
    even when stdout is redirected to a file."""
    print(f"[INFO {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN {datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")
CONFIG_PATH = ROOT / "process" / "nn" / "configs.json"

PARAMS_DIR = ROOT / "output" / "nn" / "parameters"
CHECKPOINT_PATH = PARAMS_DIR / "parameters.pt"

EVAL_DIR = ROOT / "output" / "nn" / "evaluation"
PLOTS_DIR = ROOT / "output" / "nn" / "plots"

BATCH_LOG_EVERY = 20  # print a running-loss line every N training batches; set None to disable


# ============================================================
# CONFIG LOADING
# ============================================================

def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _paths_to_path(ns: SimpleNamespace) -> SimpleNamespace:
    """Convert every string attribute of a flat namespace into a Path."""
    for key, value in vars(ns).items():
        setattr(ns, key, Path(value))
    return ns


def load_config(path: str | Path) -> SimpleNamespace:
    """Load config.json into nested SimpleNamespace objects.

    `commodities` (Stage 1 branches) and `exo_raw_paths` (exogenous channels
    stacked into Stage 2's enc_in) are derived from the keys of `renewables`
    and `boundary_conditions`, so they can never drift out of sync with the
    path dicts they come from.
    """
    import json

    info(f"Loading config from {path}")
    with open(path) as f:
        raw = json.load(f)

    config = _to_namespace(raw)

    # --- data source paths ---
    config.meteodata = Path(raw["meteodata"])
    config.price = Path(raw["price"])
    config.renewables = _paths_to_path(config.renewables)
    config.boundary_conditions = _paths_to_path(config.boundary_conditions)

    # --- commodities / exo channels, derived from the path dicts above ---
    config.commodities = list(vars(config.renewables).keys())
    config.exo_raw_paths = list(vars(config.boundary_conditions).keys())
    info(f"Commodities ({len(config.commodities)}): {config.commodities}")
    info(f"Exogenous channels ({len(config.exo_raw_paths)}): {config.exo_raw_paths}")

    # --- derived fields, computed once so they can't drift out of sync ---
    config.hours_per_day = config.slots_per_day // 4

    config.stage2.enc_in = len(config.commodities) + len(config.exo_raw_paths)
    config.stage2.seq_len = config.slots_per_day
    config.stage2.pred_len = config.slots_per_day

    config.stage3.slots_per_day = config.slots_per_day

    # `stage1_pretrained_weights` is used as a dict (commodity -> path) in
    # main(); keep it a plain dict rather than letting _to_namespace turn it
    # into a SimpleNamespace (which has no .items()).
    config.training.stage1_pretrained_weights = dict(raw["training"]["stage1_pretrained_weights"])
    config.training.stage2_pretrained_weights = raw["training"]["stage2_pretrained_weights"]

    info(
        f"Config loaded: slots_per_day={config.slots_per_day}, "
        f"stage2.enc_in={config.stage2.enc_in}, "
        f"train_range={config.date_ranges.train}, val_range={config.date_ranges.val}"
    )
    return config


config = load_config(CONFIG_PATH)

SLOTS_PER_DAY = config.slots_per_day
HOURS_PER_DAY = config.hours_per_day
COMMODITIES = config.commodities
EXO_RAW_PATHS = config.exo_raw_paths
TRAIN_RANGE = config.date_ranges.train
VAL_RANGE = config.date_ranges.val
COL_CFG = config.columns

BATCH_SIZE = config.training.batch_size
N_EPOCHS = config.training.epochs
LR = config.training.lr
WEIGHT_DECAY = config.training.weight_decay
T0 = config.training.T0
T_MIN = config.training.T_min
LAMBDA_START = config.training.lambda_start
LAMBDA_END = config.training.lambda_end
STAGE1_PRETRAINED = config.training.stage1_pretrained_weights
STAGE2_PRETRAINED = config.training.stage2_pretrained_weights


# ============================================================
# DATA LOADING HELPERS
# ============================================================

def wide_to_15min(df: pd.DataFrame, sub_cols: list[str]) -> pd.Series:
    """Turn hourly rows with N sub-columns (e.g. *_00/_15/_30/_45) into one
    long Series at 15-min resolution."""
    step = pd.Timedelta(hours=1) / len(sub_cols)
    parts = []
    for i, col in enumerate(sub_cols):
        s = df[col].copy()
        s.index = s.index + i * step
        parts.append(s)
    return pd.concat(parts).sort_index()


def load_actual_forecast(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path, parse_dates=[COL_CFG.timestamp]).set_index(COL_CFG.timestamp).sort_index()
    forecast = wide_to_15min(df, COL_CFG.forecast)
    actual = wide_to_15min(df, COL_CFG.actual)
    return forecast, actual


def load_forecast_only(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[COL_CFG.timestamp]).set_index(COL_CFG.timestamp).sort_index()
    return wide_to_15min(df, COL_CFG.forecast)


def build_valid_days(cube: xr.DataArray) -> list[pd.Timestamp]:
    """Calendar days for which the shared NWP cube has full HOURS_PER_DAY coverage."""
    times = pd.DatetimeIndex(cube["time"].values)
    candidate_days = sorted(set(times.normalize()))
    valid = []
    for day in candidate_days:
        day = pd.Timestamp(day)
        end = day + pd.Timedelta(hours=HOURS_PER_DAY - 1)
        if ((times >= day) & (times <= end)).sum() == HOURS_PER_DAY:
            valid.append(day)
    n_dropped = len(candidate_days) - len(valid)
    info(f"Valid days with full {HOURS_PER_DAY}h NWP coverage: {len(valid)}/{len(candidate_days)} "
         f"({n_dropped} dropped for incomplete coverage)")
    return valid


def fit_residual_stats(actual: pd.Series, forecast: pd.Series, days: list[pd.Timestamp]) -> tuple[float, float]:
    residuals = []
    skipped = 0
    for day in days:
        end = day + pd.Timedelta(days=1)
        mask = (actual.index >= day) & (actual.index < end)
        a = actual.loc[mask].to_numpy()
        f = forecast.loc[mask].to_numpy()
        if len(a) != SLOTS_PER_DAY or len(f) != SLOTS_PER_DAY:
            skipped += 1
            continue
        residuals.append(a - f)
    if skipped:
        warn(f"fit_residual_stats: skipped {skipped}/{len(days)} days with incomplete 15-min coverage")
    residuals = np.concatenate(residuals)
    return float(residuals.mean()), float(residuals.std())


def temperature_schedule(epoch: int, n_epochs: int, t0: float, t_min: float) -> float:
    """Exponential decay from t0 down to t_min over n_epochs."""
    frac = epoch / max(n_epochs - 1, 1)
    return t0 * (t_min / t0) ** frac


# ============================================================
# DATASET
# ============================================================

class nnDataset(Dataset):
    """One sample = one calendar day, aligned across every stage.

    Per day `d` returns:
      weather[c]        (24, 7, 104, 225) float32    -- shared hourly NWP cube (same input, per commodity)
      forecast_day[c]   (96,) float32                -- commodity c's forecast production, 15-min
      actual_day[c]     (96,) float32                -- commodity c's actual production, 15-min
      exo_day[name]     (96,) float32                -- exogenous forecast production, 15-min
      price_actual_day  (96,) float32                -- actual price, 15-min
      date              pd.Timestamp (day start, 00:00)
    """

    def __init__(self, days, cube, forecast, actual, exo, price_actual):
        self.days = days
        self.cube = cube
        self.forecast = forecast
        self.actual = actual
        self.exo = exo
        self.price_actual = price_actual

    def __len__(self):
        return len(self.days)

    @staticmethod
    def _slice_96(series: pd.Series, day: pd.Timestamp) -> torch.Tensor:
        mask = (series.index >= day) & (series.index < day + pd.Timedelta(days=1))
        vals = series.loc[mask].to_numpy()
        assert len(vals) == SLOTS_PER_DAY, f"{day.date()}: expected {SLOTS_PER_DAY} slots, got {len(vals)}"
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx):
        day = self.days[idx]
        end = day + pd.Timedelta(hours=HOURS_PER_DAY - 1)

        w = self.cube.sel(time=slice(day, end)).values
        weather_tensor = torch.tensor(w, dtype=torch.float32)  # (24, 7, 104, 225)

        forecast_day, actual_day = {}, {}
        weather = {}
        for c in COMMODITIES:
            weather[c] = weather_tensor  # same shared cube fed to every commodity's Stage 1 branch
            forecast_day[c] = self._slice_96(self.forecast[c], day)
            actual_day[c] = self._slice_96(self.actual[c], day)

        exo_day = {name: self._slice_96(series, day) for name, series in self.exo.items()}
        price_actual_day = self._slice_96(self.price_actual, day)

        return weather, forecast_day, actual_day, exo_day, price_actual_day, day


def collate_days(batch):
    weathers, forecasts, actuals, exos, prices, dates = zip(*batch)

    weather = {c: torch.stack([w[c] for w in weathers]) for c in weathers[0]}
    forecast_day = {c: torch.stack([f[c] for f in forecasts]) for c in forecasts[0]}
    actual_day = {c: torch.stack([a[c] for a in actuals]) for c in actuals[0]}
    exo_day = {n: torch.stack([e[n] for e in exos]) for n in exos[0]}
    price_actual_day = torch.stack(prices)

    return weather, forecast_day, actual_day, exo_day, price_actual_day, list(dates)


def load_everything():
    t_start = time.time()
    info(f"Opening meteodata cube from {config.meteodata}")
    ds = xr.open_dataset(config.meteodata)
    cube = ds["cube"]
    info(f"Cube loaded: dims={dict(cube.sizes)}")

    forecast, actual = {}, {}
    for c in COMMODITIES:
        info(f"Loading renewable series [{c}] from {getattr(config.renewables, c)}")
        f, a = load_actual_forecast(getattr(config.renewables, c))
        forecast[c], actual[c] = f, a
        info(f"  [{c}] forecast: {len(f)} rows ({f.index.min()} -> {f.index.max()})")
        info(f"  [{c}] actual:   {len(a)} rows ({a.index.min()} -> {a.index.max()})")

    exo = {}
    for name in EXO_RAW_PATHS:
        info(f"Loading exogenous channel [{name}] from {getattr(config.boundary_conditions, name)}")
        exo[name] = load_forecast_only(getattr(config.boundary_conditions, name))
        info(f"  [{name}]: {len(exo[name])} rows")

    info(f"Loading price series from {config.price}")
    price_df = pd.read_csv(config.price, parse_dates=[COL_CFG.timestamp]).set_index(COL_CFG.timestamp).sort_index()
    price_actual = wide_to_15min(price_df, COL_CFG.actual)
    info(f"Price series: {len(price_actual)} rows ({price_actual.index.min()} -> {price_actual.index.max()})")

    info(f"All data loaded in {time.time() - t_start:.1f}s")
    return cube, forecast, actual, exo, price_actual


def build_datasets():
    """Load raw data and build train/val nnDataset objects plus per-commodity
    residual (mean, std) stats fit on the training days only. Shared by
    train.py and evaluate.py so both scripts see identical splits."""
    cube, forecast, actual, exo, price_actual = load_everything()

    valid_days = build_valid_days(cube)
    train_days = [d for d in valid_days if pd.Timestamp(TRAIN_RANGE[0]) <= d <= pd.Timestamp(TRAIN_RANGE[1])]
    val_days = [d for d in valid_days if pd.Timestamp(VAL_RANGE[0]) <= d <= pd.Timestamp(VAL_RANGE[1])]
    info(f"Train days: {len(train_days)}  ({TRAIN_RANGE[0]} -> {TRAIN_RANGE[1]})")
    info(f"Val days:   {len(val_days)}  ({VAL_RANGE[0]} -> {VAL_RANGE[1]})")
    if not train_days:
        warn("No training days matched TRAIN_RANGE — check date_ranges in configs.json")
    if not val_days:
        warn("No validation days matched VAL_RANGE — check date_ranges in configs.json")

    residual_stats = {c: fit_residual_stats(actual[c], forecast[c], train_days) for c in COMMODITIES}
    for c, (m, s) in residual_stats.items():
        info(f"Residual stats [{c}]: mean={m:.4f} std={s:.4f}")

    train_ds = nnDataset(train_days, cube, forecast, actual, exo, price_actual)
    val_ds = nnDataset(val_days, cube, forecast, actual, exo, price_actual)

    return train_ds, val_ds, residual_stats


def load_model(residual_stats, device, checkpoint_path: Path | None = None):
    """Build the S123NN model, load the given (or default best) checkpoint,
    and move it to `device` in eval mode. Shared by evaluate.py; train.py
    builds a fresh model itself since it also handles Stage1/2 warm-starts."""
    model = S123NN(config, residual_stats).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"Model built with {n_params:,} trainable parameters")

    ckpt_path = checkpoint_path or CHECKPOINT_PATH
    info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    info(
        f"Checkpoint loaded (epoch={ckpt.get('epoch')}, "
        f"val_hard_revenue={ckpt.get('val_hard_revenue')}, "
        f"val_oracle_revenue={ckpt.get('val_oracle_revenue')})"
    )
    model.eval()
    return model, ckpt
