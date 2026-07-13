
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import dataclasses
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

ROOT = Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")

METEO_PATH = {
    "meteodata":        ROOT / "output/raw_datasets/raw_nwp_dataset.nc"

}
TARGET_PATHS = {
    "solar":            ROOT / "output/raw_datasets/Photovoltaic.csv",
    "wind":             ROOT / "output/raw_datasets/Wind_Power.csv",
    "hydro":            ROOT / "output/raw_datasets/Hydro_Power.csv",
    "non_marketized":   ROOT / "output/raw_datasets/Non_Marketized_Unit.csv",
    "tie_line":         ROOT / "output/raw_datasets/Tie_Line.csv",
    "system_load":      ROOT / "output/raw_datasets/System_Load.csv",
    "price":            ROOT / "output/raw_datasets/Price.csv"
}

TRAIN_RANGE = ("2025-01-02", "2025-10-31")
VAL_RANGE = ("2025-11-01", "2025-12-30")

SEQ_LEN = 24       
STRIDE = 24         
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COL_CFG = dict(
    timestamp="datetime",
    actual=["actual_00", "actual_15", "actual_30", "actual_45"],
    forecast=["forecast_00", "forecast_15", "forecast_30", "forecast_45"],
)

EPS = 1e-6


@dataclasses.dataclass
class RawData:
    times: pd.DatetimeIndex
    meteo: dict[str, np.ndarray]          # name -> (T,C,H,W)
    actual: dict[str, np.ndarray]         # name -> (T,4)  (ALL_ENERGY_TYPES)
    forecast: dict[str, np.ndarray]       # name -> (T,4)  (ALL_ENERGY_TYPES)
    price_actual: np.ndarray              # (T,4)
    price_forecast: np.ndarray            # (T,4)


def _load_nc(path: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    da = ds[var].transpose("time", "channel", "lat", "lon")
    times = pd.DatetimeIndex(da["time"].values)
    return times, da.values.astype(np.float32)


def _load_csv(path: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, parse_dates=[COL_CFG["timestamp"]])
    df = df.sort_values(COL_CFG["timestamp"])
    times = pd.DatetimeIndex(df[COL_CFG["timestamp"]])
    actual = df[COL_CFG["actual"]].to_numpy(dtype=np.float32)
    forecast = df[COL_CFG["forecast"]].to_numpy(dtype=np.float32)
    return times, actual, forecast


def load_raw_arrays(meteo_paths: dict[str, str], csv_paths: dict[str, str]) -> RawData:
    # un único cubo meteo (channel,lat,lon), compartido por solar/wind/hydro
    t_meteo, arr_meteo = _load_nc(meteo_paths["meteodata"])
    meteo_raw = {name: arr_meteo for name in SPATIAL_TYPES}
    meteo_times = {name: t_meteo for name in SPATIAL_TYPES}

    csv_raw, csv_times = {}, {}
    for name in ALL_ENERGY_TYPES + ["price"]:
        t, actual, forecast = _load_csv(csv_paths[name])
        csv_raw[name] = (actual, forecast)
        csv_times[name] = t

    common = None
    for t in list(meteo_times.values()) + list(csv_times.values()):
        common = t if common is None else common.intersection(t)
    common = common.sort_values()
    if len(common) == 0:
        raise ValueError("No hay timestamps comunes entre meteo/csvs/precio.")

    def reindex(times, arr):
        idx = pd.Series(np.arange(len(times)), index=times)
        pos = idx.loc[common].to_numpy()
        return arr[pos]

    meteo = {name: reindex(meteo_times[name], meteo_raw[name]) for name in SPATIAL_TYPES}
    actual = {name: reindex(csv_times[name], csv_raw[name][0]) for name in ALL_ENERGY_TYPES}
    forecast = {name: reindex(csv_times[name], csv_raw[name][1]) for name in ALL_ENERGY_TYPES}
    price_actual = reindex(csv_times["price"], csv_raw["price"][0])
    price_forecast = reindex(csv_times["price"], csv_raw["price"][1])

    H = {meteo[name].shape[-2] for name in SPATIAL_TYPES}
    W = {meteo[name].shape[-1] for name in SPATIAL_TYPES}
    assert len(H) == 1 and len(W) == 1, "lat/lon deben coincidir entre solar/wind/hydro"

    return RawData(common, meteo, actual, forecast, price_actual, price_forecast)


def split_raw(raw: RawData, start: str, end: str) -> RawData:
    mask = (raw.times >= pd.Timestamp(start)) & (raw.times < pd.Timestamp(end))
    idx = np.where(mask)[0]
    return RawData(
        times=raw.times[idx],
        meteo={k: v[idx] for k, v in raw.meteo.items()},
        actual={k: v[idx] for k, v in raw.actual.items()},
        forecast={k: v[idx] for k, v in raw.forecast.items()},
        price_actual=raw.price_actual[idx],
        price_forecast=raw.price_forecast[idx],
    )


@dataclasses.dataclass
class Stats:
    meteo: dict[str, tuple[np.ndarray, np.ndarray]]      # name -> (mean(C,1,1), std(C,1,1))
    actual: dict[str, tuple[np.ndarray, np.ndarray]]     # name -> (mean(4,), std(4,))
    forecast: dict[str, tuple[np.ndarray, np.ndarray]]
    price_actual: tuple[np.ndarray, np.ndarray]
    price_forecast: tuple[np.ndarray, np.ndarray]


def compute_stats(raw_train: RawData) -> Stats:
    def mstd(arr, axis):
        m = arr.mean(axis=axis, keepdims=False)
        s = arr.std(axis=axis, keepdims=False) + EPS
        return m.astype(np.float32), s.astype(np.float32)

    meteo = {}
    for name, arr in raw_train.meteo.items():
        m, s = mstd(arr, axis=(0, 2, 3))          # per canal
        meteo[name] = (m.reshape(-1, 1, 1), s.reshape(-1, 1, 1))

    actual = {name: mstd(arr, axis=0) for name, arr in raw_train.actual.items()}
    forecast = {name: mstd(arr, axis=0) for name, arr in raw_train.forecast.items()}
    price_actual = mstd(raw_train.price_actual, axis=0)
    price_forecast = mstd(raw_train.price_forecast, axis=0)

    return Stats(meteo, actual, forecast, price_actual, price_forecast)


def _calendar_features(times: pd.DatetimeIndex) -> np.ndarray:
    hour = times.hour.to_numpy(dtype=np.float32)
    doy = times.dayofyear.to_numpy(dtype=np.float32)
    return np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ], axis=-1).astype(np.float32)          # (T,4)


class EnergyDataset(Dataset):
    """Cada item es una secuencia de longitud seq_len lista para
    EnergyPriceModel.forward(meteo, nonspatial_features, boundary_conditions).
    """

    def __init__(self, raw: RawData, stats: Stats, seq_len: int = 24, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride
        self.n_boundary_features = len(ALL_ENERGY_TYPES) + 1  # forecast prod (x5) + forecast precio

        self.meteo = {
            name: torch.from_numpy((raw.meteo[name] - stats.meteo[name][0]) / stats.meteo[name][1])
            for name in SPATIAL_TYPES
        }

        self.actual_norm = {
            name: torch.from_numpy((raw.actual[name] - stats.actual[name][0]) / stats.actual[name][1])
            for name in ALL_ENERGY_TYPES
        }

        forecast_norm = {
            name: (raw.forecast[name] - stats.forecast[name][0]) / stats.forecast[name][1]
            for name in ALL_ENERGY_TYPES
        }
        price_forecast_norm = (raw.price_forecast - stats.price_forecast[0]) / stats.price_forecast[1]
        price_actual_norm = (raw.price_actual - stats.price_actual[0]) / stats.price_actual[1]
        self.price_actual_norm = torch.from_numpy(price_actual_norm)

        cal = _calendar_features(raw.times) 
        self.nonspatial_features = {
            name: torch.from_numpy(np.concatenate([forecast_norm[name], cal], axis=-1))
            for name in NONSPATIAL_TYPES
        }

        T = len(raw.times)
        parts = [forecast_norm[name][:, :, None] for name in ALL_ENERGY_TYPES]  
        parts.append(price_forecast_norm[:, :, None])
        self.boundary_conditions = torch.from_numpy(np.concatenate(parts, axis=-1))  

        self.T = T

    def __len__(self):
        return max(0, (self.T - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx):
        i0 = idx * self.stride
        sl = slice(i0, i0 + self.seq_len)

        meteo = {name: self.meteo[name][sl] for name in SPATIAL_TYPES}
        nonspatial = {name: self.nonspatial_features[name][sl] for name in NONSPATIAL_TYPES}
        boundary = self.boundary_conditions[sl]
        productions_target = {name: self.actual_norm[name][sl] for name in ALL_ENERGY_TYPES}
        price_target = self.price_actual_norm[sl].reshape(-1)  
        return {
            "meteo": meteo,
            "nonspatial_features": nonspatial,
            "boundary_conditions": boundary,
            "productions_target": productions_target,
            "price_target": price_target,
        }


def collate(batch: list[dict]) -> dict:
    def stack_dict(key):
        keys = batch[0][key].keys()
        return {k: torch.stack([b[key][k] for b in batch]) for k in keys}

    return {
        "meteo": stack_dict("meteo"),
        "nonspatial_features": stack_dict("nonspatial_features"),
        "boundary_conditions": torch.stack([b["boundary_conditions"] for b in batch]),
        "productions_target": stack_dict("productions_target"),
        "price_target": torch.stack([b["price_target"] for b in batch]),
    }


def build_dataloaders():
    raw = load_raw_arrays(METEO_PATH, TARGET_PATHS)

    raw_train = split_raw(raw, *TRAIN_RANGE)
    raw_val = split_raw(raw, *VAL_RANGE)

    stats = compute_stats(raw_train)  # fit SOLO en train

    ds_train = EnergyDataset(raw_train, stats, seq_len=SEQ_LEN, stride=STRIDE)
    ds_val = EnergyDataset(raw_val, stats, seq_len=SEQ_LEN, stride=SEQ_LEN)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # canales de meteo por tipo espacial (para instanciar el modelo)
    spatial_in_channels = {name: raw.meteo[name].shape[1] for name in SPATIAL_TYPES}
    H, W = raw.meteo["solar"].shape[-2], raw.meteo["solar"].shape[-1]
    nonspatial_in_features = {name: ds_train.nonspatial_features[name].shape[-1] for name in NONSPATIAL_TYPES}
    n_boundary_features = ds_train.n_boundary_features

    return (dl_train, dl_val,   
            spatial_in_channels, H, W, nonspatial_in_features, n_boundary_features)


def to_device(batch, device):
    def move_dict(d):
        return {k: v.to(device) for k, v in d.items()}
    return {
        "meteo": move_dict(batch["meteo"]),
        "nonspatial_features": move_dict(batch["nonspatial_features"]),
        "boundary_conditions": batch["boundary_conditions"].to(device),
        "productions_target": move_dict(batch["productions_target"]),
        "price_target": batch["price_target"].to(device),
    }


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"total": 0.0, "stage1": 0.0, "stage2": 0.0, "tv": 0.0}
    n = 0
    for batch in loader:
        batch = to_device(batch, DEVICE)
        with torch.set_grad_enabled(train_mode):
            outputs = model(batch["meteo"], batch["nonspatial_features"], batch["boundary_conditions"])
            targets = {"productions": batch["productions_target"], "price": batch["price_target"]}
            losses = compute_loss(outputs, targets)
            if train_mode:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
        bs = batch["price_target"].shape[0]
        for k in totals:
            totals[k] += losses[k].item() * bs
        n += bs
    return {k: v / n for k, v in totals.items()}


def main():
    (dl_train, dl_val, spatial_in_channels, H, W,
     nonspatial_in_features, n_boundary_features) = build_dataloaders()

    model = EnergyPriceModel(
        spatial_in_channels=spatial_in_channels,
        H=H, W=W,
        nonspatial_in_features=nonspatial_in_features,
        n_boundary_features=n_boundary_features,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_losses = run_epoch(model, dl_train, optimizer)
        val_losses = run_epoch(model, dl_val)
        print(f"[{epoch:03d}] train {train_losses} | val {val_losses}")

        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            torch.save(model.state_dict(), "/mnt/user-data/outputs/best_model.pt")

    model.load_state_dict(torch.load("/mnt/user-data/outputs/best_model.pt"))
    test_losses = run_epoch(model, dl_val)  
    print(f"test {test_losses}")


if __name__ == "__main__":
    main()
