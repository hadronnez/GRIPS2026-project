
from pathlib import Path

import pandas as pd
import xarray as xr

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from model import EnergyPriceModel, compute_loss, SPATIAL_TYPES, NONSPATIAL_TYPES, ALL_ENERGY_TYPES


#
#   PATHS AND CONFIGURATIONS
#

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

COLS_TARGET = dict(
    timestamp="time",
    actual=["actual_00", "actual_15", "actual_30", "actual_45"],
    forecast=["forecast_00", "forecast_15", "forecast_30", "forecast_45"],
)


TRAIN_RANGE = ("2025-01-02", "2025-10-31")
VAL_RANGE = ("2025-11-01", "2025-12-30")

SEQ_LEN = 24       
STRIDE = 24         
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6

#
# DEFINITIONS
#


def meteodata_to_tensor(path: str):
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    da = ds[var].transpose("time", "channel", "lat", "lon")
    time = pd.DatetimeIndex(da["time"].values)
    return time, torch.from_numpy(da.values.astype(np.float32))  # (T,C,H,W)

def target_to_tensor(path: str):
    df = pd.read_csv(path)
    forecast = torch.from_numpy(df[COLS_TARGET["forecast"]].to_numpy(dtype=np.float32))  # (T,4)
    t = pd.DatetimeIndex(df["time"])
    return t, forecast


class MeteoPriceDataset(Dataset):
    def __init__(self, meteo_paths: dict, target_paths: dict, transform=None):
        assert meteo_paths.keys() == target_paths.keys()
        self.cubes, self.targets, self.index = {}, {}, []

        for key in meteo_paths:
            t_meteo, cube = meteodata_to_tensor(meteo_paths[key])
            t_target, target = target_to_tensor(target_paths[key])


            self.cubes[key] = cube[im]
            self.targets[key] = target[it]
            self.index.extend((key, i) for i in range(len(common)))

        self.transform = transform  # e.g. per-split normalization stats applied here

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, t = self.index[idx]
        x = self.cubes[key][t]
        if self.transform is not None:
            x = self.transform(x, key)
        return x, self.targets[key][t]
