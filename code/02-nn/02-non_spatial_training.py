
from pathlib import Path

import xarray as xr
import pandas as pd

import numpy as np

import torch
from torch.utils.data import DataLoader
from dataset_creator import Spatial_dataset
import model 


ROOT =                  Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")
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
    timestamp     =    "time",
    actual        =    ["actual_00", "actual_15", "actual_30", "actual_45"],
)

# Charge the datasets
meteo           =      xr.open_dataset(METEO_PATH["meteodata"], chunks={"time": 24})    # (N, 4)
solar           =      pd.read_csv(TARGET_PATHS["solar"])[COLS_TARGET["actual"]]        # (N, 4)
wind            =      pd.read_csv(TARGET_PATHS["wind"])[COLS_TARGET["actual"]]         # (N, 4)
hydro           =      pd.read_csv(TARGET_PATHS["hydro"])[COLS_TARGET["actual"]]        # (N, 4)


ds = Spatial_dataset(
    meteo,
    solar,
    wind,
    hydro
)

loader = DataLoader(
    ds,
    batch_size = 32,
    shuffle = False,
)