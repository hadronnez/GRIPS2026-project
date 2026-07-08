
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

COLS_TARGET = {
    "timestamp": "time",
    "actual": ["actual_00", "actual_15", "actual_30", "actual_45"],
}


def load_and_align(path, time_index):
    """Read a target CSV, align it to the meteo time index, return (N,4)
    actual production values. Raises if alignment introduces NaNs."""
    df = pd.read_csv(path)
    df[COLS_TARGET["timestamp"]] = pd.to_datetime(df[COLS_TARGET["timestamp"]])
    df = df.set_index(COLS_TARGET["timestamp"]).reindex(time_index)
    arr = df[COLS_TARGET["actual"]].to_numpy(np.float32)
    if np.isnan(arr).any():
        raise ValueError(f"{path.name} contains NaNs after alignment to meteo time index")
    return arr  # (N,4)


class Spatial_dataset(Dataset):
    def __init__(self, meteo, solar, wind, hydro, indices=None):
        var = list(meteo.data_vars)[0]
        self.meteo = meteo[var].transpose("time", "channel", "lat", "lon")
        self.indices = np.asarray(indices) if indices is not None else None
        self.solar = torch.as_tensor(solar, dtype=torch.float32)
        self.wind = torch.as_tensor(wind, dtype=torch.float32)
        self.hydro = torch.as_tensor(hydro, dtype=torch.float32)
        self.n = self.meteo.sizes["time"]
        assert len(self.solar) == self.n
        assert len(self.wind) == self.n
        assert len(self.hydro) == self.n

    def __len__(self):
        return len(self.indices) if self.indices is not None else self.n

    def __getitem__(self, i):
        idx = int(self.indices[i]) if self.indices is not None else i
        m = self.meteo.isel(time=idx).values.astype(np.float32)  # (C,H,W)
        m = torch.from_numpy(m).unsqueeze(0)  # (T=1,C,H,W)
        return {
            "meteo": m,
            "targets": {
                "productions": {
                    "solar": self.solar[idx],
                    "wind": self.wind[idx],
                    "hydro": self.hydro[idx],
                }
            }
        }
    
    class Non_spatial_dataset(Dataset):
        pass

    class Price_dataset(Dataset):
        pass