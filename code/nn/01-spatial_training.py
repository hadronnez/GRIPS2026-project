from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_creator import Spatial_dataset, load_and_align
import model


# -------------------------------------------------
# PATHS
# -------------------------------------------------

ROOT = Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")

METEO_PATH = ROOT / "output/raw_datasets/raw_nwp_dataset.nc"

TARGET_PATHS = {
    "solar": ROOT / "output/raw_datasets/Photovoltaic.csv",
    "wind":  ROOT / "output/raw_datasets/Wind_Power.csv",
    "hydro": ROOT / "output/raw_datasets/Hydro_Power.csv",
}

TRAIN_RANGE = ("2025-01-02", "2025-10-31")
VAL_RANGE   = ("2025-11-01", "2025-12-30")


# -------------------------------------------------
# METEO (reference time axis)
# -------------------------------------------------

ds = xr.open_dataset(METEO_PATH, chunks={"time": 24})
time_index = pd.to_datetime(ds["time"].values)


# -------------------------------------------------
# LOAD + ALIGN TARGETS
# -------------------------------------------------

solar = load_and_align(TARGET_PATHS["solar"], time_index)
wind  = load_and_align(TARGET_PATHS["wind"], time_index)
hydro = load_and_align(TARGET_PATHS["hydro"], time_index)


# -------------------------------------------------
# SPLIT
# -------------------------------------------------

train_start, train_end = map(pd.to_datetime, TRAIN_RANGE)
val_start, val_end     = map(pd.to_datetime, VAL_RANGE)

train_mask = (time_index >= train_start) & (time_index <= train_end)
val_mask   = (time_index >= val_start) & (time_index <= val_end)

train_idx = np.where(train_mask)[0]
val_idx   = np.where(val_mask)[0]


# -------------------------------------------------
# DATASET / DATALOADER
# -------------------------------------------------

train_ds = Spatial_dataset(ds, solar, wind, hydro, train_idx)
val_ds   = Spatial_dataset(ds, solar, wind, hydro, val_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)


# -------------------------------------------------
# TRAINER
# -------------------------------------------------

class Trainer:

    def __init__(self, model, train_loader, val_loader, lr=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def forward_step(self, batch):
        meteo = batch["meteo"].to(self.device)

        targets = batch["targets"]["productions"]
        solar = targets["solar"].to(self.device)
        wind  = targets["wind"].to(self.device)
        hydro = targets["hydro"].to(self.device)

        out = self.model(meteo)  # dict: name -> (B,4)

        loss = (
            self.loss_fn(out["solar"], solar) +
            self.loss_fn(out["wind"], wind) +
            self.loss_fn(out["hydro"], hydro)
        )
        return loss

    def train_one_epoch(self):
        self.model.train()
        total = 0.0

        for batch in self.train_loader:
            loss = self.forward_step(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()

        return total / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.forward_step(batch)
                total += loss.item()

        return total / len(self.val_loader)

    def fit(self, epochs=10):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            print(f"\nEpoch {epoch+1}")
            print(f"  train loss: {train_loss:.5f}")
            print(f"  val loss:   {val_loss:.5f}")


# -------------------------------------------------
# MODEL + TRAINING
# -------------------------------------------------

trainer = Trainer(
    model=model.MultiFontSpatialModel(in_channels=7, H=104, W=225),
    train_loader=train_loader,
    val_loader=val_loader,
)

if __name__ == "__main__":
    trainer.fit(epochs=20)