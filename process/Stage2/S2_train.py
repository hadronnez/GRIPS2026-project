
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import sys, os

ROOT = Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")
sys.path.insert(0, str(ROOT))

import torch.nn as nn
from types import SimpleNamespace
from process.stage2.model.iTransformer import Model

CHECKPOINT_PATH = ROOT / "process" / "stage2" / "parameters" / "best_model_enhanced_forecast.pt"

TARGET_PATHS = {
    "solar":            ROOT / "output/stage1/solar_forecast.csv",                                  #enhanced forecast (stage1 output, long format)
    "wind":             ROOT / "output/stage1/wind_forecast.csv",                                   #enhanced forecast (stage1 output, long format)
    "hydro":            ROOT / "output/stage1/hydro_forecast.csv",                                  #enhanced forecast (stage1 output, long format)
    "non_marketized":   ROOT / "output/data_preprocessing/clean_datasets/Non_Marketized_Unit.csv",
    "tie_line":         ROOT / "output/data_preprocessing/clean_datasets/Tie_Line.csv",
    "system_load":      ROOT / "output/data_preprocessing/clean_datasets/System_Load.csv",
    "price":            ROOT / "output/data_preprocessing/clean_datasets/Price.csv",
}


output_path = ROOT / "output" / "stage2" / "predictions_validation.csv"

TRAIN_RANGE = ("2025-01-02", "2025-10-31")
VAL_RANGE = ("2025-11-01", "2025-12-30")
FULL_RANGE = (TRAIN_RANGE[0], VAL_RANGE[1])  

COL_CFG = dict(
    timestamp="time",
    actual=["actual_00", "actual_15", "actual_30", "actual_45"],
    forecast=["forecast_00", "forecast_15", "forecast_30", "forecast_45"],
)


STAGE1_NAMES = {"solar", "wind", "hydro"}

SEQ_LEN = 48
PRED_LEN = 48
STRIDE = 48
BATCH_SIZE = 8
N_EPOCHS = 30


def wide_to_15min(df: pd.DataFrame, col_order: list[str] | None = None) -> pd.Series:
    """
    df: index = hourly timestamp, columns = 4 values of 15 min within that hour.
    col_order: columns in increasing temporal order (:00, :15, :30, :45).
               If None, assumes df.columns is already in that order.
    """
    if col_order is None:
        col_order = df.columns.tolist()
    assert len(col_order) == 4, "Expected 4 columns (one per quarter-hour)"

    values = df[col_order].to_numpy().reshape(-1)  # row-major: hour0[00,15,30,45], hour1[...]
    new_index = pd.date_range(start=df.index[0], periods=len(values), freq="15min")
    return pd.Series(values, index=new_index)


def load_series(
    path: Path,
    col_cfg: dict,
    value_type: str = "actual",
    date_range: tuple[str, str] | None = None,
) -> pd.Series:
    """
    Reads a CSV with a timestamp column + 4 quarter-hour columns (actual or forecast),
    and returns a continuous series at 15-min resolution. If date_range is given,
    the raw hourly rows are filtered BEFORE expanding to 15-min, so only the
    train/val window is ever reshaped.
    """
    df = pd.read_csv(path, parse_dates=[col_cfg["timestamp"]])
    df = df.set_index(col_cfg["timestamp"]).sort_index()
    if date_range is not None:
        df = df.loc[date_range[0]:date_range[1]]
    return wide_to_15min(df, col_order=col_cfg[value_type])


def load_long_series(
    path: Path,
    value_col: str = "improved_forecast",
    date_range: tuple[str, str] | None = None,
) -> pd.Series:
    """
    Reads a stage1 output CSV (time, actual, forecast, improved_forecast),
    already at 15-min resolution. Doesn't need wide_to_15min.
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    if date_range is not None:
        df = df.loc[date_range[0]:date_range[1]]
    return df[value_col]


def load_exo_series(name: str, date_range: tuple[str, str] | None = None) -> pd.Series:
    """
    Dispatches to the correct loader depending on whether the exogenous variable
    comes from stage1 (long format) or from the raw preprocessed datasets
    (wide, hourly format).
    """
    path = TARGET_PATHS[name]
    if name in STAGE1_NAMES:
        return load_long_series(path, value_col="improved_forecast", date_range=date_range)
    return load_series(path, COL_CFG, value_type="forecast", date_range=date_range)

# ---------------------------------------------------------------------------
# 2. Load: forecast (or improved_forecast for stage1) for the exogenous
#    variables, actual ONLY for price (target) — restricted to FULL_RANGE
#    (train + val) at load time, not sliced afterwards.
# ---------------------------------------------------------------------------

EXO_NAMES = [
    "solar",
    "wind",
    "hydro",
    "non_marketized",
    "tie_line",
    "system_load",
]

N = len(EXO_NAMES)

exo_series_list = [load_exo_series(name, date_range=FULL_RANGE) for name in EXO_NAMES]

combined = pd.concat(exo_series_list, axis=1)
combined.columns = EXO_NAMES

price_actual = load_series(
    TARGET_PATHS["price"],
    COL_CFG,
    value_type="actual",
    date_range=FULL_RANGE,
)

combined, price_actual = combined.align(
    price_actual,
    join="inner",
    axis=0
)

n_missing = combined.isna().sum()
if n_missing.any():
    print("Warning: NaNs per column after concat:")
    print(n_missing)


# ---------------------------------------------------------------------------
# 3. Train / val split (combined/price_actual now already only span FULL_RANGE,
#    this just separates the two sub-windows within it)
# ---------------------------------------------------------------------------
train_df = combined.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
val_df = combined.loc[VAL_RANGE[0]:VAL_RANGE[1]]

price_train = price_actual.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
price_val = price_actual.loc[VAL_RANGE[0]:VAL_RANGE[1]]

data_train = torch.tensor(train_df.values, dtype=torch.float32)  # (T_train, N) -- price channel = 0
data_val = torch.tensor(val_df.values, dtype=torch.float32)

price_train_t = torch.tensor(price_train.values, dtype=torch.float32)  # (T_train,)
price_val_t = torch.tensor(price_val.values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 5. Dataset: X comes from "combined" (exogenous + price placeholder),
#    y comes from the real price series, aligned by index
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, data, price, seq_len, pred_len, stride=1):
        if seq_len != pred_len:
            raise ValueError("For simultaneous prediction, seq_len must equal pred_len.")

        n_windows = (data.shape[0] - seq_len) // stride + 1
        if n_windows <= 0:
            raise ValueError(
                f"Series too short: need at least {seq_len} samples, got {data.shape[0]}"
            )

        self.data = data
        self.price = price
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.n_windows = n_windows

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len

        X = self.data[start:end]
        y = self.price[start:end]

        return X, y


train_dataset = WindowDataset(
    data_train,
    price_train_t,
    SEQ_LEN,
    PRED_LEN,
    stride=STRIDE,
)

val_dataset = WindowDataset(
    data_val,
    price_val_t,
    SEQ_LEN,
    PRED_LEN,
    stride=STRIDE,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# 6. iTransformer model + training, using train_loader/val_loader
# ---------------------------------------------------------------------------


configs = SimpleNamespace(
    task_name="long_term_forecast",
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    output_attention=False,
    use_norm=False,

    d_model=256,
    embed="timeF",
    freq="t",
    dropout=0.10,

    class_strategy="projection",
    factor=1,
    n_heads=16,
    d_ff=256,
    activation="gelu",
    e_layers=2,

    enc_in=N,
)

model = Model(configs)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()


best_val_loss = float("inf")

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss, n_batches = 0.0, 0
    for X_batch, y_batch in train_loader:
        pred = model(X_batch, None, None, None)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    model.eval()
    val_loss, val_batches = 0.0, 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            pred_val = model(X_val, None, None, None)
            val_loss += criterion(pred_val, y_val).item()
            val_batches += 1
    val_loss /= val_batches
    train_loss = epoch_loss / n_batches

    print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "configs": vars(configs),
            },
            CHECKPOINT_PATH,
        )
        print(f"  -> new best val_loss ({val_loss:.4f}), checkpoint saved to {CHECKPOINT_PATH}")

# ---------------------------------------------------------------------------
# 7. Final evaluation: compare against the "predict the train mean" baseline
#    (already normalized -> ~0)
# ---------------------------------------------------------------------------

print(f"\nLoading best checkpoint (val_loss={best_val_loss:.4f}) from {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
final_val_loss, baseline_loss, n_val_batches = 0.0, 0.0, 0
with torch.no_grad():
    for X_val, y_val in val_loader:
        pred_val = model(X_val, None, None, None)
        final_val_loss += criterion(pred_val, y_val).item()
        baseline_loss += criterion(torch.zeros_like(y_val), y_val).item()
        n_val_batches += 1

final_val_loss /= n_val_batches
baseline_loss /= n_val_batches
print(f"\nfinal val_loss: {final_val_loss:.4f}  |  baseline (train mean): {baseline_loss:.4f}  "
      f"|  improvement: {(1 - final_val_loss / baseline_loss) * 100:.1f}%")


# ---------------------------------------------------------------------------
# 8. Save validation set predictions
# ---------------------------------------------------------------------------

model.eval()

results = []

val_index = val_df.index

with torch.no_grad():

    for window_idx in range(len(val_dataset)):

        X, y = val_dataset[window_idx]

        X = X.unsqueeze(0)

        pred = model(X, None, None, None)

        pred = pred.cpu().numpy()
        pred = pred.squeeze(0)

        start_pred = window_idx * STRIDE
        end_pred = start_pred + SEQ_LEN

        pred_times = val_index[start_pred:end_pred]

        real = y.cpu().numpy()

        for t, p, r in zip(pred_times, pred, real):

            results.append(
                {
                    "time": t,
                    "predicted_price": float(p),
                    "real_price": float(r),
                }
            )

pred_df = pd.DataFrame(results)

pred_df.to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(pred_df["real_price"], pred_df["predicted_price"])
rmse = mean_squared_error(pred_df["real_price"], pred_df["predicted_price"]) ** 0.5
r2 = r2_score(pred_df["real_price"], pred_df["predicted_price"])

print(f"MAE :  {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²  :  {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))
plt.plot(pred_df["time"], pred_df["real_price"], label="Actual price")
plt.plot(pred_df["time"], pred_df["predicted_price"], label="Predicted price")
plt.title(f"Price prediction (R² = {r2:.4f})")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()