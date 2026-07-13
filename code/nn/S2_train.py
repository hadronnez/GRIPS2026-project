import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT = Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")

TARGET_PATHS = {
    "solar":            ROOT / "output/raw_datasets/Photovoltaic.csv",
    "wind":             ROOT / "output/raw_datasets/Wind_Power.csv",
    "hydro":            ROOT / "output/raw_datasets/Hydro_Power.csv",
    "non_marketized":   ROOT / "output/raw_datasets/Non_Marketized_Unit.csv",
    "tie_line":         ROOT / "output/raw_datasets/Tie_Line.csv",
    "system_load":      ROOT / "output/raw_datasets/System_Load.csv",
    "price":            ROOT / "output/raw_datasets/Price.csv",
}

TRAIN_RANGE = ("2025-01-02", "2025-10-31")
VAL_RANGE = ("2025-11-01", "2025-12-30")

COL_CFG = dict(
    timestamp="time",
    actual=["actual_00", "actual_15", "actual_30", "actual_45"],
    forecast=["forecast_00", "forecast_15", "forecast_30", "forecast_45"],
)

SEQ_LEN = 96
PRED_LEN = 96
STRIDE = 96
BATCH_SIZE = 1


def wide_to_15min(df: pd.DataFrame, col_order: list[str] | None = None) -> pd.Series:
    """
    df: index = timestamp horario, columnas = 4 valores de 15 min dentro de esa hora.
    col_order: columnas en orden temporal creciente (:00, :15, :30, :45).
               Si None, se asume que df.columns ya está en ese orden.
    """
    if col_order is None:
        col_order = df.columns.tolist()
    assert len(col_order) == 4, "Se esperan 4 columnas (una por cuarto de hora)"

    values = df[col_order].to_numpy().reshape(-1)  # row-major: hora0[00,15,30,45], hora1[...]
    new_index = pd.date_range(start=df.index[0], periods=len(values), freq="15min")
    return pd.Series(values, index=new_index)


def load_series(path: Path, col_cfg: dict, value_type: str = "actual") -> pd.Series:
    """
    Lee un CSV con columna de timestamp + 4 columnas de 15 min (actual o forecast),
    y devuelve una serie continua a resolución de 15 min.
    """
    df = pd.read_csv(path, parse_dates=[col_cfg["timestamp"]])
    df = df.set_index(col_cfg["timestamp"]).sort_index()
    return wide_to_15min(df, col_order=col_cfg[value_type])

# ---------------------------------------------------------------------------
# 2. Carga: forecast para las exógenas, actual SOLO para price (target)
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

exo_series_list = [
    load_series(TARGET_PATHS[name], COL_CFG, value_type="actual")
    for name in EXO_NAMES
]

combined = pd.concat(exo_series_list, axis=1)
combined.columns = EXO_NAMES

price_actual = load_series(
    TARGET_PATHS["price"],
    COL_CFG,
    value_type="actual"
)

combined, price_actual = combined.align(
    price_actual,
    join="inner",
    axis=0
)

n_missing = combined.isna().sum()
if n_missing.any():
    print("Aviso: NaN por columna tras el concat:")
    print(n_missing)


# ---------------------------------------------------------------------------
# 3. Split train / val (igual que antes, pero aplicado también a price_actual)
# ---------------------------------------------------------------------------
train_df = combined.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
val_df = combined.loc[VAL_RANGE[0]:VAL_RANGE[1]]

price_train = price_actual.loc[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
price_val = price_actual.loc[VAL_RANGE[0]:VAL_RANGE[1]]

data_train = torch.tensor(train_df.values, dtype=torch.float32)  # (T_train, N) -- canal price = 0
data_val = torch.tensor(val_df.values, dtype=torch.float32)

price_train_t = torch.tensor(price_train.values, dtype=torch.float32)  # (T_train,)
price_val_t = torch.tensor(price_val.values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 5. Dataset: X viene de "combined" (exógenas + placeholder price),
#    y viene de la serie de precio real normalizada, alineada por índice
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

        # Boundary (and other exogenous variables)
        X = self.data[start:end]

        # Price at the SAME timestamps
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
# 6. Modelo iTransformer + entrenamiento, usando train_loader/val_loader
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

import torch.nn as nn
from types import SimpleNamespace
from models.iTransformer import Model

configs = SimpleNamespace(
    task_name="long_term_forecast",
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    output_attention=False,
    use_norm=False,

    d_model=128,
    embed="timeF",
    freq="t",
    dropout=0.1,

    class_strategy="projection",
    factor=1,
    n_heads=4,
    d_ff=256,
    activation="gelu",
    e_layers=2,

    enc_in=N,
)

model = Model(configs)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
N_EPOCHS = 6

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss, n_batches = 0.0, 0
    for X_batch, y_batch in train_loader:
        pred = model(X_batch, None, None, None)[0]
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    if epoch % 1 == 0 or epoch == N_EPOCHS - 1:
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                pred_val = model(X_val, None, None, None)[0]
                val_loss += criterion(pred_val, y_val).item()
                val_batches += 1
        print(f"epoch {epoch:2d}  train_loss={epoch_loss/n_batches:.4f}  val_loss={val_loss/val_batches:.4f}")

# ---------------------------------------------------------------------------
# 7. Evaluación final: comparar contra baseline "predecir la media de train" (ya normalizada -> ~0)
# ---------------------------------------------------------------------------

model.eval()
final_val_loss, baseline_loss, n_val_batches = 0.0, 0.0, 0
with torch.no_grad():
    for X_val, y_val in val_loader:
        pred_val = model(X_val, None, None, None)[0]
        final_val_loss += criterion(pred_val, y_val).item()
        baseline_loss += criterion(torch.zeros_like(y_val), y_val).item()
        n_val_batches += 1

final_val_loss /= n_val_batches
baseline_loss /= n_val_batches
print(f"\nval_loss final: {final_val_loss:.4f}  |  baseline (media de train): {baseline_loss:.4f}  "
      f"|  mejora: {(1 - final_val_loss / baseline_loss) * 100:.1f}%")


# ---------------------------------------------------------------------------
# 8. Guardar predicciones del conjunto de validación
# ---------------------------------------------------------------------------

model.eval()

results = []

val_index = val_df.index

with torch.no_grad():

    for window_idx in range(len(val_dataset)):

        X, y = val_dataset[window_idx]

        X = X.unsqueeze(0)

        pred = model(X, None, None, None)[0]

        # Desnormalizar
        pred = pred.cpu().numpy()

        # Timestamps del día predicho
        start_pred = window_idx * STRIDE 
        end_pred = start_pred + SEQ_LEN

        pred_times = val_index[start_pred:end_pred]

        # Precio real (desnormalizado)
        real = (
            y.cpu().numpy()
        )

        for t, p, r in zip(pred_times, pred, real):

            results.append(
                {
                    "time": t,
                    "predicted_price": float(p),
                    "real_price": float(r),
                }
            )

pred_df = pd.DataFrame(results)

output_path = ROOT / "output" / "predictions_validation.csv"

pred_df.to_csv(output_path, index=False)

print(f"Predicciones guardadas en: {output_path}")

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(
    pred_df["real_price"],
    pred_df["predicted_price"]
)

rmse = mean_squared_error(
    pred_df["real_price"],
    pred_df["predicted_price"],
) ** 0.5

r2 = r2_score(
    pred_df["real_price"],
    pred_df["predicted_price"]
)

print(f"MAE :  {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²  :  {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))

plt.plot(
    pred_df["time"],
    pred_df["real_price"],
    label="Precio real",
)

plt.plot(
    pred_df["time"],
    pred_df["predicted_price"],
    label="Predicción",
)

plt.title(
    f"Predicción day-ahead sobre validación (R² = {r2:.4f})"
)

plt.xlabel("Tiempo")
plt.ylabel("Precio")

plt.legend()

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()