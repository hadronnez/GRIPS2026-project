
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------
class SolarDataset(Dataset):
    """
    Espera:
      meteo:  array (T, 5, H, W)  -> T horas, 5 variables meteorológicas
      target: array (T,)          -> producción total agregada por hora

    Ajusta esta clase a como tengas guardados tus datos (npy, NetCDF con
    xarray, zarr, etc.). Aquí se asume que ya tienes arrays de numpy.
    """

    def __init__(self, meteo: np.ndarray, target: np.ndarray, normalize=True):
        assert meteo.shape[0] == target.shape[0], "T debe coincidir en meteo y target"
        self.H, self.W = meteo.shape[2], meteo.shape[3]

        if normalize:
            # normalización por canal (media/std sobre todo el año)
            mean = meteo.mean(axis=(0, 2, 3), keepdims=True)
            std = meteo.std(axis=(0, 2, 3), keepdims=True) + 1e-8
            meteo = (meteo - mean) / std
            self.meteo_mean = mean
            self.meteo_std = std

        self.meteo = torch.tensor(meteo, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.meteo.shape[0]

    def __getitem__(self, idx):
        return self.meteo[idx], self.target[idx]


# ----------------------------------------------------------------------
# 2. Capacity map (estático, un único tensor para todo el año)
# ----------------------------------------------------------------------
class CapacityMap(nn.Module):
    def __init__(self, H, W, init_value=0.1):
        super().__init__()
        # un parámetro entrenable por píxel, NO depende del input
        self.raw_capacity = nn.Parameter(torch.ones(1, 1, H, W) * init_value)

    def forward(self):
        # softplus: mantiene la capacidad no-negativa
        return torch.nn.functional.softplus(self.raw_capacity)  # (1,1,H,W)


# ----------------------------------------------------------------------
# 3. Efficiency net (depende de la meteorología de cada hora)
# ----------------------------------------------------------------------
class EfficiencyNet(nn.Module):
    def __init__(self, in_channels=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, meteo):
        # meteo: (batch, 5, H, W)
        raw = self.net(meteo)
        return torch.sigmoid(raw)  # (batch, 1, H, W), acotado en [0,1]


# ----------------------------------------------------------------------
# 4. Modelo combinado
# ----------------------------------------------------------------------
class SolarAttributionModel(nn.Module):
    def __init__(self, H, W, in_channels=5, hidden=32):
        super().__init__()
        self.capacity_map = CapacityMap(H, W)
        self.efficiency_net = EfficiencyNet(in_channels, hidden)

    def forward(self, meteo):
        C = self.capacity_map()                # (1,1,H,W) -> broadcast sobre batch
        E = self.efficiency_net(meteo)          # (batch,1,H,W)
        pixel_energy = C * E                    # (batch,1,H,W)
        total_energy = pixel_energy.sum(dim=(2, 3)).squeeze(1)  # (batch,)
        return total_energy, pixel_energy


# ----------------------------------------------------------------------
# 5. Regularización: sparsity (L1) + suavidad espacial (TV)
# ----------------------------------------------------------------------
def sparsity_loss(capacity_map: torch.Tensor) -> torch.Tensor:
    return capacity_map.abs().mean()


def total_variation_loss(capacity_map: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(capacity_map[:, :, 1:, :] - capacity_map[:, :, :-1, :]).mean()
    dw = torch.abs(capacity_map[:, :, :, 1:] - capacity_map[:, :, :, :-1]).mean()
    return dh + dw


# ----------------------------------------------------------------------
# 6. Entrenamiento
# ----------------------------------------------------------------------
def train(
    meteo: np.ndarray,
    target: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_sparsity: float = 1e-4,
    lambda_smooth: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dataset = SolarDataset(meteo, target)
    H, W = dataset.H, dataset.W

    # split temporal por bloques (no aleatorio) para evitar fuga por autocorrelación
    n = len(dataset)
    val_frac = 0.15
    split_idx = int(n * (1 - val_frac))
    train_idx = list(range(0, split_idx))
    val_idx = list(range(split_idx, n))

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False
    )

    model = SolarAttributionModel(H, W, in_channels=meteo.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.HuberLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for meteo_batch, target_batch in train_loader:
            meteo_batch = meteo_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            pred, _ = model(meteo_batch)

            loss_fit = huber(pred, target_batch)
            C = model.capacity_map()
            loss_sp = sparsity_loss(C)
            loss_tv = total_variation_loss(C)

            loss = loss_fit + lambda_sparsity * loss_sp + lambda_smooth * loss_tv
            loss.backward()
            optimizer.step()

            train_loss += loss_fit.item() * meteo_batch.size(0)

        train_loss /= len(train_idx)

        # validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for meteo_batch, target_batch in val_loader:
                meteo_batch = meteo_batch.to(device)
                target_batch = target_batch.to(device)
                pred, _ = model(meteo_batch)
                val_loss += huber(pred, target_batch).item() * meteo_batch.size(0)
        val_loss /= len(val_idx)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    return model, dataset


# ----------------------------------------------------------------------
# 7. Inspección del capacity map aprendido
# ----------------------------------------------------------------------
def get_capacity_map_numpy(model: SolarAttributionModel) -> np.ndarray:
    with torch.no_grad():
        C = model.capacity_map().squeeze().cpu().numpy()  # (H, W)
    return C


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EJEMPLO DE USO con datos sintéticos (sustituye por tus datos reales)
    # ------------------------------------------------------------------
    T, C_in, H, W = 8760, 5, 32, 32  # un año de horas, 5 variables, grid 32x32

    rng = np.random.default_rng(42)
    meteo_synth = rng.normal(size=(T, C_in, H, W)).astype(np.float32)

    # target sintético de juguete: solo para verificar que el pipeline corre
    target_synth = meteo_synth[:, 0].mean(axis=(1, 2)).astype(np.float32)

    model, dataset = train(meteo_synth, target_synth, epochs=20, batch_size=64)

    capacity = get_capacity_map_numpy(model)
    print("Capacity map shape:", capacity.shape)
    print("Capacity map min/max:", capacity.min(), capacity.max())

    # Para visualizar en tu entorno real:
    # import matplotlib.pyplot as plt
    # plt.imshow(capacity, cmap="viridis")
    # plt.colorbar(label="Capacidad aprendida")
    # plt.title("Mapa de capacidad solar aprendido")
    # plt.show()
