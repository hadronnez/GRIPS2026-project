
from torch.utils.data import DataLoader

dataset = EnergyDataset(
    meteo,
    solar,
    wind,
    hydro,
    non_marketized,
    tie_line,
    price,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)