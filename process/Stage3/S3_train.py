
import numpy as np
import pandas as pd


def optimal_buy_sell_windows(
    df,
    price_col="real_price",
    window=8,
    use_mean=True,
    signal=1000,
):
    """
    Finds the optimal buy/sell windows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a price column.
    price_col : str
        Name of the price column.
    window : int
        Window length (default 8).
    use_mean : bool
        If True compares window means.
        If False compares window sums.
    signal : int
        Absolute value of the output signal.

    Returns
    -------
    buy_start : int
    sell_start : int
    actions : np.ndarray
        Vector with -signal during buy window and +signal during sell window.
    window_values : np.ndarray
        Mean/sum of every window.
    """

    prices = df[price_col].to_numpy(dtype=float)
    n = len(prices)

    if n < 2 * window:
        raise ValueError("Series is too short.")

    # Window sums
    kernel = np.ones(window)
    window_values = np.convolve(prices, kernel, mode="valid")

    if use_mean:
        window_values /= window

    m = len(window_values)

    # Linear search
    min_idx = 0
    best_buy = 0
    best_sell = window
    best_profit = window_values[window] - window_values[0]

    for sell in range(window, m):

        candidate_buy = sell - window

        if window_values[candidate_buy] < window_values[min_idx]:
            min_idx = candidate_buy

        profit = window_values[sell] - window_values[min_idx]

        if profit > best_profit:
            best_profit = profit
            best_buy = min_idx
            best_sell = sell

    actions = np.zeros(n, dtype=int)

    actions[best_buy:best_buy + window] = -signal
    actions[best_sell:best_sell + window] = signal

    return best_buy, best_sell, actions, window_values

#-----------------------------------------------------------------

import sys, os
from pathlib import Path

ROOT = Path("C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project")
sys.path.insert(0, str(ROOT))

file_path = ROOT / "output" / "stage2" / "predicted_price.csv"

df = pd.read_csv(file_path)

buy, sell, actions, window_values = optimal_buy_sell_windows(
    df,
    price_col="real_price",
    window=8
)

df["accion"] = actions

print(f"Buy window : {buy} -> {buy+7}")
print(f"Sell window: {sell} -> {sell+7}")

print(df.iloc[buy])
print(df.iloc[sell])