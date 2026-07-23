"""
Stage 1/2/3 result visualization.

Reads the .npz file(s) written by evaluate.py and produces plots for
price, production, and revenue. Does no model inference itself -- run
train.py then evaluate.py first.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import COMMODITIES, EVAL_DIR, PLOTS_DIR, info, warn


def load_eval(split_name: str) -> dict:
    path = EVAL_DIR / f"eval_{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run evaluate.py --split {split_name} first."
        )
    info(f"Loading evaluation arrays from {path}")
    data = dict(np.load(path, allow_pickle=False))
    info(f"  {len(data['dates'])} days, keys: {sorted(data.keys())}")
    return data


def plot_price(data: dict, split_name: str, out_dir):
    dates = data["dates"]
    actual = data["price_actual"].reshape(-1)
    predicted = data["price_predicted"].reshape(-1)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    n_show = min(len(actual), 96 * 14)  # first ~2 weeks of 15-min slots
    axes[0].plot(actual[:n_show], label="actual", linewidth=1.2)
    axes[0].plot(predicted[:n_show], label="predicted", linewidth=1.0, alpha=0.8)
    axes[0].set_title(f"Price: actual vs predicted, first {n_show // 96} days ({split_name})")
    axes[0].set_xlabel("15-min slot")
    axes[0].set_ylabel("price")
    axes[0].legend()

    axes[1].scatter(actual, predicted, s=4, alpha=0.3)
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[1].plot(lims, lims, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title(f"Price scatter (RMSE={rmse:.4f})")
    axes[1].set_xlabel("actual")
    axes[1].set_ylabel("predicted")

    fig.tight_layout()
    out_path = out_dir / f"{split_name}_price.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    info(f"Saved price plot -> {out_path}")


def plot_production(data: dict, split_name: str, out_dir):
    n_commodities = len(COMMODITIES)
    fig, axes = plt.subplots(n_commodities, 1, figsize=(12, 3.2 * n_commodities), squeeze=False)

    for i, c in enumerate(COMMODITIES):
        ax = axes[i][0]
        actual = data[f"production_actual__{c}"].reshape(-1)
        forecast = data[f"production_forecast__{c}"].reshape(-1)
        corrected = data[f"production_corrected__{c}"].reshape(-1)

        mae_forecast = float(np.mean(np.abs(actual - forecast)))
        mae_corrected = float(np.mean(np.abs(actual - corrected)))

        n_show = min(len(actual), 96 * 7)  # first week
        ax.plot(actual[:n_show], label="actual", linewidth=1.2)
        ax.plot(forecast[:n_show], label=f"forecast (MAE={mae_forecast:.3f})", linewidth=1.0, alpha=0.7)
        ax.plot(corrected[:n_show], label=f"corrected (MAE={mae_corrected:.3f})", linewidth=1.0, alpha=0.9)
        ax.set_title(f"Production [{c}] -- first {n_show // 96} days ({split_name})")
        ax.set_xlabel("15-min slot")
        ax.legend(fontsize=8)

        improvement = (mae_forecast - mae_corrected) / mae_forecast * 100 if mae_forecast else float("nan")
        info(f"  [{c}] forecast MAE={mae_forecast:.4f}  corrected MAE={mae_corrected:.4f}  "
             f"({improvement:+.1f}% change from correction)")

    fig.tight_layout()
    out_path = out_dir / f"{split_name}_production.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    info(f"Saved production plot -> {out_path}")


def plot_revenue(data: dict, split_name: str, out_dir):
    revenue_hard = data["revenue_hard"]
    revenue_oracle = data["revenue_oracle"]
    n_days = len(revenue_hard)
    capture_ratio = np.divide(
        revenue_hard, revenue_oracle,
        out=np.full_like(revenue_hard, np.nan), where=revenue_oracle != 0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    x = np.arange(n_days)
    axes[0].bar(x - 0.2, revenue_hard, width=0.4, label="realized (hard)")
    axes[0].bar(x + 0.2, revenue_oracle, width=0.4, label="oracle (perfect price)", alpha=0.7)
    axes[0].set_title(f"Daily dispatch revenue ({split_name})")
    axes[0].set_xlabel("day index")
    axes[0].set_ylabel("revenue")
    axes[0].legend()

    axes[1].plot(x, capture_ratio, marker="o", markersize=3, linewidth=1)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    mean_cr = float(np.nanmean(capture_ratio))
    axes[1].set_title(f"Capture ratio per day (mean={mean_cr:.3f})")
    axes[1].set_xlabel("day index")
    axes[1].set_ylabel("hard / oracle revenue")

    fig.tight_layout()
    out_path = out_dir / f"{split_name}_revenue.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    info(f"Revenue summary ({split_name}): total_hard={revenue_hard.sum():.1f}  "
         f"total_oracle={revenue_oracle.sum():.1f}  mean_capture_ratio={mean_cr:.3f}")
    info(f"Saved revenue plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluate.py outputs.")
    parser.add_argument("--split", choices=["val", "train", "both"], default="val",
                         help="Which evaluated split to plot (default: val)")
    args = parser.parse_args()

    info("=" * 60)
    info("STARTING VISUALIZATION")
    info("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    splits_to_run = ["val", "train"] if args.split == "both" else [args.split]

    for split_name in splits_to_run:
        try:
            data = load_eval(split_name)
        except FileNotFoundError as e:
            warn(str(e))
            continue

        plot_price(data, split_name, PLOTS_DIR)
        plot_production(data, split_name, PLOTS_DIR)
        plot_revenue(data, split_name, PLOTS_DIR)

    info("=" * 60)
    info(f"VISUALIZATION COMPLETE. Plots saved under: {PLOTS_DIR}")
    info("=" * 60)


if __name__ == "__main__":
    main()
