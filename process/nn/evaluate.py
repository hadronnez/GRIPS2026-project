"""
Stage 1/2/3 evaluation.

Loads the checkpoint saved by train.py, runs it over a chosen split
(default: validation) with no gradient updates, and saves the
day-by-day price / production / revenue arrays to an .npz file plus a
per-day summary .csv. visualize.py reads those files to make plots --
this script does not plot anything itself.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (
    CHECKPOINT_PATH,
    COMMODITIES,
    EVAL_DIR,
    T_MIN,
    build_datasets,
    collate_days,
    info,
    load_model,
    warn,
)
from metrics import metrics as compute_metrics


def evaluate_split(model, loader, device, split_name):
    info(f"Running inference on '{split_name}' split ({len(loader.dataset)} days, {len(loader)} batches)")
    t_start = time.time()

    all_dates = []
    price_actual_chunks, price_pred_chunks = [], []
    prod_actual = {c: [] for c in COMMODITIES}
    prod_forecast = {c: [] for c in COMMODITIES}
    prod_corrected = {c: [] for c in COMMODITIES}
    power_hard_chunks, power_oracle_chunks = [], []
    revenue_hard_chunks, revenue_oracle_chunks = [], []
    p_null_chunks = []
    rmse_stage1_sum, rmse_stage2_sum, n_batches_seen = 0.0, 0.0, 0

    used_fallback_production = False

    with torch.no_grad():
        for batch_idx, (weather, forecast_day, actual_day, exo_day, price_actual_day, dates) in enumerate(loader):
            weather = {c: w.to(device) for c, w in weather.items()}
            forecast_day_dev = {c: v.to(device) for c, v in forecast_day.items()}
            actual_day_dev = {c: v.to(device) for c, v in actual_day.items()}
            exo_day = {n: v.to(device) for n, v in exo_day.items()}
            price_actual_day = price_actual_day.to(device)

            out = model(weather, forecast_day_dev, actual_day_dev, exo_day, temperature=T_MIN)

            power_hard = model.dispatch.hard(out["predicted_price"])
            power_oracle = model.dispatch.hard(price_actual_day)
            revenue_hard = (power_hard * price_actual_day).sum(dim=-1)
            revenue_oracle = (power_oracle * price_actual_day).sum(dim=-1)

            # Same metrics.py used in training validation, for a consistent
            # rmse_stage1/rmse_stage2 definition across train/eval. It
            # recomputes model.dispatch.hard() internally (never the soft
            # schedule), so its revenue_mean should match revenue_hard.mean()
            # above up to floating point noise.
            m = compute_metrics(model, out, price_actual_day)
            rmse_stage1_sum += m["rmse_stage1"].item()
            rmse_stage2_sum += m["rmse_stage2"].item()
            n_batches_seen += 1

            all_dates.extend(dates)
            price_actual_chunks.append(price_actual_day.cpu().numpy())
            price_pred_chunks.append(out["predicted_price"].cpu().numpy())
            power_hard_chunks.append(power_hard.cpu().numpy())
            power_oracle_chunks.append(power_oracle.cpu().numpy())
            revenue_hard_chunks.append(revenue_hard.cpu().numpy())
            revenue_oracle_chunks.append(revenue_oracle.cpu().numpy())
            if "p_null" in out:
                p_null_chunks.append(out["p_null"].cpu().numpy())

            for c in COMMODITIES:
                prod_actual[c].append(actual_day[c].numpy())
                prod_forecast[c].append(forecast_day[c].numpy())
                if "corrected_production" in out:
                    prod_corrected[c].append(out["corrected_production"][c].cpu().numpy())
                else:
                    # architecture.py doesn't expose corrected production directly
                    # in this build, so reconstruct it from the predicted residual:
                    # corrected = forecast + (residual_pred_norm * std + mean).
                    used_fallback_production = True
                    mean, std = model.residual_stats[c]
                    residual_pred = out["residual_preds_norm"][c].cpu().numpy() * std + mean
                    prod_corrected[c].append(forecast_day[c].numpy() + residual_pred)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                info(f"  [{split_name}] batch {batch_idx + 1}/{len(loader)} processed")

    if used_fallback_production:
        warn(
            "'corrected_production' not found in model output; production_corrected was "
            "reconstructed from forecast + denormalized residual_preds_norm. If "
            "architecture.py exposes true corrected production under a different key, "
            "update evaluate.py to use it directly."
        )

    results = {
        "dates": np.array([str(d) for d in all_dates]),
        "price_actual": np.concatenate(price_actual_chunks, axis=0),
        "price_predicted": np.concatenate(price_pred_chunks, axis=0),
        "power_hard": np.concatenate(power_hard_chunks, axis=0),
        "power_oracle": np.concatenate(power_oracle_chunks, axis=0),
        "revenue_hard": np.concatenate(revenue_hard_chunks, axis=0),
        "revenue_oracle": np.concatenate(revenue_oracle_chunks, axis=0),
    }
    if p_null_chunks:
        results["p_null"] = np.concatenate(p_null_chunks, axis=0)
    for c in COMMODITIES:
        results[f"production_actual__{c}"] = np.concatenate(prod_actual[c], axis=0)
        results[f"production_forecast__{c}"] = np.concatenate(prod_forecast[c], axis=0)
        results[f"production_corrected__{c}"] = np.concatenate(prod_corrected[c], axis=0)

    n_days = len(all_dates)
    total_hard = results["revenue_hard"].sum()
    total_oracle = results["revenue_oracle"].sum()
    capture_ratio = total_hard / total_oracle if total_oracle else float("nan")
    price_rmse = float(np.sqrt(np.mean((results["price_actual"] - results["price_predicted"]) ** 2)))

    info(f"'{split_name}' split summary over {n_days} days:")
    info(f"  revenue(hard)  total={total_hard:.1f}  mean/day={total_hard / n_days:.1f}")
    info(f"  revenue(oracle) total={total_oracle:.1f}  mean/day={total_oracle / n_days:.1f}")
    info(f"  capture_ratio={capture_ratio:.3f}")
    info(f"  price RMSE={price_rmse:.4f}")
    if n_batches_seen:
        info(f"  rmse_stage1={rmse_stage1_sum / n_batches_seen:.4f}  "
             f"rmse_stage2={rmse_stage2_sum / n_batches_seen:.4f}  (metrics.py, batch-averaged)")
    for c in COMMODITIES:
        mae = float(np.mean(np.abs(results[f"production_actual__{c}"] - results[f"production_corrected__{c}"])))
        info(f"  [{c}] corrected-production MAE vs actual = {mae:.4f}")
    info(f"Inference on '{split_name}' finished in {time.time() - t_start:.1f}s")

    return results


def save_results(results: dict, split_name: str):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"eval_{split_name}.npz"
    np.savez_compressed(out_path, **results)
    info(f"Saved full evaluation arrays to {out_path}")

    # Small per-day summary CSV for quick inspection without loading the npz.
    import csv

    summary_path = EVAL_DIR / f"eval_{split_name}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "revenue_hard", "revenue_oracle", "capture_ratio"])
        for i, date in enumerate(results["dates"]):
            rh = results["revenue_hard"][i]
            ro = results["revenue_oracle"][i]
            cr = rh / ro if ro else float("nan")
            writer.writerow([date, f"{rh:.2f}", f"{ro:.2f}", f"{cr:.3f}"])
    info(f"Saved per-day summary CSV to {summary_path}")


def save_timestep_csv(results: dict, split_name: str):
    """
    Fine-grained CSV: one row per (day, timestep) at the model's native
    resolution (15 minutes, for slots_per_day=96), with the realized
    hard-dispatch operation at that timestep.

    operation: -1000 while charging, +1000 while discharging, 0 while idle.
    This follows S3_model.Model's sign convention for `power_hard`
    (>0 = discharge, <0 = charge, 0 = null/idle) -- it does not re-derive
    charge/discharge from anything else, just reads the sign.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    power_hard = results["power_hard"]          # (n_days, T)
    price_actual = results["price_actual"]        # (n_days, T)
    price_predicted = results["price_predicted"]  # (n_days, T)
    n_days, T = power_hard.shape

    minutes_per_slot = 24 * 60 / T
    if minutes_per_slot != 15:
        warn(f"slots_per_day={T} implies {minutes_per_slot:.2f} minutes/slot, not the "
             f"expected 15 -- timestamps in the timestep CSV will use this spacing instead.")
    slot_delta = pd.Timedelta(minutes=minutes_per_slot)

    operation = np.zeros_like(power_hard)
    operation[power_hard > 0] = 1000.0   # discharging
    operation[power_hard < 0] = -1000.0  # charging

    rows = []
    for i, date_str in enumerate(results["dates"]):
        day_start = pd.Timestamp(str(date_str))
        timestamps = [day_start + t * slot_delta for t in range(T)]
        for t in range(T):
            rows.append((
                timestamps[t],
                power_hard[i, t],
                operation[i, t],
                price_actual[i, t],
                price_predicted[i, t],
            ))

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "power_hard", "operation", "price_actual", "price_predicted"],
    )
    out_path = EVAL_DIR / f"eval_{split_name}_timesteps.csv"
    df.to_csv(out_path, index=False)
    info(f"Saved per-timestep dispatch CSV ({n_days} days x {T} slots, "
         f"{minutes_per_slot:.0f}min/slot) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained S123NN checkpoint.")
    parser.add_argument("--split", choices=["val", "train", "both"], default="val",
                         help="Which split to evaluate (default: val)")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a checkpoint (defaults to CHECKPOINT_PATH from common.py)")
    args = parser.parse_args()

    info("=" * 60)
    info("STARTING EVALUATION")
    info("=" * 60)

    train_ds, val_ds, residual_stats = build_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device}")

    checkpoint_path = args.checkpoint if args.checkpoint else CHECKPOINT_PATH
    model, ckpt = load_model(residual_stats, device, checkpoint_path=checkpoint_path)
    model.residual_stats = residual_stats  # used by the production fallback above

    splits_to_run = ["val", "train"] if args.split == "both" else [args.split]

    for split_name in splits_to_run:
        ds = val_ds if split_name == "val" else train_ds
        if len(ds) == 0:
            warn(f"'{split_name}' split has 0 days -- skipping")
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_days)
        results = evaluate_split(model, loader, device, split_name)
        save_results(results, split_name)
        save_timestep_csv(results, split_name)

    info("=" * 60)
    info("EVALUATION COMPLETE")
    info(f"Results directory: {EVAL_DIR}")
    info("Next: run visualize.py to plot price, production, and revenue.")
    info("=" * 60)


if __name__ == "__main__":
    main()