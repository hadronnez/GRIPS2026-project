"""
Stage 1/2/3 joint training.

Trains the decision-focused (revenue-based) S123NN model and checkpoints
the best-validation-revenue epoch to CHECKPOINT_PATH. Run evaluate.py
afterwards to score that checkpoint and dump price/production/revenue
arrays, then visualize.py to plot them.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    BATCH_LOG_EVERY,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    LAMBDA_END,
    LAMBDA_START,
    LR,
    N_EPOCHS,
    STAGE1_PRETRAINED,
    STAGE2_PRETRAINED,
    S123NN,
    T0,
    T_MIN,
    WEIGHT_DECAY,
    build_datasets,
    collate_days,
    config,
    info,
    temperature_schedule,
    warn,
)
from loss import LossFn
from metrics import metrics as compute_metrics


WARMUP_FRACTION = getattr(config.training, "warmup_fraction", 1)
GAMMA_START = getattr(config.training, "gamma_start", 0.15)
GAMMA_END = getattr(config.training, "gamma_end", 0.15)
ALPHA_BETA_FLOOR = getattr(config.training, "alpha_beta_floor", 1.5)
GRAD_CLIP_NORM = getattr(config.training, "grad_clip_norm", 5.0)
SEED = getattr(config.traning, "seed", None)


def load_pretrained_weights(model):
    """Stage 1 has one branch per commodity (model.stage1[c]), so its
    pretrained weights are a dict keyed by commodity. Stage 2 is a single
    shared branch that takes all three corrected series (plus exogenous
    channels) concatenated as input, so it has exactly one pretrained
    checkpoint, not one per commodity."""
    n_loaded = 0
    for c, path in STAGE1_PRETRAINED.items():
        if path is not None:
            ckpt = torch.load(path, weights_only=False)
            model.stage1[c].load_state_dict(ckpt["model_state_dict"])
            info(f"Loaded pretrained Stage 1 [{c}] from {path}")
            n_loaded += 1
        else:
            info(f"Stage 1 [{c}]: no pretrained weights configured, training from scratch")

    if STAGE2_PRETRAINED is not None:
        ckpt = torch.load(STAGE2_PRETRAINED, weights_only=False)
        model.stage2.load_state_dict(ckpt["model_state_dict"])
        info(f"Loaded pretrained Stage 2 from {STAGE2_PRETRAINED}")
        n_loaded += 1
    else:
        info("Stage 2: no pretrained weights configured, training from scratch")

    return n_loaded


def run_train_epoch(model, loader, optimizer, loss_fn, T, lam, gamma, device, epoch, n_epochs):
    """loss_fn is a loss.LossFn instance, kept alive across epochs so its
    per-component EMA scale estimates keep adapting through the whole run.
    `lam` (Stage 1/2 weight, floored at ALPHA_BETA_FLOOR) is applied to both
    alpha and beta every epoch; `gamma` (revenue weight, ramping up from
    GAMMA_START to GAMMA_END) is applied on the same schedule -- see the
    warm-up comment near the top of this file for why both move together."""
    model.train()
    loss_fn.alpha = lam
    loss_fn.beta = lam
    loss_fn.gamma = gamma

    total_loss, total_revenue, total_price_mse, total_p_null = 0.0, 0.0, 0.0, 0.0
    n_batches = len(loader)

    for batch_idx, (weather, forecast_day, actual_day, exo_day, price_actual_day, _dates) in enumerate(loader):
        weather = {c: w.to(device) for c, w in weather.items()}
        forecast_day = {c: v.to(device) for c, v in forecast_day.items()}
        actual_day = {c: v.to(device) for c, v in actual_day.items()}
        exo_day = {n: v.to(device) for n, v in exo_day.items()}
        price_actual_day = price_actual_day.to(device)

        out = model(weather, forecast_day, actual_day, exo_day, temperature=T)

        loss, components = loss_fn(out, price_actual_day)

        optimizer.zero_grad()
        loss.backward()
        # BUG FIX: was max_norm=float("inf") -- see GRAD_CLIP_NORM comment above.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += components["loss"].item()
        total_revenue += components["revenue"].item()
        total_price_mse += components["mse_stage2"].item()
        total_p_null += out["p_null"].mean().item()

        if BATCH_LOG_EVERY and (batch_idx + 1) % BATCH_LOG_EVERY == 0:
            info(
                f"epoch {epoch:3d}/{n_epochs} batch {batch_idx + 1:4d}/{n_batches}  "
                f"loss={components['loss'].item():.2f}  revenue(soft)={components['revenue'].item():.1f}  "
                f"mse_stage1={components['mse_stage1'].item():.4f}  "
                f"price_mse={components['mse_stage2'].item():.4f}  grad_norm={grad_norm.item():.3f}  "
                f"p_null={out['p_null'].mean().item():.3f}"
            )

    return {
        "loss": total_loss / n_batches,
        "revenue": total_revenue / n_batches,
        "price_mse": total_price_mse / n_batches,
        "p_null": total_p_null / n_batches,
    }


def run_val_epoch(model, loader, T_eval, device):
    """Realized (hard) revenue is the number that actually matters -- it uses
    a discrete dispatch decision rather than the soft/differentiable one used
    for gradients. rmse/revenue_mean here come from metrics.compute_metrics,
    which itself always dispatches via model.dispatch.hard(), never the soft
    schedule. Oracle revenue and the null-action rate aren't part of that
    shared metrics dict (oracle needs a second hard() call on the true price,
    and null-rate isn't a generic pipeline metric), so they're still tracked
    separately here.

    NOTE: the returned dict key "price_mse" actually holds an RMSE value
    (val_rmse_stage2 / n_val, from metrics.compute_metrics's rmse_stage2) --
    naming carried over unchanged from the original, flagging it here so it
    doesn't get read as an MSE downstream (e.g. in evaluate.py) by mistake.
    """
    model.eval()
    val_hard_revenue, val_oracle_revenue = 0.0, 0.0
    val_rmse_stage1, val_rmse_stage2, val_hard_null_rate, n_val = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for weather, forecast_day, actual_day, exo_day, price_actual_day, _dates in loader:
            weather = {c: w.to(device) for c, w in weather.items()}
            forecast_day = {c: v.to(device) for c, v in forecast_day.items()}
            actual_day = {c: v.to(device) for c, v in actual_day.items()}
            exo_day = {n: v.to(device) for n, v in exo_day.items()}
            price_actual_day = price_actual_day.to(device)

            out = model(weather, forecast_day, actual_day, exo_day, temperature=T_eval)
            m = compute_metrics(model, out, price_actual_day)

            power_hard = model.dispatch.hard(out["predicted_price"])
            power_oracle = model.dispatch.hard(price_actual_day)

            val_hard_revenue += m["revenue_mean"].item()
            val_oracle_revenue += (power_oracle * price_actual_day).sum(dim=-1).mean().item()
            val_rmse_stage1 += m["rmse_stage1"].item()
            val_rmse_stage2 += m["rmse_stage2"].item()
            val_hard_null_rate += (power_hard.abs().sum(dim=-1) == 0).float().mean().item()
            n_val += 1

    if n_val == 0:
        warn("Validation loader produced 0 batches -- val metrics are undefined")
        return {"hard_revenue": float("nan"), "oracle_revenue": float("nan"),
                "rmse_stage1": float("nan"), "price_mse": float("nan"),
                "hard_null_rate": float("nan"), "capture_ratio": float("nan")}

    hard_revenue = val_hard_revenue / n_val
    oracle_revenue = val_oracle_revenue / n_val
    capture_ratio = hard_revenue / oracle_revenue if oracle_revenue else float("nan")
    return {
        "hard_revenue": hard_revenue,
        "oracle_revenue": oracle_revenue,
        "rmse_stage1": val_rmse_stage1 / n_val,
        "price_mse": val_rmse_stage2 / n_val,
        "hard_null_rate": val_hard_null_rate / n_val,
        "capture_ratio": capture_ratio,
    }


def main():
    info("=" * 60)
    info("STARTING TRAINING")
    info("=" * 60)

    if SEED is not None:
        torch.manual_seed(SEED)
        info(f"Seed set: {SEED}")

    train_ds, val_ds, residual_stats = build_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_days)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_days)
    info(f"DataLoaders ready: batch_size={BATCH_SIZE}, "
         f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = S123NN(config, residual_stats).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"Model built: {type(model).__name__}, {n_params:,} trainable parameters")

    load_pretrained_weights(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # alpha/beta and gamma are overwritten every epoch by the warm-up
    # schedule below (see the comment near the top of this file).
    # normalize=True means the three loss components are each divided by a
    # running EMA of their own magnitude before being combined -- see
    # loss.py for why that matters.
    loss_fn = LossFn(alpha=LAMBDA_START, beta=LAMBDA_START, gamma=GAMMA_START, normalize=True)
    info(f"Optimizer: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})")
    info(f"Schedule: T {T0} -> {T_MIN} over {N_EPOCHS} epochs; "
         f"alpha/beta {LAMBDA_START} -> floor={ALPHA_BETA_FLOOR:.3f}, "
         f"gamma {GAMMA_START} -> {GAMMA_END}, both over first {WARMUP_FRACTION:.0%} of epochs")
    info(f"Grad clip max_norm={GRAD_CLIP_NORM}")

    best_val_revenue = float("-inf")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        T = temperature_schedule(epoch, N_EPOCHS, t0=T0, t_min=T_MIN)

        frac = min(epoch / max(N_EPOCHS * WARMUP_FRACTION, 1), 1.0)
        lam = LAMBDA_START + (LAMBDA_END - LAMBDA_START) * frac
        lam = max(lam, ALPHA_BETA_FLOOR)
        gamma = GAMMA_START + (GAMMA_END - GAMMA_START) * frac

        info(f"--- epoch {epoch:3d}/{N_EPOCHS} start (T={T:.2f}, alpha=beta={lam:.3f}, gamma={gamma:.3f}) ---")

        train_stats = run_train_epoch(model, train_loader, optimizer, loss_fn, T, lam, gamma, device, epoch, N_EPOCHS)
        val_stats = run_val_epoch(model, val_loader, T_MIN, device)

        epoch_time = time.time() - epoch_start

        info(
            f"epoch {epoch:3d} done in {epoch_time:5.1f}s | "
            f"train: loss={train_stats['loss']:.2f} revenue(soft)={train_stats['revenue']:.1f} "
            f"price_mse={train_stats['price_mse']:.4f} p_null={train_stats['p_null']:.3f}"
        )
        info(
            f"epoch {epoch:3d} val: revenue(hard)={val_stats['hard_revenue']:.1f} "
            f"oracle={val_stats['oracle_revenue']:.1f} capture_ratio={val_stats['capture_ratio']:.3f} "
            f"rmse_stage1={val_stats['rmse_stage1']:.4f} price_rmse={val_stats['price_mse']:.4f} "
            f"null_rate={val_stats['hard_null_rate']:.3f}"
        )

        if val_stats["hard_revenue"] > best_val_revenue:
            improvement = val_stats["hard_revenue"] - best_val_revenue if best_val_revenue != float("-inf") else None
            best_val_revenue = val_stats["hard_revenue"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    # ADDED: optimizer state wasn't being saved, so resuming
                    # from a checkpoint previously meant restarting AdamW's
                    # moment estimates from zero. Doesn't change training
                    # behavior, just makes the checkpoint actually resumable.
                    "optimizer_state_dict": optimizer.state_dict(),
                    "residual_stats": residual_stats,
                    "epoch": epoch,
                    "val_hard_revenue": val_stats["hard_revenue"],
                    "val_oracle_revenue": val_stats["oracle_revenue"],
                },
                CHECKPOINT_PATH,
            )
            msg = f"NEW BEST checkpoint (capture_ratio={val_stats['capture_ratio']:.3f})"
            if improvement is not None:
                msg += f", +{improvement:.1f} revenue vs previous best"
            msg += f" -> saved to {CHECKPOINT_PATH}"
            info(msg)
        else:
            info(f"No improvement (best so far: {best_val_revenue:.1f})")

    info("=" * 60)
    info(f"TRAINING COMPLETE. Best val hard revenue: {best_val_revenue:.1f}")
    info(f"Checkpoint saved at: {CHECKPOINT_PATH}")
    info("Next: run evaluate.py to score this checkpoint and dump price/production/revenue arrays.")
    info("=" * 60)


if __name__ == "__main__":
    main()