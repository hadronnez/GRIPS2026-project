"""
Validation metrics for the Stage 1/2/3 pipeline.

Deliberately separate from loss.py: validation must score the *realized*
(hard, discrete) dispatch decision an operator would actually run, never
the soft/differentiable relaxation used to get gradients during training.
`power_soft` can look profitable purely because it's a weighted blend of
many candidate windows and never has to commit to one -- `model.dispatch.hard`
is the only number that reflects real, dispatchable revenue.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def metrics(model, outputs, target_price):
    """
    Parameters
    ----------
    model : the full S123NN model (needs model.dispatch.hard()).
    outputs : dict
        Output of Model.forward().
    target_price : (B, T)
        Ground-truth electricity price.

    Returns
    -------
    dict of scalar tensors.
    """

    # -------------------------
    # Stage 1
    # -------------------------
    mse_stage1 = 0.0
    for commodity in outputs["residual_preds_norm"]:
        mse_stage1 = mse_stage1 + F.mse_loss(
            outputs["residual_preds_norm"][commodity],
            outputs["residual_targets_norm"][commodity],
        )
    mse_stage1 = mse_stage1 / len(outputs["residual_preds_norm"])
    rmse_stage1 = torch.sqrt(mse_stage1)

    # -------------------------
    # Stage 2
    # -------------------------
    rmse_stage2 = torch.sqrt(
        F.mse_loss(
            outputs["predicted_price"],
            target_price,
        )
    )

    # -------------------------
    # Stage 3 -- hard dispatch only
    # -------------------------
    power = model.dispatch.hard(outputs["predicted_price"])

    revenue_per_day = torch.einsum(
        "bt,bt->b",
        power,
        target_price,
    )

    score = revenue_per_day.mean()

    return {
        "rmse_stage1": rmse_stage1,
        "rmse_stage2": rmse_stage2,
        "revenue_mean": score,
        "revenue_std": revenue_per_day.std(),
        "revenue_min": revenue_per_day.min(),
        "revenue_max": revenue_per_day.max(),
        "score": score,
    }
