"""
Composite end-to-end loss for the Stage 1/2/3 pipeline.

Combines:
    - Stage 1 residual MSE (weather -> production residual correction)
    - Stage 2 price MSE
    - Stage 3 (negative) soft revenue

Why a class instead of a plain function
----------------------------------------
mse_stage1, mse_stage2 and revenue typically live on very different scales
(e.g. ~0.1, ~5, ~500-5000). With fixed alpha/beta/gamma weights, whichever
term has the largest raw magnitude will dominate the gradient almost
entirely, regardless of how alpha/beta/gamma are set -- the loss will
effectively just be "maximize revenue" and Stage 1/2 quality will barely
move.

`LossFn` keeps an exponential moving average (EMA) of the magnitude of each
component and divides by it before combining them. That turns alpha, beta,
gamma into true *relative* weights: alpha=beta=gamma=1 means "care about
these three things equally", independent of their raw units. Set
`normalize=False` to recover the plain, unnormalized weighted sum from the
original spec.
"""

import torch
import torch.nn.functional as F


class LossFn:
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.2,
        gamma: float = 1.0,
        normalize: bool = True,
        ema_decay: float = 0.98,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        alpha, beta, gamma : relative weights for Stage 1 MSE, Stage 2 MSE,
            and (soft) revenue respectively. With normalize=True these are
            comparable regardless of the components' raw scales.
        normalize : if True (default), each component is divided by a
            running EMA of its own magnitude before being combined.
        ema_decay : decay factor for the running scale estimates. Higher
            = smoother/slower to adapt. The EMA is only updated during
            calls made with a graph attached (i.e. during training), since
            `_update_scale` reads `.detach()` values regardless -- callers
            should not call this on validation-only passes if they want the
            training-time scale untouched (see metrics.py instead, which
            never touches this state).
        eps : numerical floor so we never divide by ~0 early in training.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalize = normalize
        self.ema_decay = ema_decay
        self.eps = eps

        self._scale_mse1 = None
        self._scale_mse2 = None
        self._scale_revenue = None

    def _update_scale(self, current: torch.Tensor, running):
        value = current.detach().abs().clamp_min(self.eps)
        if running is None:
            return value
        return self.ema_decay * running + (1.0 - self.ema_decay) * value

    def __call__(self, outputs, target_price):
        """
        Parameters
        ----------
        outputs : dict
            Output of Model.forward() (architecture.py), containing
            residual_preds_norm, residual_targets_norm, predicted_price,
            power_soft.
        target_price : (B, T)
            Ground-truth electricity price.

        Returns
        -------
        loss : scalar (attached to the autograd graph)
        components : dict of detached scalars, for logging
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

        # -------------------------
        # Stage 2
        # -------------------------
        mse_stage2 = F.mse_loss(
            outputs["predicted_price"],
            target_price,
        )

        # -------------------------
        # Stage 3
        # -------------------------
        revenue = torch.einsum(
            "bt,bt->b",
            outputs["power_soft"],
            target_price,
        )
        mean_revenue = revenue.mean()

        # -------------------------
        # Combine (with optional adaptive scale normalization)
        # -------------------------
        if self.normalize:
            self._scale_mse1 = self._update_scale(mse_stage1, self._scale_mse1)
            self._scale_mse2 = self._update_scale(mse_stage2, self._scale_mse2)
            self._scale_revenue = self._update_scale(mean_revenue, self._scale_revenue)

            norm_mse1 = mse_stage1 / self._scale_mse1
            norm_mse2 = mse_stage2 / self._scale_mse2
            norm_revenue = mean_revenue / self._scale_revenue
        else:
            norm_mse1, norm_mse2, norm_revenue = mse_stage1, mse_stage2, mean_revenue

        loss = (
            self.alpha * norm_mse1
            + self.beta * norm_mse2
            - self.gamma * norm_revenue
        )

        return loss, {
            "loss": loss.detach(),
            "mse_stage1": mse_stage1.detach(),
            "mse_stage2": mse_stage2.detach(),
            "revenue": mean_revenue.detach(),
        }
