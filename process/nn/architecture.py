from types import SimpleNamespace

import torch
import torch.nn as nn

from modules.S1_model import Model as S1NN
from modules.S2_model import Model as S2NN
from modules.S3_model import Model as S3NN


class Model(nn.Module):
    """Wires the three stages into one differentiable graph so gradients from
    actual revenue flow back through Stage 2 into every Stage 1 branch.
    """

    def __init__(self, config: SimpleNamespace, residual_stats: dict[str, tuple[float, float]]):
        super().__init__()

        self.config = config
        self.commodities = list(config.commodities)
        self.exo_raw_paths = list(config.exo_raw_paths)

        self.stage1 = nn.ModuleDict({c: S1NN(config.stage1) for c in self.commodities})

        for c in self.commodities:
            mean, std = residual_stats[c]
            self.register_buffer(f"_res_mean_{c}", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer(f"_res_std_{c}", torch.tensor(std, dtype=torch.float32))

        self.stage2 = S2NN(config.stage2)
        self.dispatch = S3NN(config.stage3)

    def _res_mean(self, c: str) -> torch.Tensor:
        return getattr(self, f"_res_mean_{c}")

    def _res_std(self, c: str) -> torch.Tensor:
        return getattr(self, f"_res_std_{c}")

    def forward(self, weather, forecast_day, actual_day, exo_day, temperature: float):
        residual_preds_norm, residual_targets_norm, corrected = {}, {}, {}

        for c in self.commodities:
            pred_norm = self.stage1[c](weather[c])                     # (B, slots_per_day)
            residual_preds_norm[c] = pred_norm

            mean, std = self._res_mean(c), self._res_std(c)
            corrected[c] = forecast_day[c] + pred_norm * std + mean    # (B, slots_per_day), real units

            target_real = actual_day[c] - forecast_day[c]
            residual_targets_norm[c] = (target_real - mean) / std

        channels = [corrected[c] for c in self.commodities] + [exo_day[n] for n in self.exo_raw_paths]
        x_enc = torch.stack(channels, dim=-1)                          # (B, slots_per_day, enc_in)

        predicted_price = self.stage2(x_enc, None, None, None)
        if predicted_price.dim() == 3:                                 # safety: squeeze trailing c_out=1
            predicted_price = predicted_price.squeeze(-1)

        power_soft, p_null = self.dispatch(predicted_price, temperature)

        return {
            "predicted_price": predicted_price,
            "power_soft": power_soft,
            "p_null": p_null,
            "residual_preds_norm": residual_preds_norm,
            "residual_targets_norm": residual_targets_norm,
        }