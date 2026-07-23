from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, config: SimpleNamespace):
        super().__init__()

        self.slots_per_day = config.slots_per_day
        self.window_length = config.rolling_window_length
        self.charge_power = float(config.charge_power)
        self.discharge_power = float(config.discharge_power)
        self.min_gap = int(getattr(config, "min_gap", 8))

        charge_starts = list(range(config.charge_start_min, config.charge_start_max + 1))
        discharge_starts = list(range(config.discharge_start_min, config.discharge_start_max + 1))

        if max(charge_starts) + self.window_length > self.slots_per_day:
            raise ValueError("charge_start_max + rolling_window_length exceeds slots_per_day")
        if max(discharge_starts) + self.window_length > self.slots_per_day:
            raise ValueError("discharge_start_max + rolling_window_length exceeds slots_per_day")

        # -----------------------------------------------------------------
        # Build every VALID (charge_start, discharge_start) pair.
        #
        # A pair is valid iff:
        #   - the charge window fits inside the day
        #   - the discharge window fits inside the day
        #   - the discharge window starts at least `min_gap` timesteps after
        #     the charge window ENDS (c + window_length), which is exactly
        #     the "discharge at least 8 timesteps after charge" constraint
        #     from the original Stage 3 optimizer.
        #
        # Each candidate therefore contains a charge AND a discharge by
        # construction, so both constraints are satisfied automatically:
        #   - there can never be a charge without a matching discharge
        #   - there can never be a discharge less than min_gap after charge
        #   - there can never be a day that ends still charged (the only
        #     way to end up with no discharge is the explicit "null" action)
        # -----------------------------------------------------------------
        valid_pairs = []
        for c in charge_starts:
            if c + self.window_length > self.slots_per_day:
                continue
            for d in discharge_starts:
                if d + self.window_length > self.slots_per_day:
                    continue
                if d < c + self.window_length + self.min_gap:
                    continue
                valid_pairs.append((c, d))

        if len(valid_pairs) == 0:
            raise ValueError(
                "No valid (charge, discharge) pairs found — check start "
                "ranges, window_length and min_gap."
            )

        self.n_pairs = len(valid_pairs)
        self.valid_pairs = valid_pairs  # kept for inspection/debugging

        # One mask per valid pair, each already scaled by the corresponding
        # power, so scores/dispatch reconstruction is a simple sum.
        pair_charge_masks = torch.zeros(self.n_pairs, self.slots_per_day)
        pair_discharge_masks = torch.zeros(self.n_pairs, self.slots_per_day)

        for i, (c, d) in enumerate(valid_pairs):
            pair_charge_masks[i, c: c + self.window_length] = self.charge_power
            pair_discharge_masks[i, d: d + self.window_length] = self.discharge_power

        self.register_buffer("pair_charge_masks", pair_charge_masks)
        self.register_buffer("pair_discharge_masks", pair_discharge_masks)

    def _pair_scores(self, price: torch.Tensor) -> torch.Tensor:
        """(B, n_pairs) score for every valid (charge, discharge) pair."""
        # masks are already power-scaled, so the score is directly
        # power * sum(price) for both legs of the pair, summed together.
        charge_value = torch.einsum("nt,bt->bn", self.pair_charge_masks, price)
        discharge_value = torch.einsum("nt,bt->bn", self.pair_discharge_masks, price)
        return charge_value + discharge_value

    def forward(self, price: torch.Tensor, temperature: float):
        """
        price:       (B, slots_per_day) Stage 2 predicted price for the day,
                     still attached to the autograd graph during end-to-end training.
        temperature: scalar (python float or 0-d tensor) annealed externally by the
                     training loop (e.g. from config.training.T0 down to T_min).

        Returns:
            power_soft: (B, slots_per_day) soft dispatch schedule
                        (>0 discharge/sell, <0 charge/buy)
            p_null:     (B,) softmax weight assigned to the "do nothing" action
        """
        B, T = price.shape

        pair_scores = self._pair_scores(price)  # (B, n_pairs)
        null_scores = torch.zeros(B, 1, device=price.device, dtype=price.dtype)

        scores = torch.cat([pair_scores, null_scores], dim=1)
        weights = F.softmax(scores / temperature, dim=1)

        w_pairs = weights[:, :-1]
        p_null = weights[:, -1]

        charge = torch.einsum("bn,nt->bt", w_pairs, self.pair_charge_masks)
        discharge = torch.einsum("bn,nt->bt", w_pairs, self.pair_discharge_masks)

        power_soft = charge + discharge

        return power_soft, p_null

    @torch.no_grad()
    def hard(self, price: torch.Tensor) -> torch.Tensor:
        """Argmax (discrete, non-differentiable) dispatch — the schedule an
        operator would actually run. Used at validation/inference time to
        measure realized revenue, since power_soft is a training-time
        relaxation and doesn't correspond to a schedule you can dispatch.

        price: (B, slots_per_day)
        Returns: power (B, slots_per_day), same sign convention as forward().
        """
        B, T = price.shape

        pair_scores = self._pair_scores(price)  # (B, n_pairs)
        null_scores = torch.zeros(B, 1, device=price.device, dtype=price.dtype)

        scores = torch.cat([pair_scores, null_scores], dim=1)
        best_idx = scores.argmax(dim=1)  # (B,)

        is_null = best_idx == self.n_pairs
        pair_idx = best_idx.clamp(max=self.n_pairs - 1)

        power = self.pair_charge_masks[pair_idx] + self.pair_discharge_masks[pair_idx]
        power = torch.where(is_null.unsqueeze(1), torch.zeros_like(power), power)

        return power
