"""
models.py
=========
Two-stage energy-price forecasting network.

STAGE 1  (per energy font e in {solar, wind, hydro, non_marketized, tie_line})
    meteo_e(t)  --[CNN/U-Net]-->  EfficiencyMap_e(t)   in [0,1], shape (H,W)
                                        x
                CapacityMap_e (learnable, time-invariant), shape (H,W)
                                        =
                        ProductionMap_e(t), shape (H,W)
                                        |  sum over (H,W)
                                        v
                        AggregatedProduction_e(t)   shape (T,)

    "solar", "wind", "hydro" use SpatialEnergyBranch (CapacityMap is a 2-D
    field learned via a small U-Net-fed factorization).

    "non_marketized" and "tie_line" have no meaningful spatial meteo driver,
    so NonSpatialEnergyBranch reuses the same capacity x efficiency
    factorization but with a scalar capacity and an MLP efficiency.

STAGE 2
    All five AggregatedProduction_e(t) timeseries are concatenated with
    boundary conditions (demand forecast, fuel price, calendar features...)
    and fed to a temporal model (TCN by default, MLP baseline available)
    that outputs the predicted price(t).

The whole thing is one differentiable graph: gradients from the price loss
flow back through Stage 2 into the Stage-1 capacity maps and efficiency
networks, on top of the direct Stage-1 production loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SPATIAL_TYPES = ["solar", "wind", "hydro"]
NONSPATIAL_TYPES = ["non_marketized", "tie_line"]
ALL_ENERGY_TYPES = SPATIAL_TYPES + NONSPATIAL_TYPES


# ---------------------------------------------------------------------------
# Stage 1 - spatial branch
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU, the basic U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetEfficiency(nn.Module):
    """Small U-Net: meteo channels (C,H,W) at a single timestep -> efficiency
    map in [0,1] of shape (1,H,W). H and W must be divisible by 4.
    Swap this for a ResNet encoder-decoder if you want a deeper model; the
    interface (forward returns a sigmoid map of shape (B,1,H,W)) is what the
    rest of the pipeline depends on.
    """

    def __init__(self, in_channels: int, base_ch: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))  # (B,1,H,W)


class CapacityMap(nn.Module):
    """Learnable, time-invariant installed-capacity distribution over the
    grid for one energy font. Softplus keeps it positive; it is *not* a
    function of meteo, only of position, matching the "statical" capacity
    map requested.
    """

    def __init__(self, H: int, W: int, init_scale: float = 1.0):
        super().__init__()
        self.raw = nn.Parameter(torch.randn(1, 1, H, W) * 0.1)
        self.init_scale = init_scale

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw) * self.init_scale  # (1,1,H,W) > 0

    def total_variation(self) -> torch.Tensor:
        """Spatial-smoothness regularizer. Aggregated production alone only
        weakly constrains the *shape* of the capacity map (many maps with
        the same spatial sum reproduce the target equally well), so this
        keeps the learned map physically plausible instead of noisy.
        """
        m = self.forward()
        dh = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
        dw = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
        return dh + dw


class SpatialEnergyBranch(nn.Module):
    """CapacityMap x EfficiencyMap(t) -> production map(t) -> aggregated
    production(t), for one spatially-resolved energy font.
    """

    def __init__(self, in_channels: int, H: int, W: int, base_ch: int = 16,
                 area_weights: torch.Tensor | None = None):
        super().__init__()
        self.efficiency_net = UNetEfficiency(in_channels, base_ch)
        self.capacity_map = CapacityMap(H, W)
        if area_weights is None:
            area_weights = torch.ones(1, 1, H, W)
        self.register_buffer("area_weights", area_weights)

    def forward(self, meteo):

        B, T, C, H, W = meteo.shape
        x = meteo.reshape(B * T, C, H, W)
        eff = self.efficiency_net(x)
        eff = eff.reshape(B, T, 4, H, W)
        cap = self.capacity_map().unsqueeze(1)
        weighted_area = self.area_weights.unsqueeze(1)
        prod_map = cap * eff * weighted_area
        agg = prod_map.sum(dim=(-1, -2))
        return (
            agg,
            prod_map,
            cap.squeeze()
        )


# ---------------------------------------------------------------------------
# Stage 1 - non-spatial branch (non_marketized, tie_line interchange)
# ---------------------------------------------------------------------------

class NonSpatialEnergyBranch(nn.Module):
    """Same capacity x efficiency idea, but for energy fonts without a
    meaningful (lon, lat) meteo driver: capacity is a single learnable
    scalar and efficiency(t) comes from an MLP over exogenous features
    (schedule, price-differential proxies, calendar, etc.).
    """

    def __init__(self, in_features, hidden=32):

        super().__init__()

        self.capacity_raw = nn.Parameter(torch.tensor(0.0))

        self.efficiency_net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)
        )

    def forward(self, features):

        cap = F.softplus(self.capacity_raw)
        eff = torch.sigmoid(self.efficiency_net(features))
        agg = cap * eff
        return agg, eff, cap


# ---------------------------------------------------------------------------
# Stage 2 - price heads
# ---------------------------------------------------------------------------

class CausalConv1dBlock(nn.Module):
    """Dilated causal 1-D conv residual block (TCN building block)."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.pad != 0:
            out = out[..., :-self.pad]  # trim right side -> causal, same length
        out = F.relu(self.norm(out))
        out = self.dropout(out)
        return out + x


class TCNPriceHead(nn.Module):
    """Temporal Convolutional Network: stacked dilated causal convolutions
    give it a growing receptive field over past timesteps, so price(t) can
    depend on production/demand history, not just the instantaneous values.
    This is the "takes autocorrelation into account" option.
    """

    def __init__(self, in_features: int, hidden: int = 64, n_blocks: int = 4,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalConv1dBlock(hidden, dilation=2 ** i, kernel_size=kernel_size,
                               dropout=dropout)
            for i in range(n_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,F) -> price (B,T)"""
        x = x.transpose(1, 2)          # (B,F,T)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x).squeeze(1)  # (B,T)



# ---------------------------------------------------------------------------
# Full NN
# ---------------------------------------------------------------------------

class EnergyPriceModel(nn.Module):
    def __init__(self, spatial_in_channels: dict[str, int], H: int, W: int,
                 nonspatial_in_features: dict[str, int],
                 n_boundary_features: int,
                 area_weights: torch.Tensor | None = None):
        super().__init__()
        self.spatial_branches = nn.ModuleDict({
            name: SpatialEnergyBranch(spatial_in_channels[name], H, W,
                                       area_weights=area_weights)
            for name in spatial_in_channels
        })
        self.nonspatial_branches = nn.ModuleDict({
            name: NonSpatialEnergyBranch(nonspatial_in_features[name])
            for name in nonspatial_in_features
        })
        n_energy = len(spatial_in_channels) + len(nonspatial_in_features)
        stage2_in = n_energy + n_boundary_features
        self.price_head = TCNPriceHead(stage2_in)
        

    def forward(self, meteo: dict[str, torch.Tensor],
            nonspatial_features: dict[str, torch.Tensor],
            boundary_conditions: torch.Tensor) -> dict:

        productions, capacity_maps = {}, {}
        tv_reg = 0.0


        for name, branch in self.spatial_branches.items():
            agg, _prod_map, cap = branch(meteo[name])
            productions[name] = agg                  # (B,T,4)
            capacity_maps[name] = cap
            tv_reg = tv_reg + branch.capacity_map.total_variation()


        for name, branch in self.nonspatial_branches.items():
            agg, _eff, cap = branch(nonspatial_features[name])
            productions[name] = agg                  # (B,T,4)
            capacity_maps[name] = cap

        prod_stack = torch.stack(
            [productions[k] for k in ALL_ENERGY_TYPES],
            dim=-1
        )  
        B, T, Q, F = prod_stack.shape  

        prod_stack = prod_stack.reshape(B, T * Q, F)  

        B, T, Q, Fbc = boundary_conditions.shape
        boundary_conditions = boundary_conditions.reshape(B, T * Q, Fbc)

        stage2_input = torch.cat(
            [prod_stack, boundary_conditions],
            dim=-1
        ) 
        price_pred = self.price_head(stage2_input)  # (B, 4T)

        return {
            "productions": productions,      # dict of (B,T,4)
            "capacity_maps": capacity_maps,
            "price": price_pred,             # (B,4T)
            "tv_reg": tv_reg,
        }


def compute_loss(outputs: dict, targets: dict, tv_weight: float = 1e-3,
                  stage1_weight: float = 1.0, stage2_weight: float = 1.0) -> dict:
    """targets = {"productions": {name: (B,T)}, "price": (B,T)}"""
    stage1_loss = sum(
        F.mse_loss(outputs["productions"][k], targets["productions"][k])
        for k in ALL_ENERGY_TYPES
    )
    
    stage2_loss = F.smooth_l1_loss(outputs["price"], targets["price"])
    tv_loss = outputs["tv_reg"]
    total = stage1_weight * stage1_loss + stage2_weight * stage2_loss + tv_weight * tv_loss
    return {"total": total, "stage1": stage1_loss, "stage2": stage2_loss, "tv": tv_loss}
