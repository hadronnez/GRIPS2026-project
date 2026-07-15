
import torch
import torch.nn as nn

# ============================================================
# MODEL
#
# Same computation as before, reordered for speed. This is
# NOT an approximation -- it produces mathematically identical
# output, just much cheaper to compute.
#
# WHY THE REORDER IS EXACT:
# temporal_up1/temporal_up2 are per-pixel affine maps (kernel
# size 1 in H,W -- no cross-pixel mixing) with weights shared
# across every spatial location. For any affine f applied
# identically to every pixel, mean_pixels(f(x)) == f(mean_pixels(x)).
# So pooling H,W away right after the encoder, THEN doing the
# temporal upsampling on the pooled (B,32,12) sequence, gives
# the exact same numbers as upsampling first at full spatial
# resolution and pooling afterward.
#
# The old order built and back-propped through a (B,32,48,104,225)
# tensor -- ~36M values per sample -- just to average it away
# at the end. That tensor, and the two ConvTranspose3d layers
# that produced it at full spatial resolution, were the
# overwhelming majority of the compute and memory cost. None
# of it changed the final answer.
#
# Only the encoder genuinely needs full spatial resolution --
# its 3x3 spatial kernel is the only place pixels actually mix
# with their neighbors, which is real information (e.g. local
# cloud patterns) that pooling first would destroy.
# ============================================================



class ResidualNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.weather_encoder = nn.Sequential(
            nn.Conv3d(7, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
        )

        # learn 12h -> 24h, and 24h -> 48 quarter-hours.
        # Now 1D: operates on the pooled (B,32,T) sequence, not
        # on (B,32,T,H,W). Same weights-per-timestep idea as
        # before, at a tiny fraction of the compute.
        self.temporal_up1 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2)
        self.temporal_up2 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2)

        # scalar-per-timestep head, operating on (B, 32, 48)
        self.head = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, weather):
        # incoming: B,T,C,H,W (T=12) -> Conv3D expects B,C,T,H,W
        weather = weather.permute(0, 2, 1, 3, 4)

        x = self.weather_encoder(weather)
        # B,32,12,H,W -- last point where spatial resolution matters

        x = x.mean(dim=(3, 4))
        # B,32,12 -- pool NOW, while the tensor is still small

        x = self.temporal_up1(x)
        x = self.temporal_up2(x)
        # B,32,48 -- upsampling on a tiny sequence, not a spatial grid

        residual_pred = self.head(x).squeeze(1)
        # B,48

        return residual_pred
