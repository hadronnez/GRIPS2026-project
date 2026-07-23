import torch
import torch.nn as nn

from types import SimpleNamespace


class Model(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config

        encoder_layers = []
        in_ch = config.in_channels
        for out_ch, groups in zip(config.encoder_channels, config.encoder_norm_groups):
            encoder_layers += [
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=config.encoder_kernel_size,
                    padding=config.encoder_kernel_size // 2,
                    padding_mode=config.encoder_padding_mode,
                ),
                nn.GroupNorm(num_groups=groups, num_channels=out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.weather_encoder = nn.Sequential(*encoder_layers)

        encoder_out_channels = config.encoder_channels[-1]

        self.temporal_ups = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    encoder_out_channels,
                    encoder_out_channels,
                    kernel_size=config.upsample_kernel_size,
                    stride=config.upsample_stride,
                )
                for _ in range(config.num_temporal_upsamples)
            ]
        )

        self.head = nn.Sequential(
            nn.Conv1d(
                encoder_out_channels,
                config.head_hidden_channels,
                kernel_size=config.head_kernel_size,
                padding=config.head_kernel_size // 2,
            ),
            nn.GroupNorm(num_groups=config.head_norm_groups, num_channels=config.head_hidden_channels),
            nn.GELU(),
            nn.Conv1d(config.head_hidden_channels, config.out_channels, kernel_size=1),
        )

    def forward(self, weather):
        # Input (B, T, C, H, W)
        weather = weather.permute(0, 2, 1, 3, 4)

        x = self.weather_encoder(weather)
        x = x.mean(dim=(3, 4))
        for up in self.temporal_ups:
            x = up(x)
        residual_pred = self.head(x)

        if self.config.out_channels == 1:
            residual_pred = residual_pred.squeeze(1)
        return residual_pred