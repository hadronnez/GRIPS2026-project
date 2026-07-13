
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- UNet2D ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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


class Up(nn.Module):
    """Upsample + pad-to-match + concat (225 is not divisible by 8)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    def __init__(self, in_channels=7, out_channels=4, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_ch*4, base_ch*8)
        self.up3 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up1 = Up(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out_conv(d1)          # [B, 4, 104, 225]


# ---------- Head: maps -> totals ----------
class SpatialSumHead(nn.Module):
    def __init__(self, in_channels=7, out_channels=4, base_ch=32):
        super().__init__()
        self.unet = UNet2D(in_channels, out_channels, base_ch)

    def forward(self, x):
        maps = F.softplus(self.unet(x))       # [B, 4, 104, 225], >= 0
        totals = maps.sum(dim=(2, 3))          # [B, 4]
        return totals, maps


# ---------- Regularization: sparsity (soft-L0) + compactness ----------
class SparsityCompactnessReg(nn.Module):
    def __init__(self, H, W, sigma_l0=0.05):
        super().__init__()
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
        )
        self.register_buffer("ys", ys)
        self.register_buffer("xs", xs)
        self.sigma_l0 = sigma_l0

    def soft_l0(self, maps):
        active = 1.0 - torch.exp(-(maps ** 2) / (2 * self.sigma_l0 ** 2))
        return active.sum(dim=(2, 3))               # [B, 4]

    def compactness(self, maps):
        mass = maps.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        cy = (maps * self.ys).sum(dim=(2, 3), keepdim=True) / mass
        cx = (maps * self.xs).sum(dim=(2, 3), keepdim=True) / mass
        d2 = (self.ys - cy) ** 2 + (self.xs - cx) ** 2
        var = (maps * d2).sum(dim=(2, 3)) / mass.squeeze(-1).squeeze(-1)
        return var                                    # [B, 4]

    def forward(self, maps):
        return self.soft_l0(maps).mean(), self.compactness(maps).mean()


# ----------

def build_model_and_reg(device):
    model = SpatialSumHead(in_channels=7, out_channels=4, base_ch=32).to(device)
    reg = SparsityCompactnessReg(H=104, W=225, sigma_l0=0.05).to(device)
    return model, reg


def train_step(model, reg, x, target, optimizer,
                lambda_l0, lambda_compact):
    pred, maps = model(x)
    recon_loss = F.mse_loss(pred, target)
    l0_term, spread_term = reg(maps)
    loss = recon_loss + lambda_l0 * l0_term + lambda_compact * spread_term

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "recon": recon_loss.item(),
        "l0": l0_term.item(),
        "compact": spread_term.item(),
    }


# ---------- ejemplo de loop ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, reg = build_model_and_reg(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 20
    warmup_epochs = 10
    lambda_l0_max, lambda_compact_max = 1e-3, 1e-2

    x = torch.randn(8, 7, 104, 225).to(device)          # placeholder
    target = torch.randn(8, 4).to(device)                # placeholder

    for epoch in range(n_epochs):
        warmup = min(1.0, epoch / warmup_epochs)
        lambda_l0 = lambda_l0_max * warmup
        lambda_compact = lambda_compact_max * warmup

        stats = train_step(model, reg, x, target, optimizer,
                            lambda_l0, lambda_compact)
        print(epoch, stats)