"""Standalone high-frequency noise module extracted from NoiseDecoupling."""

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.model(x) + self.conv(x)


class _ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
        )

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1).flatten(1))
        peak = self.mlp(F.adaptive_max_pool2d(x, 1).flatten(1))
        return (avg + peak).sigmoid()[:, :, None, None]


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        pooled = torch.cat((x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values), dim=1)
        return self.conv(pooled).sigmoid()


class _CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = _ChannelAttention(channels)
        self.spatial_att = _SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        return x * self.spatial_att(x)


class HFNM(nn.Module):
    """Map RGB reflectance (B, 3, H, W) to a fused feature (B, 1, H, W)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.res_block1 = _ResidualBlock(32, 64)
        self.res_block2 = _ResidualBlock(64, 64)
        self.attention = _CBAM(64)
        self.conv_noise = nn.Conv2d(64, 1, 1)
        self.conv_fuse = nn.Conv2d(2, 1, 1)

        coords = torch.arange(5, dtype=torch.float32) - 2
        squared = coords.square()
        kernel = torch.exp(-(squared[:, None] + squared[None, :]) / (2 * 1.5 ** 2))
        self.register_buffer("gaussian_kernel", (kernel / kernel.sum())[None, None], persistent=False)

    def _high_frequency(self, gray):
        gray = gray.float()
        residual = gray - F.conv2d(gray, self.gaussian_kernel.float(), padding=2)
        flat = residual.flatten(1)
        low = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1, 1)
        high = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1)
        span = high - low
        # 分位点重合时沿用原始残差，避免除零。
        scale = torch.where(span == 0, torch.ones_like(span), span)
        return torch.where(span == 0, residual, ((residual - low) / scale).clamp(0, 1))

    def forward(self, reflectance):
        r, g, b = reflectance.unbind(dim=1)
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1)
        high_frequency = self._high_frequency(gray).to(gray.dtype)
        x = F.relu(self.conv1(gray))
        x = self.res_block2(self.res_block1(x))
        attention = self.conv_noise(self.attention(x)).sigmoid()
        return self.conv_fuse(torch.cat((high_frequency, attention), dim=1))
