from __future__ import annotations
import torch
from torch import nn

class ResidualMLPClassifier(nn.Module):

    def __init__(self, input_dim, width=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, width)) for _ in range(2)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 10))

    def forward(self, x, nonce=None):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)

class DeepSetsClassifier(nn.Module):

    def __init__(self, nonce_dim=0, width=16):
        super().__init__()
        self.nonce_dim = nonce_dim
        self.phi = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, 2 * width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(6 * width + 4 + nonce_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 10))

    def forward(self, x, nonce=None):
        tokens = torch.stack([x, torch.abs(x), x * x], dim=-1)
        h = self.phi(tokens)
        summary = torch.cat([h.mean(1), h.amax(1), h.std(1, unbiased=False), x.mean(1, keepdim=True), x.square().mean(1, keepdim=True), x.pow(4).mean(1, keepdim=True), x.pow(6).mean(1, keepdim=True)], dim=1)
        if self.nonce_dim:
            summary = torch.cat([summary, nonce], dim=1)
        return self.rho(summary)

class ConvReconstructor(nn.Module):

    def __init__(self, input_dim, channels=32):
        super().__init__()
        c = channels
        self.fc = nn.Sequential(nn.Linear(input_dim, 1024), nn.SiLU(), nn.Linear(1024, c * 4 * 4), nn.SiLU())
        self.res = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))
        self.head = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c, c // 2, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c // 2, c // 4, 3, padding=1), nn.SiLU(), nn.Conv2d(c // 4, 3, 3, padding=1), nn.Sigmoid())
        self.channels = c

    def forward(self, x, nonce=None):
        h = self.fc(x).reshape(-1, self.channels, 4, 4)
        return self.head(h + self.res(h))
