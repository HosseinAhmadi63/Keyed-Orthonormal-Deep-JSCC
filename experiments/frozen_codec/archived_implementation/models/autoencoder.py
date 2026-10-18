import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, c: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h

class DownBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.n = nn.GroupNorm(8, c_in)
        self.c = nn.Conv2d(c_in, c_out, 3, 2, 1)
        self.r = ResBlock(c_out)

    def forward(self, x):
        x = self.c(F.silu(self.n(x)))
        return self.r(x)

class UpBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.r = ResBlock(c_out)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.c(x)
        return self.r(x)

class Encoder(nn.Module):

    def __init__(self, latent_n: int):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, 1, 1)
        self.b0 = ResBlock(64)
        self.d1 = DownBlock(64, 128)
        self.d2 = DownBlock(128, 256)
        self.d3 = DownBlock(256, 256)
        self.n = nn.GroupNorm(8, 256)
        self.fc = nn.Linear(256 * 4 * 4, latent_n)
        self.ln = nn.LayerNorm(latent_n)

    def forward(self, x):
        h = self.b0(self.stem(x))
        h = self.d1(h)
        h = self.d2(h)
        h = self.d3(h)
        h = F.silu(self.n(h))
        h = h.flatten(1)
        z = self.ln(self.fc(h))
        return z

class Decoder(nn.Module):

    def __init__(self, latent_n: int):
        super().__init__()
        self.fc = nn.Linear(latent_n, 256 * 4 * 4)
        self.n = nn.GroupNorm(8, 256)
        self.r0 = ResBlock(256)
        self.u1 = UpBlock(256, 256)
        self.u2 = UpBlock(256, 128)
        self.u3 = UpBlock(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], 256, 4, 4)
        h = self.r0(F.silu(self.n(h)))
        h = self.u1(h)
        h = self.u2(h)
        h = self.u3(h)
        x = torch.sigmoid(self.out(h))
        return x
