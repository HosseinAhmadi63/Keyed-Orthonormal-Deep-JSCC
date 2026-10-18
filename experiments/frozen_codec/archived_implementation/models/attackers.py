import torch
import torch.nn as nn
from models.autoencoder import Decoder

class NonceEmbed(nn.Module):

    def __init__(self, out_dim: int=32, hidden: int=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, out_dim), nn.SiLU())

    def forward(self, nonce_i64: torch.Tensor) -> torch.Tensor:
        x = nonce_i64
        m = torch.tensor(65535, device=x.device, dtype=torch.int64)
        c0 = (x & m).to(torch.float32)
        c1 = (x >> 16 & m).to(torch.float32)
        c2 = (x >> 32 & m).to(torch.float32)
        c3 = (x >> 48 & m).to(torch.float32)
        chunks = torch.stack([c0, c1, c2, c3], dim=-1)
        chunks = chunks / 65535.0 * 2.0 - 1.0
        return self.mlp(chunks)

class EveRecon(nn.Module):

    def __init__(self, latent_n: int, nonce_emb_dim: int=32):
        super().__init__()
        self.nonce_emb = NonceEmbed(out_dim=nonce_emb_dim)
        self.fc = nn.Sequential(nn.Linear(latent_n + nonce_emb_dim, 512), nn.SiLU(), nn.Linear(512, 256 * 4 * 4), nn.SiLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, y_cipher: torch.Tensor, nonce_i64: torch.Tensor) -> torch.Tensor:
        ne = self.nonce_emb(nonce_i64)
        h = self.fc(torch.cat([y_cipher, ne], dim=-1)).view(y_cipher.shape[0], 256, 4, 4)
        return self.deconv(h)

class EveReconNoNonce(nn.Module):

    def __init__(self, latent_n: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_n, 512), nn.SiLU(), nn.Linear(512, 256 * 4 * 4), nn.SiLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, y_cipher: torch.Tensor) -> torch.Tensor:
        h = self.fc(y_cipher).view(y_cipher.shape[0], 256, 4, 4)
        return self.deconv(h)

class EveReconStrong(nn.Module):

    def __init__(self, latent_n: int, nonce_emb_dim: int=64):
        super().__init__()
        self.nonce_emb = NonceEmbed(out_dim=nonce_emb_dim, hidden=128)
        self.fc = nn.Sequential(nn.Linear(latent_n + nonce_emb_dim, 1024), nn.SiLU(), nn.Linear(1024, 256 * 4 * 4), nn.SiLU())
        self.res = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.SiLU(), nn.Conv2d(256, 256, 3, 1, 1))
        self.deconv = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, y_cipher: torch.Tensor, nonce_i64: torch.Tensor) -> torch.Tensor:
        ne = self.nonce_emb(nonce_i64)
        h = self.fc(torch.cat([y_cipher, ne], dim=-1)).view(y_cipher.shape[0], 256, 4, 4)
        h = h + self.res(h)
        return self.deconv(h)

class InterceptClf(nn.Module):

    def __init__(self, latent_n: int, num_classes: int=10, nonce_emb_dim: int=32):
        super().__init__()
        self.nonce_emb = NonceEmbed(out_dim=nonce_emb_dim)
        self.mlp = nn.Sequential(nn.Linear(latent_n + nonce_emb_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, num_classes))

    def forward(self, y_cipher: torch.Tensor, nonce_i64: torch.Tensor) -> torch.Tensor:
        ne = self.nonce_emb(nonce_i64)
        return self.mlp(torch.cat([y_cipher, ne], dim=-1))

class EveBobStyleDecoder(nn.Module):

    def __init__(self, latent_n: int):
        super().__init__()
        self.dec = Decoder(latent_n)

    def forward(self, y_cipher: torch.Tensor) -> torch.Tensor:
        return self.dec(y_cipher)
