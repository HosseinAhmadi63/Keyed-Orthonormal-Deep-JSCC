import torch
import torch.nn as nn

class AWGN(nn.Module):

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        x32 = x.float()
        power = x32.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-06)
        snr = 10.0 ** (snr_db / 10.0)
        sigma = (power / snr).sqrt()
        return x + torch.randn_like(x) * sigma.to(x.dtype)
