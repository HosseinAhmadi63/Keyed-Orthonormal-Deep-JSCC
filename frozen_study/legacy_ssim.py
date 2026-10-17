from typing import Dict
import torch
import torch.nn.functional as F
_ssim_window_cache: Dict[tuple, torch.Tensor] = {}

@torch.no_grad()
def _gaussian_window(window_size: int, sigma: float, device, dtype):
    key = (window_size, float(sigma), device.type, device.index, str(dtype))
    if key in _ssim_window_cache:
        return _ssim_window_cache[key]
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.view(1, 1, -1)
    window_2d = window_1d.transpose(-1, -2) @ window_1d
    window_2d = window_2d.to(dtype=torch.float32)
    _ssim_window_cache[key] = window_2d
    return window_2d

@torch.no_grad()
def ssim_per_image(img1: torch.Tensor, img2: torch.Tensor, window_size: int=11, sigma: float=1.5, eps: float=1e-08) -> torch.Tensor:
    img1 = img1.float()
    img2 = img2.float()
    device = img1.device
    B, C, _, _ = img1.shape
    window = _gaussian_window(window_size, sigma, device, img1.dtype).view(1, 1, window_size, window_size)
    window = window.repeat(C, 1, 1, 1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu12
    sigma1_sq = sigma1_sq.clamp_min(0.0)
    sigma2_sq = sigma2_sq.clamp_min(0.0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    ssim_map = num / den
    ssim_val = ssim_map.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    return ssim_val

@torch.no_grad()
def _ssim_cs_per_image(img1: torch.Tensor, img2: torch.Tensor, window_size: int=11, sigma: float=1.5, eps: float=1e-08):
    img1 = img1.float()
    img2 = img2.float()
    device = img1.device
    B, C, _, _ = img1.shape
    window = _gaussian_window(window_size, sigma, device, img1.dtype).view(1, 1, window_size, window_size)
    window = window.repeat(C, 1, 1, 1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu12
    sigma1_sq = sigma1_sq.clamp_min(0.0)
    sigma2_sq = sigma2_sq.clamp_min(0.0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    l = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1 + eps)
    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2 + eps)
    ssim = (l * cs).mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    cs = cs.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    return (ssim, cs)

@torch.no_grad()
def ms_ssim_mean(img1: torch.Tensor, img2: torch.Tensor, scales: int=4, eps: float=1e-08) -> torch.Tensor:
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=img1.device, dtype=torch.float32)
    x1 = img1.float()
    x2 = img2.float()
    mcs = []
    ssim_last = None
    for i in range(scales):
        ssim_i, cs_i = _ssim_cs_per_image(x1, x2, eps=eps)
        if i < scales - 1:
            mcs.append(cs_i)
            x1 = F.avg_pool2d(x1, kernel_size=2, stride=2)
            x2 = F.avg_pool2d(x2, kernel_size=2, stride=2)
        else:
            ssim_last = ssim_i
    ms = torch.ones_like(ssim_last)
    for i in range(scales - 1):
        ms = ms * mcs[i] ** weights[i]
    ms = ms * ssim_last ** weights[-1]
    return ms.mean().clamp(0.0, 1.0)
