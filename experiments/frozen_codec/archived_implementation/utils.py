import random
from typing import Optional, Dict, List
import torch
try:
    import torchvision
except Exception as e:
    raise ImportError('torchvision is required for this project.') from e

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_power_of_two(n: int) -> bool:
    return n & n - 1 == 0 and n > 0

def wrap_to_int64(x: int) -> int:
    x = x % (1 << 64)
    if x >= 1 << 63:
        x -= 1 << 64
    return int(x)

@torch.no_grad()
def make_image_grid(x: torch.Tensor, nrow: int=8) -> torch.Tensor:
    return torchvision.utils.make_grid(x, nrow=nrow)

def psnr_from_mse(mse: torch.Tensor, max_val: float=1.0, eps: float=1e-08) -> torch.Tensor:
    return 10.0 * torch.log10(max_val * max_val / (mse + eps))

def _randperm_no_fixed_points(B: int, device: torch.device) -> Optional[torch.Tensor]:
    if B < 2:
        return None
    idx = torch.arange(B, device=device)
    if B == 2:
        return idx.flip(0)
    perm = torch.randperm(B, device=device)
    for _ in range(10):
        if not (perm == idx).any():
            return perm
        perm = torch.randperm(B, device=device)
    return (idx + 1) % B

def _summary_stats(values: List[float], pctl: float=95.0) -> Optional[Dict[str, float]]:
    if values is None:
        return None
    v = torch.tensor(values, dtype=torch.float32)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return None
    mean = float(v.mean().item())
    mx = float(v.max().item())
    p = float(torch.quantile(v, torch.tensor(pctl / 100.0)).item())
    return {'mean': mean, f'p{int(pctl)}': p, 'max': mx}
