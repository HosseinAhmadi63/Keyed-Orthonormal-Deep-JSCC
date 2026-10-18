import math
from typing import Optional, Dict
import torch
import torch.nn as nn
from utils import is_power_of_two, wrap_to_int64
MASK32 = 4294967295
MASK16 = 65535

def _mask32(x: torch.Tensor) -> torch.Tensor:
    return x & torch.tensor(MASK32, device=x.device, dtype=torch.int64)

def add32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _mask32(a + b)

def xor32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _mask32(a ^ b)

def mul32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m16 = torch.tensor(MASK16, device=a.device, dtype=torch.int64)
    a0 = a & m16
    a1 = a >> 16 & m16
    b0 = b & m16
    b1 = b >> 16 & m16
    p0 = a0 * b0
    p1 = a0 * b1 + a1 * b0
    res = p0 + ((p1 & m16) << 16)
    return _mask32(res)

def _to_u32_halves_from_i64(x_i64: torch.Tensor):
    mask = torch.tensor(MASK32, device=x_i64.device, dtype=torch.int64)
    lo = x_i64 & mask
    hi = x_i64 >> 32 & mask
    return (lo, hi)

def seed32_from_key_nonce(key_i64: torch.Tensor, nonce_i64: torch.Tensor) -> torch.Tensor:
    klo, khi = _to_u32_halves_from_i64(key_i64)
    nlo, nhi = _to_u32_halves_from_i64(nonce_i64)
    A = torch.tensor(2654435769, device=nonce_i64.device, dtype=torch.int64)
    s = add32(xor32(nlo, klo), mul32(A, xor32(nhi, khi)))
    return s

def splitmix32(seed32: torch.Tensor, n: int) -> torch.Tensor:
    device = seed32.device
    B = seed32.shape[0]
    gamma = torch.tensor(2654435769, device=device, dtype=torch.int64)
    mul1 = torch.tensor(2246822507, device=device, dtype=torch.int64)
    mul2 = torch.tensor(3266489909, device=device, dtype=torch.int64)
    i = torch.arange(1, n + 1, device=device, dtype=torch.int64).view(1, -1)
    x = add32(seed32.view(B, 1), mul32(gamma, i))
    z = x
    z = xor32(z, z >> 16)
    z = mul32(z, mul1)
    z = xor32(z, z >> 13)
    z = mul32(z, mul2)
    z = xor32(z, z >> 16)
    return z

def fwht(x: torch.Tensor) -> torch.Tensor:
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, -1, 2 * h)
        a = x[..., :h]
        b = x[..., h:2 * h]
        x = torch.cat([a + b, a - b], dim=-1)
        x = x.view(B, N)
        h *= 2
    return x

class KeyedScrambler(nn.Module):

    def __init__(self, latent_n: int, key_pyint: int, use_perm: bool=True):
        super().__init__()
        assert is_power_of_two(latent_n), 'LATENT_N must be a power of 2.'
        self.N = latent_n
        self.use_perm = use_perm
        key_i64 = wrap_to_int64(key_pyint)
        self.register_buffer('key_i64', torch.tensor(key_i64, dtype=torch.int64), persistent=False)
        self.register_buffer('K_SIGN1', torch.tensor(324508639, dtype=torch.int64), persistent=False)
        self.register_buffer('K_SIGN2', torch.tensor(610839776, dtype=torch.int64), persistent=False)
        self.register_buffer('K_PERM', torch.tensor(826366246, dtype=torch.int64), persistent=False)

    def _params(self, nonce_i64: torch.Tensor, key_i64: torch.Tensor):
        device = nonce_i64.device
        B = nonce_i64.shape[0]
        N = self.N
        key_i64 = key_i64.to(device)
        if key_i64.numel() == 1:
            key_i64 = key_i64.view(1).expand(B)
        seed0 = seed32_from_key_nonce(key_i64, nonce_i64)
        r1 = splitmix32(xor32(seed0, self.K_SIGN1.to(device)), N)
        r2 = splitmix32(xor32(seed0, self.K_SIGN2.to(device)), N)
        sign1 = 1.0 - 2.0 * (r1 & 1).to(torch.float32)
        sign2 = 1.0 - 2.0 * (r2 & 1).to(torch.float32)
        if self.use_perm:
            rp = splitmix32(xor32(seed0, self.K_PERM.to(device)), N)
            perm = torch.argsort(rp, dim=-1).to(torch.long)
            inv_perm = torch.empty_like(perm)
            ar = torch.arange(N, device=device, dtype=torch.long).view(1, -1).expand(B, N)
            inv_perm.scatter_(1, perm, ar)
        else:
            perm = torch.arange(N, device=device, dtype=torch.long).view(1, -1).expand(B, N)
            inv_perm = perm
        return (perm, inv_perm, sign1, sign2)

    def scramble(self, x: torch.Tensor, nonce_i64: torch.Tensor) -> torch.Tensor:
        perm, _, sign1, sign2 = self._params(nonce_i64, self.key_i64)
        N = self.N
        scale = 1.0 / math.sqrt(N)
        y = x * sign1
        y = fwht(y) * scale
        y = y * sign2
        y = y.gather(1, perm)
        return y

    def descramble(self, y: torch.Tensor, nonce_i64: torch.Tensor, key_override_pyint: Optional[int]=None) -> torch.Tensor:
        if key_override_pyint is None:
            key_i64 = self.key_i64
        else:
            key_i64 = torch.tensor(wrap_to_int64(key_override_pyint), device=y.device, dtype=torch.int64)
        _, inv_perm, sign1, sign2 = self._params(nonce_i64, key_i64)
        N = self.N
        scale = 1.0 / math.sqrt(N)
        x = y.gather(1, inv_perm)
        x = x * sign2
        x = fwht(x) * scale
        x = x * sign1
        return x

def power_normalize_per_sample(x: torch.Tensor, target_power: float=1.0, eps: float=1e-06) -> torch.Tensor:
    x32 = x.float()
    p = x32.pow(2).mean(dim=1, keepdim=True).clamp_min(eps)
    s = (x32.new_tensor(target_power) / p).sqrt()
    return (x32 * s).to(dtype=x.dtype)

def per_sample_power(x: torch.Tensor, eps: float=1e-08) -> torch.Tensor:
    x32 = x.float()
    return x32.pow(2).mean(dim=1).clamp_min(eps)
