
import torch
from typing import Dict, Tuple

__all__ = ["spectral_summary", "r2_linear", "cohens_d", "downsample_mask"]

def spectral_summary(y_freq: torch.Tensor, B: int, Ht: int, Wt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y_freq: [N,K] absolute spectral magnitudes per token.
    Return:
      mu: [B,Ht,Wt] spectral centroid over k
      k_peak: [B,Ht,Wt] argmax index
    """
    N, K = y_freq.shape
    y = y_freq.abs().view(B, Ht, Wt, K)
    w = y / (y.sum(-1, keepdim=True) + 1e-6)
    k_grid = torch.arange(K, device=y.device).view(1,1,1,K)
    mu = (w * k_grid).sum(-1)
    k_peak = y.argmax(-1)
    return mu, k_peak

def r2_linear(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute R^2 for linear regression y_pred ~ y_true in closed-form.
    Here we just use correlation squared when both are centered.
    """
    yp = y_pred.float().view(-1)
    yt = y_true.float().view(-1)
    yt_mean = yt.mean()
    ss_tot = ((yt - yt_mean)**2).sum()
    if ss_tot.abs() < 1e-12:
        return 0.0
    # fit scalar a,b by least squares
    A = torch.stack([yt, torch.ones_like(yt)], dim=1)  # [N,2]
    sol, _ = torch.lstsq(yp.unsqueeze(1), A)  # deprecated but works offline; returning first 2 rows
    a, b = sol[:2, 0]
    yp_hat = a*yt + b
    ss_res = ((yp - yp_hat)**2).sum()
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2.clamp(min=0.0, max=1.0).item())

def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    va = a.var(unbiased=True)
    vb = b.var(unbiased=True)
    sp = torch.sqrt(0.5*(va+vb) + 1e-6)
    d = (a.mean() - b.mean()) / sp
    return float(d.item())

def downsample_mask(M: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """
    M: [B,H,W] bool mask at pixel grid -> [B,Ht,Wt] bool mask at token grid via avg pool + threshold
    """
    B, H, W = M.shape
    mt = torch.nn.functional.adaptive_avg_pool2d(M.float().unsqueeze(1), (Ht, Wt)).squeeze(1)
    return (mt > 0.5)
