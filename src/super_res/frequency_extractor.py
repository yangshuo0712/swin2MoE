from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_dct_basis(C: int, K: int) -> torch.Tensor:
    """Return [K, C] DCT-II basis (row-wise) with L2 normalization.
    C: input feature dim; K: number of frequency bands to keep.
    """
    n = torch.arange(C, dtype=torch.float32).unsqueeze(0) # [1, C]
    k = torch.arange(K, dtype=torch.float32).unsqueeze(1) # [K, 1]
    basis = torch.cos(math.pi * (n + 0.5) * k / C) # [K, C]
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-6)
    return basis

class HybridFrequencyExtractor(nn.Module):
    """
    HFEv2: Fixed frequency basis (DCT) + tiny residual branch + normalization.


    * Fixed branch: token x [B, N, C] -> y_fixed = x @ DCT^T -> [B, N, K]
    * Residual branch: depthwise+pointwise conv on channel dim (zero-init) -> [B, N, K]
    * Normalize & project to desired output size.


    Optional auxiliary losses (computed when return_aux=True):
    - loss_div: de-correlation across K bands (covariance ~ I)
    - loss_order: optional ordering w.r.t. edge_score (hi-band > lo-band on edges)


    Args:
    input_size: token feature dim C
    output_size: projected frequency feature dim for the router
    K: number of kept frequency bands from DCT
    freeze_fixed_epochs: epochs to disable residual branch (call set_epoch each epoch)
    learnable_scale: whether to learn per-band scaling of fixed DCT outputs
    use_residual: enable tiny residual branch
    proj_hidden: if >0, add a GELU bottleneck MLP before output
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        K: int = 8,
        freeze_fixed_epochs: int = 5,
        learnable_scale: bool = True,
        use_residual: bool = True,
        proj_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.K = K
        self.freeze_fixed_epochs = freeze_fixed_epochs

        # ---- Fixed DCT basis branch ----
        basis = _build_dct_basis(input_size, K) # [K, C]
        self.register_buffer("dct_basis", basis, persistent=True)
        if learnable_scale:
            self.alpha = nn.Parameter(torch.ones(K))
        else:
            self.register_buffer("alpha", torch.ones(K), persistent=True)


        # ---- Tiny residual branch: depthwise + pointwise 1D conv on channel dim ----
        self.use_residual = use_residual
        if use_residual:
            self.res_dw = nn.Conv1d(
            in_channels=input_size, out_channels=input_size,
            kernel_size=3, padding=1, groups=input_size, bias=False
            )
            self.res_pw = nn.Conv1d(
            in_channels=input_size, out_channels=K,
            kernel_size=1, bias=False
            )
            nn.init.zeros_(self.res_dw.weight) # zero-init to be a no-op at start
            nn.init.zeros_(self.res_pw.weight)
        else:
            self.res_dw = None
            self.res_pw = None


        self.norm = nn.LayerNorm(K)
        if proj_hidden > 0:
            self.proj = nn.Sequential(
            nn.Linear(K, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, output_size),
            )
        else:
            self.proj = nn.Linear(K, output_size)

        self._epoch = 0 # updated externally via set_epoch

    # -------------------------------
    # Public utils
    # -------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)


    @torch.no_grad()
    def _fixed_branch(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, N, C] ; basis [K, C] -> y [B, N, K]
        y = torch.matmul(x, self.dct_basis.t())
        y = y * self.alpha # learnable per-band scaling
        return y


    def _residual_branch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_residual or self.res_dw is None:
            return torch.zeros(x.size(0), x.size(1), self.K, device=x.device, dtype=x.dtype)
        B, N, C = x.shape
        t = x.reshape(B * N, C, 1)
        t = self.res_dw(t) # [B*N, C, 1]
        t = self.res_pw(t) # [B*N, K, 1]
        t = t.squeeze(-1).reshape(B, N, self.K)
        return t

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        edge_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict] | torch.Tensor:
        """
        x: [B, N, C] token features
        edge_score: optional [B, N] edge strength map downsampled to token grid
        """
        use_res = (self._epoch >= self.freeze_fixed_epochs)
        y_fixed = self._fixed_branch(x) # [B, N, K]
        y_res = self._residual_branch(x) if use_res else 0.0
        y = y_fixed + y_res


        # Normalize band energy to avoid collapse
        y = F.normalize(y, dim=-1)
        y = self.norm(y)


        out = self.proj(y) # [B, N, output_size]
        if not return_aux:
            return out


        # ---------- Auxiliary regularizers ----------
        aux: dict[str, torch.Tensor] = {}
        B, N, K = y.shape
        z = y.reshape(B * N, K)
        cov = (z.t() @ z) / (B * N + 1e-6)
        I = torch.eye(K, device=z.device, dtype=z.dtype)
        aux["loss_div"] = ((cov - I) ** 2).mean()


        if edge_score is not None:
        # hi: avg of higher half bands; lo: avg of lower half
            hi = y[..., K // 2 :].mean(dim=-1) # [B, N]
            lo = y[..., : K // 2].mean(dim=-1) # [B, N]
            m = edge_score.median()
            hard = (edge_score > m).float()
            # Encourage hi>lo for edges; lo>=hi for smooth tokens
            loss_edges = F.relu(0.10 - (hi - lo)) * hard
            loss_smooth = F.relu(0.00 - (lo - hi)) * (1.0 - hard)
            aux["loss_order"] = (loss_edges + loss_smooth).mean()
        else:
            aux["loss_order"] = y.new_zeros(())


        return out, aux
