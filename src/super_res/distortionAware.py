from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_dct_basis(C: int, K: int) -> torch.Tensor:
    """
    Construct a row-wise DCT-II (type-II discrete cosine transform) basis of shape [K, C].
    Each of the K rows is an L2-normalized DCT vector over the C input channels.

    Notes
    -----
    * The basis is created in float32 on CPU and later cast to the input's dtype/device
      inside the forward path to be AMP-friendly.
    * Think of this as projecting token features onto K cosine modes ordered from low
      to high frequency (row index ~ frequency index).

    Args
    ----
    C : int
        Input channel/feature dimension per token.
    K : int
        Number of DCT frequency components to keep.

    Returns
    -------
    torch.Tensor
        The DCT-II basis matrix of shape [K, C] with row-wise unit norms.
    """
    n = torch.arange(C, dtype=torch.float32).unsqueeze(0)  # [1, C] sample positions
    k = torch.arange(K, dtype=torch.float32).unsqueeze(1)  # [K, 1] frequency indices
    basis = torch.cos(math.pi * (n + 0.5) * k / C)         # [K, C] unnormalized DCT-II rows
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-6)
    return basis


class DistortionAwareFrequencyExtractor(nn.Module):
    """
    Distortion-Aware Frequency Extractor
    ------------------------------------
    Produces lightweight routing features for MoE gating from two complementary views:

      (1) Frequency branch: fixed DCT basis + a learnable residual linear head.
          The fixed basis provides stable, interpretable frequency coordinates,
          while the residual allows task-specific adaptation beyond the fixed modes.

      (2) Distortion (spatial) branch: estimates two scalar cues per token:
          * Edge strength via Sobel gradients on an intensity map.
          * Noise strength as high-frequency residual (difference from a local average)
            with edge magnitude subtracted to avoid double-counting edges.

    The two branches are fused as [K (freq) + 2 (edge, noise)] → LayerNorm → Linear/MLP
    to produce an `out_dim`-dimensional feature per token for routing/gating.

    Usage
    -----
        feats = extractor(tokens[N, C], x_size=(H, W))  # -> [N, out_dim]
        # If x_size is None, only the frequency branch is used (keeps compatibility and cost low).

    Shapes
    ------
    * tokens : [N, C]    flattened spatial tokens, N = B * H * W (or arbitrary sequence length)
    * output : [N, out_dim]

    AMP & Device Notes
    ------------------
    Internal buffers (DCT basis, Sobel/avg kernels, scaling vectors) are cast on-the-fly
    to match the input's dtype/device to avoid precision/type mismatches under AMP.

    Parameters
    ----------
    input_size : int
        Per-token channel dimension C.
    out_dim : int
        Output feature dimension for gating.
    dct_K : int, default=16
        Number of DCT frequency components kept by the frequency branch.
    proj_hidden : int, default=0
        If > 0, enables a 2-layer MLP projection (Linear-GELU-Linear).
        If 0, uses a single Linear as the projection head.
    learnable_scale : bool, default=True
        If True, a learnable per-frequency scale `alpha[K]` modulates the DCT responses.
        Otherwise a fixed all-ones scale is used.
    """

    def __init__(
        self,
        input_size: int,
        out_dim: int,
        dct_K: int = 16,
        proj_hidden: int = 0,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.C = input_size
        self.out_dim = out_dim
        self.K = dct_K

        # --- Fixed DCT basis (project tokens onto K cosine modes) ---
        basis = _build_dct_basis(self.C, self.K)
        self.register_buffer("dct_basis", basis, persistent=True)

        # Optional learnable gain per frequency mode (stabilizes and adapts fixed basis)
        if learnable_scale:
            self.alpha = nn.Parameter(torch.ones(self.K))
        else:
            self.register_buffer("alpha", torch.ones(self.K), persistent=True)

        # --- Learnable frequency residual head (zero-init) ---
        # Captures task-specific frequency directions beyond the fixed DCT span.
        self.freq_res = nn.Linear(self.C, self.K, bias=False)
        nn.init.zeros_(self.freq_res.weight)

        # --- Spatial derivative & smoothing kernels for edge/noise cues ---
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32)
        sobel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("sobel_kernel", sobel, persistent=True)

        avg = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
        self.register_buffer("avg_kernel", avg, persistent=True)

        # Project token [C] → scalar intensity to reconstruct a 2D map for conv ops
        self.intensity_head = nn.Linear(self.C, 1, bias=False)
        nn.init.normal_(self.intensity_head.weight, std=1e-3)

        # Learnable global scales for edge/noise magnitudes
        self.beta_edge = nn.Parameter(torch.tensor(1.0))
        self.beta_noise = nn.Parameter(torch.tensor(1.0))

        # --- Fuse [K + 2] and project to out_dim ---
        self.norm = nn.LayerNorm(self.K + 2)
        if proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(self.K + 2, proj_hidden),
                nn.GELU(),
                nn.Linear(proj_hidden, out_dim)
            )
        else:
            self.proj = nn.Linear(self.K + 2, out_dim)

    # ---------- helpers ----------
    def _like(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Utility: cast tensor `t` to the device & dtype of `like` (AMP-friendly)."""
        return t.to(device=like.device, dtype=like.dtype)

    @torch.no_grad()
    def _infer_B(self, N: int, H: int, W: int) -> int:
        """
        Infer batch size B from flattened token length N and spatial size H×W.
        Requires N to be divisible by H*W.
        """
        assert (H * W) > 0 and (N % (H * W) == 0), \
            f"N={N} not divisible by H*W={H*W}"
        return N // (H * W)

    # ---------- branches ----------
    def _freq_branch(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Frequency branch: tokens [N, C] → [N, K]
        1) Fixed DCT projection with optional per-frequency scaling `alpha`.
        2) Learnable residual linear head (zero-initialized).
        3) Sum of the two yields robust, adaptable frequency cues.
        """
        basis = self._like(self.dct_basis, tokens)            # [K, C]
        y_fixed = F.linear(tokens, basis)                     # [N, K]
        alpha = self._like(self.alpha, tokens)                # [K]
        y_fixed = y_fixed * alpha.view(1, -1)                 # per-mode gain

        y_res = self.freq_res(tokens)                         # [N, K]
        if y_res.dtype != y_fixed.dtype:
            y_res = y_res.to(y_fixed.dtype)
        return y_fixed + y_res

    def _edge_noise_branch(self, tokens: torch.Tensor, x_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Distortion (spatial) branch with real 2D ops:
        tokens: [N, C], x_size=(H, W) → edge:[N, 1], noise:[N, 1]

        Steps
        -----
        1) Intensity map: project each token to 1 scalar and reshape to [B, 1, H, W].
        2) Edge magnitude: Sobel gradients → L2 magnitude.
        3) High-frequency residual: |intensity - avg3x3(intensity)|.
        4) Noise proxy: ReLU(residual - edge) to suppress edges from the residual.
        5) Flatten back to per-token scalars and apply learnable global scales β.

        AMP Notes
        ---------
        Convolution kernels and eps are cast to match intensity dtype/device.
        """
        H, W = x_size
        N, C = tokens.shape
        B = self._infer_B(N, H, W)

        # (1) tokens->intensity scalar, then reshape to 2D grid for convs
        intensity = self.intensity_head(tokens).view(B, H, W, 1).permute(0, 3, 1, 2).contiguous()  # [B,1,H,W]

        sobel_k = self._like(self.sobel_kernel, intensity)
        avg_k   = self._like(self.avg_kernel, intensity)
        eps     = torch.tensor(1e-6, device=intensity.device, dtype=intensity.dtype)

        # (2) Sobel gradients → edge magnitude
        grad = F.conv2d(intensity, sobel_k, padding=1)        # [B,2,H,W]
        gx, gy = grad[:, 0:1], grad[:, 1:2]
        edge = torch.sqrt(gx * gx + gy * gy + eps)            # [B,1,H,W]

        # (3)(4) Residual minus edge → noise proxy
        smooth = F.conv2d(intensity, avg_k, padding=1)        # [B,1,H,W]
        resid = (intensity - smooth).abs()                    # [B,1,H,W]
        noise = F.relu(resid - edge)                          # [B,1,H,W]

        # (5) Back to per-token scalars and global scaling
        edge = edge.view(B, H * W, 1).reshape(N, 1)           # [N,1]
        noise = noise.view(B, H * W, 1).reshape(N, 1)         # [N,1]

        edge = self._like(self.beta_edge, tokens) * edge
        noise = self._like(self.beta_noise, tokens) * noise
        return edge, noise

    def _edge_noise_fallback(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fallback path when `x_size` is None (no spatial convs available):
        * edge_approx: mean absolute first-order differences along the channel axis.
        * noise_approx: per-token channel-wise std.
        This keeps the API flexible at minimal extra cost (engineering convenience).
        """
        diff = tokens[:, 1:] - tokens[:, :-1]                 # [N, C-1]
        edge_approx = diff.abs().mean(dim=1, keepdim=True)    # [N,1]
        noise_approx = tokens.std(dim=1, keepdim=True)        # [N,1]

        edge_approx = self._like(self.beta_edge, tokens) * edge_approx
        noise_approx = self._like(self.beta_noise, tokens) * noise_approx
        return edge_approx, noise_approx

    # ---------- forward ----------
    def forward(self, tokens: torch.Tensor, x_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Compute distortion-aware frequency features per token.

        Args
        ----
        tokens : torch.Tensor
            [N, C] flattened tokens (N can be B*H*W).
        x_size : (int, int) or None
            Optional spatial size (H, W). If provided, enables true 2D edge/noise cues.
            If None, uses the fallback approximations.

        Returns
        -------
        torch.Tensor
            [N, out_dim] fused features suitable for routing/gating.
        """
        # (1) Frequency cues
        y_freq = self._freq_branch(tokens)  # [N, K]

        # (2) Distortion cues (edge, noise)
        if x_size is not None:
            edge, noise = self._edge_noise_branch(tokens, x_size)   # [N,1], [N,1]
        else:
            edge, noise = self._edge_noise_fallback(tokens)         # [N,1], [N,1]

        # (3) Fuse and project
        y = torch.cat([y_freq, edge, noise], dim=-1)  # [N, K+2]
        y = self.norm(y)                              # stabilize across tokens/modes

        # (Optional) If you need extra safety under unusual AMP configs:
        # if isinstance(self.proj, nn.Linear) and (y.dtype != self.proj.weight.dtype):
        #     y = y.to(self.proj.weight.dtype)

        out = self.proj(y)                            # [N, out_dim]
        return out
