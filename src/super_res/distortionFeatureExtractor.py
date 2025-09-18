from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistortionAwareFeatureExtractor(nn.Module):
    """
    Inputs:
        tokens: [N, C], N=B*H*W
        x_size: (H, W)

    Output:
        [N, out_dim]
    """

    def __init__(
        self,
        input_size: int,
        out_dim: int,
        proj_hidden: int = 64,
        I_channels: int = 4,
        laplacian_dilations: Tuple[int, ...] = (1, 2, 3),
        small_win: int = 3,
        large_win: int = 7,
        use_log_compression: bool = True,
        detach_tokens: bool = False,
        use_grad_energy: bool = True,  # True: gx^2+gy^2；False: sqrt(gx^2+gy^2)
        fast_mode: bool = False,
    ):
        super().__init__()

        assert small_win % 2 == 1 and large_win % 2 == 1, (
            "small_win/large_win must be odd."
        )

        self.C = input_size
        self.out_dim = out_dim
        self.Ic = I_channels
        self.use_log = use_log_compression
        self.detach_tokens = detach_tokens
        self.use_grad_energy = use_grad_energy
        self.lap_dils = (1, 3) if fast_mode else laplacian_dilations
        self.small_win = small_win
        self.large_win = large_win

        # 将 token 压成多通道“强度图像”
        self.intensity_head = nn.Linear(self.C, self.Ic, bias=False)
        nn.init.kaiming_uniform_(self.intensity_head.weight, a=math.sqrt(5))

        # Sobel
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer(
            "sobel_x_dw", sobel_x.repeat(self.Ic, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "sobel_y_dw", sobel_y.repeat(self.Ic, 1, 1, 1), persistent=False
        )

        # Laplace (4-neighborhood)
        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_dw", lap.repeat(self.Ic, 1, 1, 1), persistent=False)

        # 统计维度（去掉伪 MAD 后）：grad(2) + LoG(2*len(dils)) + resid_s(2) + resid_l(2) + var_s(2) + var_l(2)
        self.stat_dim = 2 * (len(self.lap_dils) + 5)
        self.norm = nn.LayerNorm(self.stat_dim)
        if proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(self.stat_dim, proj_hidden),
                nn.GELU(),
                nn.Linear(proj_hidden, out_dim),
            )
        else:
            self.proj = nn.Linear(self.stat_dim, out_dim)

    # ----------------- helpers -----------------
    @staticmethod
    def _infer_B(N: int, H: int, W: int) -> int:
        assert H > 0 and W > 0 and N % (H * W) == 0, (
            f"N={N} not divisible by H*W={H * W}"
        )
        return N // (H * W)

    @staticmethod
    def _log1p_stable(z: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp_min(z, 0))

    def _depthwise3x3(
        self, x: torch.Tensor, w_dw: torch.Tensor, dilation: int = 1
    ) -> torch.Tensor:
        """x:[B,M,H,W], w_dw:[M,1,3,3] depthwise conv."""
        pad = dilation  # 3x3 with dilation d -> effective size 1+2d
        # 不在这里 .to(...)，buffer 会随 model.to(...) 同步
        return F.conv2d(x, w_dw, padding=pad, dilation=dilation, groups=self.Ic)

    def _avg_mean(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        return F.avg_pool2d(
            x, kernel_size=k, stride=1, padding=pad, count_include_pad=False
        )

    # ----------------- core stats -----------------
    def _compute_stats(self, I: torch.Tensor) -> torch.Tensor:
        """
        I: [B, M, H, W]
        Return: [B, H, W, stat_dim]
        """
        eps = 1e-6
        assert I.dim() == 4 and I.shape[1] == self.Ic, (
            "I must be [B,M,H,W] with M==I_channels"
        )

        # 统计路径统一用 fp32 计算（更稳），最后再 cast 回来
        I32 = I.float()

        # 梯度
        gx = self._depthwise3x3(I32, self.sobel_x_dw, dilation=1)
        gy = self._depthwise3x3(I32, self.sobel_y_dw, dilation=1)
        grad_energy = gx * gx + gy * gy
        grad_feat = (
            grad_energy if self.use_grad_energy else torch.sqrt(grad_energy + eps)
        )
        del gx, gy, grad_energy

        def agg_mean_std(T: torch.Tensor):
            var, mu = torch.var_mean(T, dim=1, unbiased=False)  # over Ic -> [B,H,W]
            sd = torch.sqrt(var + eps)
            return mu, sd

        feats = []

        # grad
        mu, sd = agg_mean_std(grad_feat)
        feats += [mu, sd]
        del grad_feat

        # LoG at dilations
        for d in self.lap_dils:
            R = self._depthwise3x3(I32, self.lap_dw, dilation=d).abs()
            mu, sd = agg_mean_std(R)
            feats += [mu, sd]
            del R

        # local means & vars
        mean_s = self._avg_mean(I32, self.small_win)
        mean_l = self._avg_mean(I32, self.large_win)
        mean2_s = self._avg_mean(I32 * I32, self.small_win)
        mean2_l = self._avg_mean(I32 * I32, self.large_win)

        resid_s = (I32 - mean_s).abs()
        resid_l = (I32 - mean_l).abs()
        var_s = torch.clamp_min(mean2_s - mean_s * mean_s, 0.0)
        var_l = torch.clamp_min(mean2_l - mean_l * mean_l, 0.0)

        mu, sd = agg_mean_std(resid_s)
        feats += [mu, sd]
        mu, sd = agg_mean_std(resid_l)
        feats += [mu, sd]
        mu, sd = agg_mean_std(var_s)
        feats += [mu, sd]
        mu, sd = agg_mean_std(var_l)
        feats += [mu, sd]

        Z = torch.stack(feats, dim=-1)  # [B,H,W,stat_dim]
        assert Z.shape[-1] == self.stat_dim, (
            f"stat_dim mismatch: {Z.shape[-1]} vs {self.stat_dim}"
        )
        if self.use_log:
            Z = self._log1p_stable(Z)
        return Z

    # ----------------- forward -----------------
    def forward(
        self, tokens: torch.Tensor, x_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        assert x_size is not None, "require x_size=(H,W)"
        H, W = x_size
        N, C = tokens.shape
        B = self._infer_B(N, H, W)

        x = tokens.detach() if self.detach_tokens else tokens
        I = (
            self.intensity_head(x)
            .view(B, H, W, self.Ic)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        I = I.to(memory_format=torch.channels_last)

        Z = self._compute_stats(I)  # [B,H,W,stat_dim] (fp32)
        Z = Z.view(B * H * W, self.stat_dim)  # [N,stat_dim]
        Z = self.norm(Z)
        out = self.proj(Z).to(tokens.dtype)
        return out
