#
# Source code: https://github.com/mv-lab/swin2sr
#
# -----------------------------------------------------------------------------------
# Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration, https://arxiv.org/abs/2209.11345
# Written by Conde and Choi et al.
# -----------------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from utils import is_main_process

from .ps_moe import MoE
from .moe_cadr import ComplexityAwareMoE
from .caec_moe import CAEC_MoE
from .fec_dps_moe import FreqAwareExpertChoiceMoE
from .utils import Mlp, window_partition, window_reverse

class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        use_lepe=False,
        use_cpb_bias=True,
        use_rpe_bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        self.use_cpb_bias = use_cpb_bias

        if self.use_cpb_bias:
            self.cpb_mlp = nn.Sequential(
                nn.Linear(2, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_heads, bias=False),
            )

            relative_coords_h = torch.arange(
                -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
            )
            relative_coords_w = torch.arange(
                -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
            )
            relative_coords_table = (
                torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij'))
                .permute(1, 2, 0)
                .contiguous()
                .unsqueeze(0)
            )
            if pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
            else:
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
            relative_coords_table *= 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )
            self.register_buffer("relative_coords_table", relative_coords_table)

            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :])
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.use_rpe_bias = use_rpe_bias
        if self.use_rpe_bias:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :])
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            rpe_relative_position_index = relative_coords.sum(-1)
            self.register_buffer("rpe_relative_position_index", rpe_relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.use_lepe = use_lepe
        if self.use_lepe:
            self.get_v = nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
            )

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_lepe:
            lepe = self.lepe_pos(v)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1.0 / 0.01)).to(self.logit_scale.device),
        ).exp()
        attn = attn * logit_scale

        if self.use_cpb_bias:
            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attn = attn + relative_position_bias.unsqueeze(0)

        if self.use_rpe_bias:
            relative_position_bias = self.relative_position_bias_table[self.rpe_relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = attn @ v

        if self.use_lepe:
            x = x + lepe

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def lepe_pos(self, v):
        B, NH, HW, NW = v.shape
        C = NH * NW
        H = W = int(math.sqrt(HW))
        v = v.transpose(-2, -1).contiguous().view(B, C, H, W)
        lepe = self.get_v(v)
        lepe = lepe.reshape(-1, self.num_heads, NW, HW)
        lepe = lepe.permute(0, 1, 3, 2).contiguous()
        return lepe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
        use_lepe (bool): Whether to use Locality Enhanced Positional Encoding.
        use_cpb_bias (bool): Whether to use Continuous Position Bias.
        MoE_config (dict, optional): Configuration for the MoE layer.
        use_rpe_bias (bool): Whether to use Relative Positional Encoding bias.
        model_version (str, optional): The overall model version, used to determine MoE type.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
        use_lepe=False,
        use_cpb_bias=True,
        MoE_config=None,
        use_rpe_bias=False,
        model_version=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            use_lepe=use_lepe,
            use_cpb_bias=use_cpb_bias,
            use_rpe_bias=use_rpe_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.is_moe = MoE_config is not None
        if MoE_config:
            version = model_version
            if version == "FEC-DPS-MOE":
                self.mlp = FreqAwareExpertChoiceMoE(
                    input_size=dim,
                    output_size=dim,
                    hidden_size=mlp_hidden_dim,
                    num_experts=MoE_config.get("num_experts", 8),
                    num_bands=MoE_config.get("num_bands"),
                    lora_rank=MoE_config.get("lora_rank"),
                    lora_alpha=MoE_config.get("lora_alpha"),
                    capacity_factor=MoE_config.get("capacity_factor", 1.25),
                    dct_freq_features=MoE_config.get("dct_freq_features", 64),
                    dct_extractor=MoE_config.get("dct_extractor", "linear")
                )
            elif version == "CAEC-MoE":
                 self.mlp = CAEC_MoE(
                    input_size=dim, output_size=dim, hidden_size=mlp_hidden_dim,
                    num_experts=MoE_config.get("num_experts", 8),
                    k=MoE_config.get("k", 2),
                    num_bands=MoE_config.get("num_bands"),
                    lora_rank=MoE_config.get("lora_rank"),
                    lora_alpha=MoE_config.get("lora_alpha"),
                    capacity_factor=MoE_config.get("capacity_factor", 1.25),
                    complexity_loss_weight=MoE_config.get("complexity_loss_weight", 0.1)
                )
            elif version == "CADR":
                 self.mlp = ComplexityAwareMoE(
                    input_size=dim, output_size=dim, hidden_size=mlp_hidden_dim,
                    num_experts=MoE_config.get("num_experts", 8),
                    k=MoE_config.get("k", 2),
                    num_bands=MoE_config.get("num_bands"),
                    lora_rank=MoE_config.get("lora_rank"),
                    lora_alpha=MoE_config.get("lora_alpha"),
                 )
            else: # Default to standard PS-MoE
                self.mlp = MoE(
                    input_size=dim, output_size=dim, hidden_size=mlp_hidden_dim,
                    num_experts=MoE_config.get("num_experts", 8),
                    k=MoE_config.get("k", 2),
                    num_bands=MoE_config.get("num_bands"),
                    lora_rank=MoE_config.get("lora_rank"),
                    lora_alpha=MoE_config.get("lora_alpha"),
                )
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size, band_indices):
        """
        Forward pass of the Swin Transformer Block.

        Args:
            x (torch.Tensor): Input feature map, shape [B, L, C].
            x_size (Tuple[int, int]): Input resolution (H, W).
            band_indices (torch.Tensor): Band indices for each token, shape [B*L].

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output feature map, shape [B, L, C].
                - The MoE load-balancing loss, or None if not using MoE.
        """
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        x_norm = x_norm.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_norm

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(B, H * W, C)
        x = shortcut + self.drop_path(attn_x)

        shortcut2 = x
        x_ffn = self.norm2(x)
        loss_moe = None

        if self.is_moe:
            res, loss_moe = self.mlp(x_ffn.view(-1, C), band_indices=band_indices, x_size=(H, W))
            res = res.view(B, L, C)
        else:
            res = self.mlp(x_ffn)

        x = shortcut2 + self.drop_path(res)
        return x, loss_moe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    r"""A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0,
        qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
        norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
        pretrained_window_size=0, use_lepe=False, use_cpb_bias=True,
        MoE_config=None, use_rpe_bias=False, model_version=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, pretrained_window_size=pretrained_window_size,
                    use_lepe=use_lepe, use_cpb_bias=use_cpb_bias, MoE_config=MoE_config, use_rpe_bias=use_rpe_bias,
                    model_version=model_version,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, band_indices):
        """
        Forward pass of the BasicLayer.

        Args:
            x (torch.Tensor): Input feature map, shape [B, L, C].
            x_size (Tuple[int, int]): Input resolution (H, W).
            band_indices (torch.Tensor): Band indices for each token, shape [B*L].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output feature map, shape [B, L, C].
                - The aggregated MoE loss from all blocks in the layer.
        """
        loss_moe_all = 0.0
        for blk in self.blocks:
            x, loss_moe = blk(x, x_size, band_indices)
            if loss_moe is not None:
                loss_moe_all += loss_moe

        if self.downsample is not None:
            x = self.downsample(x)
        return x, loss_moe_all

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class RSTB(nn.Module):
    r"""Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        model_version (str, optional): The overall model version, used to determine MoE type.
    """

    def __init__(
        self,
        dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0,
        qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
        norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
        img_size=224, patch_size=4, resi_connection="1conv",
        use_lepe=False, use_cpb_bias=True, MoE_config=None, use_rpe_bias=False,
        model_version=None,
    ):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample,
            use_checkpoint=use_checkpoint, use_lepe=use_lepe, use_cpb_bias=use_cpb_bias,
            MoE_config=MoE_config, use_rpe_bias=use_rpe_bias, model_version=model_version,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None
        )
        self.patch_unembed = PatchUnembed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None
        )

    def forward(self, x, x_size, band_indices):
        """
        Forward pass of the Residual Swin Transformer Block (RSTB).

        Args:
            x (torch.Tensor): Input feature map, shape [B, L, C].
            x_size (Tuple[int, int]): Input resolution (H, W).
            band_indices (torch.Tensor): Band indices for each token, shape [B*L].

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output feature map, shape [B, L, C].
                - The aggregated MoE loss, or None.
        """
        shortcut = x
        res, loss_moe = self.residual_group(x, x_size, band_indices)
        res = self.conv(self.patch_unembed(res, x_size))
        res = self.patch_embed(res)
        res = res + shortcut
        return res, loss_moe

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchUnembed(nn.Module):
    r"""Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
    Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class Swin2SR(nn.Module):
    r"""Swin2SR with support for Parameter-Shared MoE and custom routing.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        MoE_config (dict): Configuration for the MoE layer, enabling its use.
        model_version (str): The overall model version, used to determine MoE type.
    """
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_lepe=False,
        use_cpb_bias=True,
        MoE_config=None,
        use_rpe_bias=False,
        model_version=None,
    ):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.is_moe = MoE_config is not None
        
        if self.is_moe:
            version = model_version if model_version else "PS-MoE"
            if is_main_process():
                print(f"-->>> Using MoE version: {version}")

        # NOTE: MoE_config is now passed directly as a dict.
        if MoE_config:
            if self.is_moe and ("num_bands" not in MoE_config or MoE_config["num_bands"] is None):
                MoE_config["num_bands"] = in_chans
                if is_main_process():
                    print(f"PS-MoE is enabled with config: {MoE_config}")
                    print(f"Set PS-MoE num_bands to {in_chans} from input channels.")

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnembed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_lepe=use_lepe,
                use_cpb_bias=use_cpb_bias,
                MoE_config=MoE_config,
                use_rpe_bias=use_rpe_bias,
                model_version=model_version,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x, band_indices):
        """
        Forward pass through the deep feature extraction body of the model.

        Args:
            x (torch.Tensor): Input tokens, shape [B, L, C].
            band_indices (torch.Tensor): Band indices for each token, shape [B*L].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output feature map, shape [B, embed_dim, H, W].
                - The total aggregated MoE loss from all layers.
        """
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x_size = (H, W)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        loss_moe_all = 0.0
        for layer in self.layers:
            x, loss_moe = layer(x, x_size, band_indices)
            if loss_moe is not None:
                loss_moe_all += loss_moe

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x, loss_moe_all

    def forward(self, x):
        """
        Main forward pass of the Swin2SR model.

        Args:
            x (torch.Tensor): Input low-resolution image, shape [B, C, H, W].

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - final_output: The upscaled high-resolution image, shape [B, C, H*s, W*s].
                - (final_output, loss_moe): If MoE is enabled and training, also returns the loss.
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x_pre = (x - self.mean) * self.img_range
        x_feat = self.conv_first(x_pre)
        x_tokens = self.patch_embed(x_feat)
        B, L, C = x_tokens.shape

        loss_moe = 0.0
        band_indices = None
        if self.is_moe:
            num_bands = self.conv_first.in_channels
            num_patches_per_band_item = L // num_bands
            single_band_indices = torch.arange(num_bands, device=x.device, dtype=torch.long)
            single_band_indices = single_band_indices.repeat_interleave(num_patches_per_band_item)
            band_indices = single_band_indices.repeat(B)

        res, loss_moe = self.forward_features(x_tokens, band_indices)

        if self.upsampler == "pixelshuffledirect":
            x_out = self.conv_after_body(res) + x_feat
            x_out = self.upsample(x_out)
        elif self.upsampler == "pixelshuffle":
            x_body = self.conv_after_body(res)
            x_feat = x_feat + x_body
            x_out = self.conv_before_upsample(x_feat)
            x_out = self.conv_last(self.upsample(x_out))
        else:
            x_body = self.conv_after_body(res)
            x_feat = x_feat + x_body
            x_out = self.conv_last(x_feat)
        
        x_out = x_out / self.img_range + self.mean
        final_output = x_out[:, :, : H * self.upscale, : W * self.upscale]

        if self.is_moe and self.training:
            return final_output, loss_moe
        else:
            return final_output

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == "pixelshuffledirect":
            flops += self.upsample.flops()
        return flops
