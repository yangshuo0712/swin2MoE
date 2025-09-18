from typing import Optional
import torch
import torch.nn as nn
import torch.cuda.amp as amp

from .distortionFeatureExtractor import DistortionAwareFeatureExtractor
from .ps_moe import WeightedSharedExpertMLP


class DisAwareExpertChoiceMoE(nn.Module):
    """
    Distortion-Aware Expert-Choice MoE with:
    - EMA-smoothed DAFE features + K-step intermittent learning (Z_new occasionally carries grad)
    - Sparse softmax over per-token candidate experts (after EC selection)
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_experts: int,
        num_bands: int,
        lora_rank: int,
        lora_alpha: float,
        capacity_factor: float = 1.25,
        dct_freq_features: int = 64,
        dct_extractor: str = "DistortionAware",  # "DistortionAware" or "linear"
        dct_K: int = 16,
        proj_hidden: int = 0,
        use_residual_hint: bool = True,
        add_inner_residual: bool = False,   # y = x + alpha*MoE(x)
        inner_residual_alpha: float = 1.0,
        router_every_k: int = 2,
        half_life_recomp: int = 2,
        max_fanout_per_token: int = 1,
        gating_temperature: float = 1.0,
        use_gating_layernorm: bool = True,
        # ---- 新增：间歇可学习开关（每 K 步用 Z_new 参与路由并反传）----
        extractor_train_every_k: int = 8,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.dct_freq_features = dct_freq_features
        self.use_residual_hint = use_residual_hint
        self.add_inner_residual = add_inner_residual
        self.inner_residual_alpha = inner_residual_alpha
        self.max_fanout_per_token = max_fanout_per_token
        self.gating_temperature = float(gating_temperature)
        self.use_gating_layernorm = use_gating_layernorm

        # ---- EMA & routing cadence ----
        self.router_every_k = max(1, int(router_every_k))
        self.half_life_recomp = max(1, int(half_life_recomp))
        self.extractor_train_every_k = max(1, int(extractor_train_every_k))
        self.register_buffer("_Z_ema", torch.empty(0), persistent=False)
        self._last_shape = None
        self._tick = 0

        # ---- Experts（Parameter-Shared + band weights）----
        self.experts = nn.ModuleList([
            WeightedSharedExpertMLP(
                input_size, hidden_size, output_size, num_bands, lora_rank, lora_alpha
            )
            for _ in range(num_experts)
        ])

        # ---- Distortion-aware feature extractor（DAFE）----
        if dct_extractor == "DCT":
            raise NotImplementedError("DCT extractor is not implemented.")
        elif dct_extractor == "DistortionAware":
            self.dct_extractor = DistortionAwareFeatureExtractor(
                input_size=input_size,
                out_dim=dct_freq_features,
                proj_hidden=proj_hidden,
                I_channels=4,
                use_log_compression=True,
                detach_tokens=True,   # 稳定：不把梯度传回 backbone tokens
                fast_mode=False,
                use_grad_energy=True
            )
        else:
            self.dct_extractor = nn.Linear(input_size, dct_freq_features, bias=False)

        # ---- Router / Gating ----
        gate_in_dim = input_size + dct_freq_features + (2 if use_residual_hint else 0)
        self.gate_ln = nn.LayerNorm(gate_in_dim) if self.use_gating_layernorm else nn.Identity()
        self.gating_network = nn.Linear(gate_in_dim, num_experts, bias=True)

        # buffers
        self.register_buffer("_expert_arange", torch.arange(num_experts), persistent=False)

    def _decay(self) -> float:
        return float(0.5 ** (1.0 / float(self.half_life_recomp)))

    def _extract_feats(self, tokens: torch.Tensor, x_size=None) -> torch.Tensor:
        if hasattr(self.dct_extractor, "forward"):
            try:
                return self.dct_extractor(tokens, x_size=x_size)
            except TypeError:
                return self.dct_extractor(tokens)
        return self.dct_extractor(tokens)

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_tensor(0.0)
        m = x.float().mean()
        v = x.float().var(correction=0)
        return v / (m * m + 1e-10)

    def forward(
        self,
        x: torch.Tensor,                 # [N, C]
        band_weights: torch.Tensor,      # [N, num_bands]
        loss_coef: float = 1e-2,
        x_size: Optional[tuple] = None,
        x_prev_tokens: Optional[torch.Tensor] = None,
        need_recompute: Optional[bool] = None,
    ):
        if self.add_inner_residual and self.output_size != self.input_size:
            raise ValueError("add_inner_residual=True requires output_size == input_size")

        device = x.device
        dtype = x.dtype
        N, C = x.shape

        # ==== EMA + K-step intermittent learning ====
        shape_changed = (self._last_shape != (N, self.dct_freq_features)) or (self._Z_ema.numel() == 0)
        if need_recompute is None:
            need_recompute = shape_changed or (self._tick % self.router_every_k == 0)

        compute_grad = self.training and need_recompute and (self._tick % self.extractor_train_every_k == 0)

        if need_recompute:
            with torch.set_grad_enabled(compute_grad):
                Z_new = self._extract_feats(x, x_size=x_size)   # [N, D]
            Z_det = Z_new.detach()
            d = self._decay()
            with torch.no_grad():
                if shape_changed:
                    self._Z_ema = Z_det.clone()
                else:
                    self._Z_ema.mul_(d).add_(Z_det, alpha=1.0 - d)
            self._last_shape = (N, self.dct_freq_features)
        Z_used = Z_new if (need_recompute and compute_grad) else self._Z_ema
        self._tick += 1

        # ==== build gating inputs ====
        parts = [x, Z_used]
        if self.use_residual_hint and (x_prev_tokens is not None):
            delta = (x - x_prev_tokens).detach()
            mu = delta.abs().mean(dim=1, keepdim=True)  # [N,1]
            sd = delta.abs().std(dim=1, keepdim=True)   # [N,1]
            mu, sd = torch.log1p(mu), torch.log1p(sd)
            parts.append(torch.cat([mu, sd], dim=1))    # [N,2]
        enhanced_tokens = torch.cat(parts, dim=1)       # [N, C + D (+2)]
        if self.use_gating_layernorm:
            enhanced_tokens = self.gate_ln(enhanced_tokens)

        with amp.autocast(enabled=False):
            logits_fp32 = self.gating_network(enhanced_tokens.float())  # [N, E]
        if self.gating_temperature != 1.0:
            logits_fp32 = logits_fp32 / self.gating_temperature

        num_tokens = N
        expert_capacity = max(1, int((num_tokens / float(self.num_experts)) * self.capacity_factor))
        k = min(expert_capacity, num_tokens)

        expert_scores_T = logits_fp32.transpose(0, 1)                          # [E, N]
        topk_scores, topk_idx = torch.topk(expert_scores_T, k=k, dim=1)        # [E, k]

        # dispatch_mask: True 表示 (token, expert) 被该专家选中
        dispatch_mask = torch.zeros((num_tokens, self.num_experts), device=device, dtype=torch.bool)
        rows = topk_idx
        cols = self._expert_arange.unsqueeze(1).expand_as(rows)
        dispatch_mask[rows, cols] = True

        uncovered = ~dispatch_mask.any(dim=1)  # [N]
        if uncovered.any():
            best_expert = logits_fp32[uncovered].argmax(dim=1)
            dispatch_mask[uncovered, best_expert] = True

        if (self.max_fanout_per_token is not None) and (0 < self.max_fanout_per_token < self.num_experts):
            m = self.max_fanout_per_token
            masked_scores = logits_fp32.masked_fill(~dispatch_mask, float("-inf"))  # [N, E]
            token_topm_idx = torch.topk(masked_scores, k=m, dim=1).indices          # [N, m]
        else:
            # 不限 fan-out，则候选集就是 dispatch_mask 里的所有 True 位置
            # 为了用同一条稀疏 softmax路径，我们仍取按分数排序的所有候选（最多 E 个）
            masked_scores = logits_fp32.masked_fill(~dispatch_mask, float("-inf"))
            # 计算每 token 的候选数，取其非 -inf 的个数
            cand_counts = dispatch_mask.sum(dim=1)  # [N]
            # 为简洁，仍取前 m= cand_counts.clamp_min(1).max()，但按每行 gather 时无效位会是 -inf→softmax≈0
            m = int(cand_counts.max().item())
            token_topm_idx = torch.topk(masked_scores, k=m, dim=1).indices

        # 仅在候选上 softmax，再 scatter 回去（显著减算）
        sel_logits  = torch.gather(logits_fp32, 1, token_topm_idx)       # [N, m]
        sel_weights = torch.softmax(sel_logits, dim=1).to(dtype)         # [N, m]
        gating_weights = torch.zeros((num_tokens, self.num_experts), device=device, dtype=dtype)
        gating_weights.scatter_(1, token_topm_idx, sel_weights)          # 稀疏写回

        # ==== Experts forward & blend ====
        final_output = torch.zeros((num_tokens, self.output_size), device=device, dtype=dtype)
        for e in range(self.num_experts):
            idx = torch.nonzero(gating_weights[:, e] > 0, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            expert_tokens = x[idx]                                   # [n_e, C]
            expert_bw     = band_weights[idx]                        # [n_e, num_bands]
            expert_gate   = gating_weights[idx, e].unsqueeze(1)      # [n_e, 1], already dtype
            expert_out    = self.experts[e](expert_tokens, expert_bw)# [n_e, out]
            final_output.index_add_(0, idx, expert_out * expert_gate)

        # ==== Aux losses & debug ====
        importance = gating_weights.sum(0)  # [E]
        load       = (gating_weights > 0).sum(0)  # [E]
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        self.debug_outputs = {
            "importance": importance.detach(),
            "load": load.detach(),
            "avg_fanout": (gating_weights > 0).sum(dim=1).float().mean().detach()
        }

        if self.add_inner_residual:
            final_output = x + self.inner_residual_alpha * final_output

        return final_output, loss
