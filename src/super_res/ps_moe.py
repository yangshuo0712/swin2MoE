import torch
import torch.nn as nn
import math
from torch.distributions.normal import Normal
from copy import deepcopy
import numpy as np

from .utils import Mlp as MLP

class LowRankAdapter(nn.Module):
    """
    实现低秩适配 (LoRA) 的模块。
    将一个大的权重更新矩阵 AW 分解为两个小的矩阵 B@A。
    """
    def __init__(self, in_features, out_features, rank, alpha=8.0):
        """
        初始化 LoRA 模块。
        参数:
        in_features (int): 输入特征维度。
        out_features (int): 输出特征维度。
        rank (int): 低秩分解的秩。
        alpha (float): 适配器的缩放因子。
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # LoRA 分解的两个线性层
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        # 缩放因子,用于控制适配器对原始权重的影响
        self.scaling = self.alpha / self.rank
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化权重。
        - lora_A 使用 Kaiming Uniform 初始化,这是一种标准做法。
        - lora_B 初始化为零,确保在训练开始时,AW为零矩阵。
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """
        前向传播,计算 W@x。
        """
        return self.lora_B(self.lora_A(x)) * self.scaling

class SharedExpertMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_bands, rank, alpha):
        super().__init__()
        self.in_features_actual = in_features - 1
        self.num_bands = num_bands

        # 共享权重
        self.fc1 = nn.Linear(self.in_features_actual, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

        # 将LoRA适配器参数堆叠，以支持向量化操作
        # lora_fc1
        self.lora_fc1_A = nn.Parameter(torch.zeros(num_bands, self.in_features_actual, rank))
        self.lora_fc1_B = nn.Parameter(torch.zeros(num_bands, rank, hidden_features))
        # lora_fc2
        self.lora_fc2_A = nn.Parameter(torch.zeros(num_bands, hidden_features, rank))
        self.lora_fc2_B = nn.Parameter(torch.zeros(num_bands, rank, out_features))

        self.scaling = alpha / rank
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self):
        for i in range(self.num_bands):
            nn.init.kaiming_uniform_(self.lora_fc1_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_fc1_B[i])
            nn.init.kaiming_uniform_(self.lora_fc2_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_fc2_B[i])

    def forward(self, x_with_band_info):
        # 分离特征和波段索引
        x = x_with_band_info[:, :-1]
        band_indices = x_with_band_info[:, -1].long()
        
        # --- 处理 fc1 ---
        h_shared = self.fc1(x)
        # 向量化LoRA计算
        # 1. 根据band_indices为每个token收集对应的LoRA权重
        lora_A1_for_tokens = self.lora_fc1_A[band_indices] # (num_tokens, in, rank)
        lora_B1_for_tokens = self.lora_fc1_B[band_indices] # (num_tokens, rank, hidden)
        # 2. 使用bmm进行批处理矩阵乘法
        # x.unsqueeze(1) -> (num_tokens, 1, in)
        # bmm -> (num_tokens, 1, rank) -> squeeze(1) -> (num_tokens, rank)
        lora_h = torch.bmm(x.unsqueeze(1), lora_A1_for_tokens).squeeze(1)
        lora_h = torch.bmm(lora_h.unsqueeze(1), lora_B1_for_tokens).squeeze(1)
        
        h = self.act(h_shared + lora_h * self.scaling)

        # --- 处理 fc2 ---
        out_shared = self.fc2(h)
        # 向量化LoRA计算
        lora_A2_for_tokens = self.lora_fc2_A[band_indices]
        lora_B2_for_tokens = self.lora_fc2_B[band_indices]
        lora_out = torch.bmm(h.unsqueeze(1), lora_A2_for_tokens).squeeze(1)
        lora_out = torch.bmm(lora_out.unsqueeze(1), lora_B2_for_tokens).squeeze(1)

        return out_shared + lora_out * self.scaling

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).mul(self._nonzero_gates.unsqueeze(1))
        zeros = torch.zeros((self._gates.size(0), expert_out[-1].size(1)), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, 
                 num_bands, lora_rank, lora_alpha,
                 noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.num_bands = num_bands
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.experts = nn.ModuleList([
            SharedExpertMLP(input_size + 1, hidden_size, output_size, num_bands, lora_rank, lora_alpha)
            for i in range(num_experts)
        ])
        
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, band_indices, loss_coef=1e-2):
        # xg = x.mean(1)
        xg = x
        gates, load = self.noisy_top_k_gating(xg, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # Augment x with band_indices
        x_with_band_info = torch.cat([x, band_indices.unsqueeze(1).float()], dim=1)
        
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_with_band_info)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y.view(x.shape), loss
