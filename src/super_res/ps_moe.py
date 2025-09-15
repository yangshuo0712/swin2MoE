import torch
import torch.nn as nn
import math
from torch.distributions.normal import Normal

class LowRankAdapter(nn.Module):
    """
    the lora module
    """
    def __init__(self, in_features, out_features, rank, alpha=8.0):
        """
        Parameter:
        in_features (int)
        out_features (int)
        rank (int): rank of lora 
        alpha (float) scaling factor of adapter
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.scaling = self.alpha / self.rank
        self.reset_parameters()

    def reset_parameters(self):
        """
        init weight
        - lora_A using Kaiming Uniform init
        - lora_B init to 0
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """
        calculate w@x
        """
        return self.lora_B(self.lora_A(x)) * self.scaling

class SharedExpertMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_bands, rank, alpha):
        super().__init__()
        self.in_features_actual = in_features - 1
        self.num_bands = num_bands

        # sharing weights
        self.fc1 = nn.Linear(self.in_features_actual, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

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
        # 1. Separate features and band indices
        x = x_with_band_info[:, :-1]
        band_indices = x_with_band_info[:, -1].long()
        
        # 2. Create an empty tensor to store the results
        output = torch.empty_like(x)

        # 3. Loop over each unique band present in this batch of tokens
        for band_idx in torch.unique(band_indices):
            # Create a boolean mask to select tokens for the current band
            mask = (band_indices == band_idx)
            
            # Skip if no tokens for this band
            if not mask.any():
                continue
            
            tokens_for_band = x[mask]

            # --- FC1 Layer ---
            # Shared part
            h_shared = self.fc1(tokens_for_band)
            # LoRA part (using standard matrix multiplication)
            lora_A1 = self.lora_fc1_A[band_idx] # Shape: (in, rank)
            lora_B1 = self.lora_fc1_B[band_idx] # Shape: (rank, hidden)
            h_lora = (tokens_for_band @ lora_A1 @ lora_B1) * self.scaling
            
            h = self.act(h_shared + h_lora)

            # --- FC2 Layer ---
            # Shared part
            out_shared = self.fc2(h)
            # LoRA part
            lora_A2 = self.lora_fc2_A[band_idx] # Shape: (hidden, rank)
            lora_B2 = self.lora_fc2_B[band_idx] # Shape: (rank, out)
            out_lora = (h @ lora_A2 @ lora_B2) * self.scaling
            
            out_band = out_shared + out_lora
            
            # 4. Place the computed results back into the correct positions in the output tensor
            output[mask] = out_band.to(output.dtype)
            
        return output

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
        """
        Sum together the expert output, weighted by the gates, in a memory-efficient manner.
        """
        # expert_out is a list of tensors, let's find the device and dtype from the first valid one.
        device = None
        dtype = None
        out_shape = None
        for out in expert_out:
            if out is not None and out.numel() > 0:
                device = out.device
                dtype = out.dtype
                out_shape = out.shape[1:]
                break
        
        # If all experts returned empty tensors, return zeros.
        if device is None:
            # You might need to adjust the output shape based on what the model expects downstream.
            # Here we assume the output size is known from the MoE layer.
            # This is a fallback; in practice, at least one expert should have output.
            output_size = self._gates.size(1) # This is a placeholder, might need adjustment
            return torch.zeros((self._gates.size(0), output_size), device=self._gates.device)

        # Pre-allocate the final combined tensor with zeros.
        combined = torch.zeros((self._gates.size(0),) + out_shape, device=device, dtype=dtype)

        # Get the gate values for each expert's outputs
        expert_gates = torch.split(self._nonzero_gates, self._part_sizes, dim=0)
        
        # Get the original batch indices for each expert's outputs
        expert_batch_indices = self._batch_index.split(self._part_sizes, dim=0)

        # Add each expert's output to the combined tensor at the correct indices.
        for i in range(self._num_experts):
            output_i = expert_out[i]
            if output_i is None or output_i.numel() == 0:
                continue

            if multiply_by_gates and expert_gates[i] is not None:
                # Use broadcasting to apply gate weights per sample
                output_i = output_i * expert_gates[i]
            
            # Use index_add_ for memory-efficient, in-place addition
            combined.index_add_(0, expert_batch_indices[i], output_i.to(combined.dtype))
            
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

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, band_indices, loss_coef=1e-2):
        # x: [N, C]
        # band_indices: [N] or [N,1] or [N,K(one-hot)]
        assert x.dim() == 2, f"x should be 2D [N,C], got {x.shape}"
        assert band_indices.size(0) == x.size(0), \
            f"band_indices first dim must match x: {band_indices.size(0)} vs {x.size(0)}"

        xg = x
        gates, load = self.noisy_top_k_gating(xg, self.training)
        importance = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        # band_indices -> [N,1] 
        if band_indices.dim() == 1:
            bi = band_indices.view(-1, 1)
        elif band_indices.dim() == 2 and band_indices.size(1) == 1:
            bi = band_indices
        elif band_indices.dim() == 2:
            bi = band_indices.argmax(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unexpected band_indices shape: {band_indices.shape}")

        bi = bi.to(device=x.device, dtype=x.dtype)
        x_with_band_info = torch.cat([x, bi], dim=1)   # [N, C+1]

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_with_band_info)   # list of [n_i, C+1]
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)  # [N, C]

        return y.view_as(x), loss

class WeightedSharedExpertMLP(nn.Module):
    """
    Parameter-shared MLP + per-band LoRA, with soft band weights.
    forward(x, band_weights) where band_weights is [N, num_bands] and rows sum to 1.
    """
    def __init__(self, in_features, hidden_features, out_features, num_bands, rank, alpha):
        super().__init__()
        self.num_bands = num_bands
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

        # per-band LoRA parameters
        self.lora_fc1_A = nn.Parameter(torch.zeros(num_bands, in_features,  rank))
        self.lora_fc1_B = nn.Parameter(torch.zeros(num_bands, rank,         hidden_features))
        self.lora_fc2_A = nn.Parameter(torch.zeros(num_bands, hidden_features, rank))
        self.lora_fc2_B = nn.Parameter(torch.zeros(num_bands, rank,           out_features))

        self.scaling = alpha / rank
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self):
        for i in range(self.num_bands):
            nn.init.kaiming_uniform_(self.lora_fc1_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_fc1_B[i])
            nn.init.kaiming_uniform_(self.lora_fc2_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_fc2_B[i])

    def forward(self, x, band_weights):
        """
        x: [N, in_features]
        band_weights: [N, num_bands] (soft weights; one-hot also works)
        """
        # fc1: shared + weighted LoRA
        h_shared = self.fc1(x)  # [N, hidden]
        t1 = torch.einsum('ni, bir -> nbr', x, self.lora_fc1_A)      # [N,B,rank]
        l1 = torch.einsum('nbr, brh -> nbh', t1, self.lora_fc1_B)    # [N,B,hidden]
        delta1 = (band_weights.unsqueeze(-1) * l1).sum(dim=1) * self.scaling
        h = self.act(h_shared + delta1)

        # fc2: shared + weighted LoRA
        out_shared = self.fc2(h)  # [N, out]
        t2 = torch.einsum('nh, bhr -> nbr', h, self.lora_fc2_A)      # [N,B,rank]
        l2 = torch.einsum('nbr, bro -> nbo', t2, self.lora_fc2_B)    # [N,B,out]
        delta2 = (band_weights.unsqueeze(-1) * l2).sum(dim=1) * self.scaling

        return out_shared + delta2
