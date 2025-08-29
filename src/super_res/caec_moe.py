import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexityHead(nn.Module):
    def __init__(self, in_features, hidden_features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        return self.net(x)  # [N_tokens, 1]

class CAEC_MoE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_experts, k, num_bands, lora_rank, lora_alpha,
                 capacity_factor=1.25, complexity_loss_weight=0.1, eps=1e-6):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.complexity_loss_weight = complexity_loss_weight
        self.eps = eps

        # Experts (user's SharedExpertMLP expects input_size + 1 for band index)
        from .ps_moe import SharedExpertMLP
        self.experts = nn.ModuleList([
            SharedExpertMLP(input_size + 1, hidden_size, output_size, num_bands, lora_rank, lora_alpha)
            for _ in range(num_experts)
        ])

        # Routing
        self.gate = nn.Linear(input_size, num_experts, bias=False)
        self.complexity_head = ComplexityHead(input_size)
        self.lambda_complexity = nn.Parameter(torch.ones(1))

        # residual skip projection: map input feature -> output dim for tokens that are not processed
        self.skip_proj = nn.Linear(input_size, output_size)

    def complexity_utilization_loss(self, complexity_scores, eps=1e-8):
        # complexity_scores: [N_tokens, 1]
        # return negative variance of scalar scores
        s = complexity_scores.view(-1)
        var_s = torch.var(s)
        return -torch.log(var_s + eps)

    def forward(self, x, band_indices):
        """
        x: [N_tokens, C_in] -- This is the key change
        band_indices: [N_tokens] -- integer band indices
        returns: final_output [N_tokens, output_size], complexity_loss (scalar)
        """
        num_tokens, in_channels = x.shape
        assert in_channels == self.input_size, "input channels mismatch"

        x_flat = x
        device = x.device
        dtype = x.dtype

        # 1. Calculate complexity and base affinity
        complexity_scores = self.complexity_head(x_flat)      # [N, 1]
        base_affinity = self.gate(x_flat)                    # [N, E]

        # 2. Modulate affinity with complexity (broadcast)
        # complexity_scores: [N,1] -> broadcast to [N,E]
        affinity_caec = base_affinity + self.lambda_complexity * complexity_scores  # [N, E]

        # 3. Expert Choice Routing: for each expert, pick top C tokens
        capacity = max(1, int((num_tokens / float(self.num_experts)) * self.capacity_factor))

        # We want top-C tokens PER EXPERT. affinity_caec.t() has shape [E, N], topk along dim=1 -> per-expert top-C
        topk_vals, topk_indices = torch.topk(affinity_caec.t(), capacity, dim=1)  # both [E, C]

        # topk_vals: raw affinities per expert per chosen token -> normalize within each expert's C choices
        # Softmax across the C choices for each expert to get relative weights inside that expert
        topk_vals_normalized = F.softmax(topk_vals, dim=1)  # [E, C]

        # Flatten expert-wise selections into a single list
        expert_ids = torch.arange(self.num_experts, device=device).unsqueeze(1).repeat(1, capacity).flatten()  # [E*C]
        token_indices_flat = topk_indices.flatten()                # [E*C]
        gates_flat = topk_vals_normalized.flatten()               # [E*C] -- non-negative, sum per expert == 1

        # Dispatch tokens to experts: gather token features and band indices
        dispatched_x = x_flat[token_indices_flat]                 # [E*C, input_size]
        dispatched_band_indices = band_indices.view(-1)[token_indices_flat].unsqueeze(1).to(dtype)  # [E*C, 1]

        # Concat band info as in user's original approach (assumes SharedExpertMLP expects band appended)
        expert_inputs_with_band_info = torch.cat([dispatched_x, dispatched_band_indices], dim=1)  # [E*C, input_size+1]

        # Now split into per-expert input chunks in the same order as expert_ids
        # Each expert gets exactly `capacity` inputs (some tokens may be selected by multiple experts)
        # We'll prepare a list for feeding to each expert
        expert_inputs_split = expert_inputs_with_band_info.split(capacity, dim=0)  # list len E, each [C, input_size+1]

        # 4. Process through experts (in parallel loop)
        expert_outputs_list = []
        for e in range(self.num_experts):
            # ensure shape: if batch is tiny and capacity==1, splitting still yields correct shape
            inp = expert_inputs_split[e]
            # If for some reason inp.shape[0] < 1, feed a zero tensor (shouldn't happen due to capacity>=1)
            if inp.shape[0] == 0:
                # create zeros to keep ordering
                expert_outputs_list.append(torch.zeros((0, self.output_size), device=device, dtype=dtype))
            else:
                out_e = self.experts[e](inp)  # expected shape [C, output_size]
                expert_outputs_list.append(out_e)

        # Concatenate outputs back in the flattened order corresponding to expert_ids / token_indices_flat
        expert_outputs_concat = torch.cat(expert_outputs_list, dim=0)  # [E*C, output_size]

        # 5. Combine expert outputs back to tokens
        # We'll compute:
        #   output_sums[token] += gate * expert_output
        #   weight_sums[token] += gate
        output_sums = torch.zeros((num_tokens, self.output_size), device=device, dtype=dtype)
        weight_sums = torch.zeros((num_tokens,), device=device, dtype=dtype)

        # Multiply expert outputs by gates and accumulate
        weighted_outputs = expert_outputs_concat * gates_flat.unsqueeze(1)  # [E*C, output_size]
        output_sums.index_add_(0, token_indices_flat, weighted_outputs)     # accumulate weighted outputs
        weight_sums.index_add_(0, token_indices_flat, gates_flat)          # accumulate weights

        # Avoid division by zero: tokens not selected will have weight_sums == 0
        denom = weight_sums.unsqueeze(1) + self.eps
        aggregated = output_sums / denom   # [N, output_size]  (for tokens with weight==0 gives near-zero)

        # Residual skip projection for all tokens
        residual = self.skip_proj(x_flat)  # [N, output_size]

        # Final output: residual + aggregated expert contribution
        combined_outputs = residual + aggregated  # [N, output_size]
        final_output = combined_outputs # The calling function will reshape it

        # 6. Complexity utilization loss
        complexity_loss = self.complexity_utilization_loss(complexity_scores) * self.complexity_loss_weight

        return final_output, complexity_loss
