import torch
import torch.nn as nn

from .ps_moe import SharedExpertMLP
from .dct_extractor import  HybridFrequencyExtractor

class FreqAwareExpertChoiceMoE(nn.Module):
    """
    A Mixture-of-Experts (MoE) layer with Frequency-Aware Expert Choice routing.
    This module combines:
    1. Frequency-aware routing: The router makes decisions based on both spatial
       and frequency domain features of the input tokens.
    2. Expert Choice routing: Experts actively select the top-k tokens they are
       most confident in handling, rather than tokens being routed to experts.
    
    This implementation is a programmable version of the FEC-DPS-MOE architecture
    from the provided PDF. It relies on existing Parameter-Shared (PS) experts 
    for multi-band data handling and adds the custom routing logic on top.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_experts: int, 
                 num_bands: int, lora_rank: int, lora_alpha: float,
                 capacity_factor: float = 1.25, dct_freq_features: int = 64):
        """
        Initializes the FreqAwareExpertChoiceMoE module.

        Args:
            input_size (int): The feature dimension of the input tokens.
            output_size (int): The feature dimension of the output.
            hidden_size (int): The hidden layer size for each expert's MLP.
            num_experts (int): The total number of experts in the mixture.
            num_bands (int): The number of spectral bands in the input data.
            lora_rank (int): The rank of the LoRA adapters within the experts.
            lora_alpha (float): The scaling factor for the LoRA adapters.
            capacity_factor (float): Factor to determine each expert's capacity.
            dct_freq_features (int): The dimensionality of the extracted frequency features.
        """
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.dct_freq_features = dct_freq_features

        # Experts (using the existing SharedExpertMLP)
        self.experts = nn.ModuleList([
            SharedExpertMLP(input_size + 1, hidden_size, output_size, num_bands, lora_rank, lora_alpha)
            for _ in range(num_experts)
        ])

        # A simple linear layer to simulate frequency feature extraction (e.g., from DCT)
        # In a full implementation, this could be a small CNN or a more complex block
        self.dct_extractor = nn.Linear(input_size, dct_freq_features, bias=False)

        # Gating network: maps concatenated spatial+frequency features to expert affinities
        self.gating_network = nn.Linear(input_size + dct_freq_features, num_experts, bias=False)

    def _extract_dct_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        A placeholder for a real DCT feature extraction method.
        Here, we use a simple linear layer to transform the input token features.
        
        Args:
            tokens (torch.Tensor): Input tokens of shape [num_tokens, input_size].

        Returns:
            torch.Tensor: Extracted frequency-like features of shape [num_tokens, dct_freq_features].
        """
        return self.dct_extractor(tokens)

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared coefficient of variation for a tensor.
        This is used as a load-balancing loss component.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor, band_indices: torch.Tensor, loss_coef: float = 1e-2):
        """
        Forward pass for the FreqAwareExpertChoiceMoE layer.

        Args:
            x (torch.Tensor): Input tokens of shape [num_tokens, input_size].
            band_indices (torch.Tensor): Integer band indices for each token, shape [num_tokens].
            loss_coef (float): Multiplier for the load-balancing loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - final_output: The combined output of all experts, shape [num_tokens, output_size].
                - loss: The load-balancing loss for the MoE layer.
        """
        num_tokens, in_channels = x.shape
        device = x.device
        dtype = x.dtype

        # Step 1: Extract frequency features and concatenate
        freq_features = self._extract_dct_features(x)
        enhanced_tokens = torch.cat([x, freq_features], dim=1)  # [N, C_in + D_freq]

        # Step 2: Compute frequency-aware affinity matrix
        affinity_scores = self.gating_network(enhanced_tokens)  # [N, E]

        # Step 3: Expert Choice Routing
        expert_capacity = max(1, int((num_tokens / float(self.num_experts)) * self.capacity_factor))
        k = min(expert_capacity, num_tokens)

        expert_scores_transposed = affinity_scores.transpose(0, 1)  # [E, N]
        top_k_scores, top_k_indices = torch.topk(expert_scores_transposed, k=k, dim=1)

        # Step 4: Create dispatch mask and gating weights
        dispatch_mask = torch.zeros((num_tokens, self.num_experts), device=device, dtype=torch.bool)  # [N, E]
        rows = top_k_indices  # [E, k]
        cols = torch.arange(self.num_experts, device=device).unsqueeze(1).expand_as(rows)  # [E, k]
        dispatch_mask[rows, cols] = True  #(token, expert)

        uncovered = ~dispatch_mask.any(dim=1)  # [N]
        if uncovered.any():
            best_expert = affinity_scores[uncovered].argmax(dim=1)  # [n_uncovered]
            dispatch_mask[uncovered, best_expert] = True

        masked_scores = affinity_scores.masked_fill(~dispatch_mask, -float('inf'))  # [N, E]
        gating_weights = torch.softmax(masked_scores, dim=1)  # [N, E]
        gating_weights = torch.where(dispatch_mask, gating_weights, torch.zeros_like(gating_weights))

        # Step 5: Dispatch tokens and compute expert outputs
        final_output = torch.zeros((num_tokens, self.output_size), device=device, dtype=dtype)
        
        for e in range(self.num_experts):
            selected_token_indices = top_k_indices[e]
            expert_tokens = x[selected_token_indices]
            expert_band_indices = band_indices[selected_token_indices]
            expert_gating_weights = gating_weights[selected_token_indices, e].unsqueeze(1)
            
            # The SharedExpertMLP expects band_indices appended as a feature
            expert_inputs_with_band_info = torch.cat([expert_tokens, expert_band_indices.float().unsqueeze(1)], dim=1)

            # Expert computation
            expert_output = self.experts[e](expert_inputs_with_band_info)

            # Accumulate weighted outputs using index_add_ for efficiency
            final_output.index_add_(0, selected_token_indices, expert_output * expert_gating_weights)

        # Calculate load-balancing loss
        importance = gating_weights.sum(0)
        load = dispatch_mask.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        return final_output, loss
