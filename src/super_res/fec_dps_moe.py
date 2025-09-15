import torch
import torch.nn as nn
from typing import Optional

from .ps_moe import WeightedSharedExpertMLP
from .frequency_extractor import HybridFrequencyExtractor
from .distortionAware import DistortionAwareFrequencyExtractor

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
        dct_extractor: str = 'linear',
        dct_K: int = 16,
        proj_hidden: int = 0,
    ):
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
            dct_extractor (str): The type of DCT extractor to use ('linear', 'DCT', 'HybridFrequencyExtractor').
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
            WeightedSharedExpertMLP(input_size + 1, hidden_size, output_size, num_bands, lora_rank, lora_alpha)
            for _ in range(num_experts)
        ])

        # A simple linear layer to simulate frequency feature extraction (e.g., from DCT)
        # In a full implementation, this could be a small CNN or a more complex block
        if dct_extractor == 'HybridFrequencyExtractor':
            self.dct_extractor = HybridFrequencyExtractor(input_size, dct_freq_features)
        elif dct_extractor == 'DCT':
            # Placeholder for DCT implementation
            raise NotImplementedError("DCT extractor is not yet implemented.")
        elif dct_extractor == "DistortionAware":
            self.dct_extractor = DistortionAwareFrequencyExtractor(
                    input_size=input_size,
                    out_dim=dct_freq_features,
                    dct_K=dct_K,
                    proj_hidden=proj_hidden,
                    learnable_scale=True,
                    )
        else: # default to linear
            self.dct_extractor = nn.Linear(input_size, dct_freq_features, bias=False)

        # Gating network: maps concatenated spatial+frequency features to expert affinities
        self.gating_network = nn.Linear(input_size + dct_freq_features, num_experts, bias=False)

    def _extract_dct_features(self, tokens: torch.Tensor, x_size=None) -> torch.Tensor:
        """
        A placeholder for a real DCT feature extraction method.
        Here, we use a simple linear layer to transform the input token features.
        
        Args:
            tokens (torch.Tensor): Input tokens of shape [num_tokens, input_size].

        Returns:
            torch.Tensor: Extracted frequency-like features of shape [num_tokens, dct_freq_features].
        """
        if hasattr(self.dct_extractor, 'forward'):
            try:
                return self.dct_extractor(tokens, x_size=x_size)
            except TypeError:
                return self.dct_extractor(tokens)
        else:
            return self.dct_extractor(tokens)

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared coefficient of variation for a tensor.
        This is used as a load-balancing loss component.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var(correction=0) / (x.float().mean() ** 2 + eps)

def forward(self, x: torch.Tensor, band_weights: torch.Tensor,
                loss_coef: float = 1e-2, x_size: Optional[tuple]=None):
        """
        x: [N, C]
        band_weights: [N, num_bands]  <-- soft weights aligned with tokens
        """
        num_tokens, in_channels = x.shape
        device = x.device
        dtype = x.dtype

        freq_features = self._extract_dct_features(x, x_size=x_size)
        enhanced_tokens = torch.cat([x, freq_features], dim=1)  # [N, C + D_freq]
        affinity_scores = self.gating_network(enhanced_tokens)  # [N, E]

        expert_capacity = max(1, int((num_tokens / float(self.num_experts)) * self.capacity_factor))
        k = min(expert_capacity, num_tokens)
        expert_scores_transposed = affinity_scores.transpose(0, 1)  # [E, N]
        top_k_scores, top_k_indices = torch.topk(expert_scores_transposed, k=k, dim=1)

        dispatch_mask = torch.zeros((num_tokens, self.num_experts), device=device, dtype=torch.bool)
        rows = top_k_indices
        cols = torch.arange(self.num_experts, device=device).unsqueeze(1).expand_as(rows)
        dispatch_mask[rows, cols] = True

        uncovered = ~dispatch_mask.any(dim=1)
        if uncovered.any():
            best_expert = affinity_scores[uncovered].argmax(dim=1)
            dispatch_mask[uncovered, best_expert] = True

        masked_scores = affinity_scores.masked_fill(~dispatch_mask, float('-inf'))
        gating_weights = torch.softmax(masked_scores, dim=1)
        gating_weights = torch.where(dispatch_mask, gating_weights, torch.zeros_like(gating_weights))

        final_output = torch.zeros((num_tokens, self.output_size), device=device, dtype=dtype)

        for e in range(self.num_experts):
            selected_token_indices = torch.nonzero(dispatch_mask[:, e], as_tuple=False).squeeze(1)
            if selected_token_indices.numel() == 0:
                continue

            expert_tokens   = x[selected_token_indices]                       # [n_e, C]
            expert_bw       = band_weights[selected_token_indices]            # [n_e, num_bands]
            expert_gate     = gating_weights[selected_token_indices, e].unsqueeze(1)  # [n_e,1]

            expert_output = self.experts[e](expert_tokens, expert_bw)         # [n_e, C]
            final_output.index_add_(0, selected_token_indices, expert_output * expert_gate)

        importance = gating_weights.sum(0)
        load = dispatch_mask.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        self.debug_outputs = {'importance': importance.detach(), 'load': load.detach()}
        return final_output, loss
