import torch
import torch.nn as nn
from .ps_moe import MoE # Import the existing MoE implementation

class ComplexityPredictionHead(nn.Module):
    """
    A lightweight MLP to predict the visual complexity of an input token.
    Outputs a scalar value used to modulate the router's logits.
    """
    def __init__(self, in_features: int, hidden_features: int = 32):
        """
        Initializes the Complexity Prediction Head.

        Args:
            in_features (int): The feature dimension of the input token (d_model).
            hidden_features (int): The dimension of the internal hidden layer,
                                   kept small to maintain a low overhead.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the complexity score.

        Args:
            x (torch.Tensor): The input token tensor, with shape (num_tokens, in_features).

        Returns:
            torch.Tensor: A tensor of complexity scores, with shape (num_tokens, 1).
        """
        return self.net(x)

class ComplexityAwareMoE(MoE):
    """
    An MoE layer integrated with Complexity-Aware Dynamic Routing (CADR).
    This module extends the standard MoE by adding a complexity-biased routing mechanism.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. Complexity Prediction Head
        self.complexity_head = ComplexityPredictionHead(
            in_features=self.input_size,
            hidden_features=32  # Kept lightweight
        )
        
        # 2. Learnable expert complexity bias vector
        self.expert_complexity_bias = nn.Parameter(torch.zeros(self.num_experts))

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        Overrides the gating mechanism to integrate the complexity-aware bias.
        """
        # 1. Calculate base routing logits
        clean_logits = x @ self.w_gate
        
        # 2. Calculate complexity score
        complexity_score = self.complexity_head(x) # shape: (num_tokens, 1)
        
        # 3. Introduce expert complexity bias
        # Broadcasting is used: (num_tokens, 1) * (1, num_experts) -> (num_tokens, num_experts)
        bias = complexity_score * self.expert_complexity_bias.unsqueeze(0)
        
        # 4. Synthesize final logits
        final_logits = clean_logits + bias
        
        # Apply the original top-k and noise logic
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = final_logits + (torch.randn_like(final_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = final_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, final_logits, noise_stddev if self.training else clean_logits, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
            
        return gates, load
