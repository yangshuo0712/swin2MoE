import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Top1Gate(nn.Module):
    """
    Switch-Transformer style Top-1 router.
    * d_model: token hidden size
    * n_expert: number of experts
    * noisy:   whether to add Gaussian noise (same as NoisyTopK)
    """
    def __init__(self, d_model: int, n_expert: int, noisy: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(d_model, n_expert, bias=False)
        self.noisy  = noisy

    def forward(self, x: torch.Tensor, noise_epsilon: float = 1e-2):
        """
        x: [batch, d_model]
        returns:
            idx:    [batch]      expert index
            scores: [batch, 1]   gates score
            mask:   [batch, nE]  one-hot matrix
        """
        logits = self.w_gate(x)                   # [B, E]
        if self.noisy and self.training:
            noise  = torch.randn_like(logits) * noise_epsilon
            logits = logits + noise
        idx    = torch.argmax(logits, dim=-1)     # Top-1
        mask   = F.one_hot(idx, num_classes=logits.size(-1)).float()
        scores = torch.sum(mask * logits, dim=-1, keepdim=True)
        return idx, scores, mask
