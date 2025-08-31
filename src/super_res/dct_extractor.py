import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFrequencyExtractor(nn.Module):
    """
    A learnable hybrid multi-scale time-frequency feature extractor.
    
    This module replaces fixed DCT transforms or simple linear layers to provide
    richer and more discriminative frequency features for MoE gating networks,
    aiming to achieve a "1+1>2" synergistic effect.

    The module contains two parallel branches:
    1. Multi-Scale CNN Branch: Uses convolution kernels of different sizes to capture
       steady-state features across multiple frequency bands.
    2. Wavelet-like CNN Branch: Uses dilated convolutions to capture localized
       transient features, similar to wavelet transforms.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Args:
            input_size (int): Number of input channels/features.
            output_size (int): Desired output feature dimension.
        """
        super().__init__()
        
        # Split output channels between the two branches
        cnn_branch_out_dim = output_size // 2
        wavelet_branch_out_dim = output_size - cnn_branch_out_dim
        
        # Multi-Scale CNN Branch
        self.kernel_sizes = [1, 2, 3]
        num_filters = self._allocate_filters(cnn_branch_out_dim, len(self.kernel_sizes))
        
        self.cnn_branch_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_size,
                out_channels=num_f,
                kernel_size=k,
                padding=k // 2  # Ensure output length matches input
            ) for k, num_f in zip(self.kernel_sizes, num_filters)
        ])
        
        # Wavelet-like CNN Branch
        self.wavelet_branch_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=wavelet_branch_out_dim,
                kernel_size=3,
                padding=1,
                dilation=1
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=wavelet_branch_out_dim,
                out_channels=wavelet_branch_out_dim,
                kernel_size=3,
                padding=2,  # accommodate dilation=2
                dilation=2
            )
        )

    def _allocate_filters(self, total_dim: int, num_kernels: int):
        """
        Allocate output channels evenly across convolution kernels.
        Any remainder is added to the last kernel to ensure the total matches total_dim.

        Args:
            total_dim (int): Total number of output channels for the branch.
            num_kernels (int): Number of convolution kernels.

        Returns:
            List[int]: Number of filters per kernel.
        """
        base = total_dim // num_kernels
        filters = [base] * num_kernels
        remainder = total_dim - base * num_kernels
        filters[-1] += remainder
        return filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid frequency extractor.

        Args:
            x (torch.Tensor): Input tensor of shape [num_tokens, input_size].

        Returns:
            torch.Tensor: Extracted frequency features of shape [num_tokens, output_size].
        """
        # Reshape for Conv1d: [N, C] -> [N, C, L=1]
        x = x.unsqueeze(2)
        
        # Multi-Scale CNN Branch
        cnn_features = [
            F.adaptive_avg_pool1d(conv(x), 1).squeeze(2) for conv in self.cnn_branch_convs
        ]
        cnn_output = torch.cat(cnn_features, dim=1)
        
        # Wavelet-like CNN Branch
        wavelet_features = self.wavelet_branch_conv(x)
        wavelet_output = F.adaptive_avg_pool1d(wavelet_features, 1).squeeze(2)
        
        # Concatenate branch outputs
        final_output = torch.cat([cnn_output, wavelet_output], dim=1)
        
        return final_output
