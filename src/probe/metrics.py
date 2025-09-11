import torch
from typing import Dict, Tuple

__all__ = ["spectral_summary", "r2_linear", "cohens_d", "downsample_mask"]

def spectral_summary(y_freq: torch.Tensor, B: int, Ht: int, Wt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the spectral centroid and peak frequency for each token from its spectral magnitudes.

    Args:
        y_freq: A tensor of absolute spectral magnitudes per token, shape [N, K].
        B: Batch size.
        Ht: Height of the token grid.
        Wt: Width of the token grid.

    Returns:
        A tuple containing:
        - mu (torch.Tensor): The spectral centroid for each token, shape [B, Ht, Wt].
        - k_peak (torch.Tensor): The index of the peak frequency component for each token, shape [B, Ht, Wt].
    """
    N, K = y_freq.shape
    y = y_freq.abs().view(B, Ht, Wt, K)
    
    # Calculate weights for the centroid calculation
    w = y / (y.sum(-1, keepdim=True) + 1e-6)
    
    # Create a grid of frequency indices
    k_grid = torch.arange(K, device=y.device).view(1, 1, 1, K)
    
    # Calculate spectral centroid (weighted average of frequencies)
    mu = (w * k_grid).sum(-1)
    
    # Find the peak frequency index
    k_peak = y.argmax(-1)
    
    return mu, k_peak

def r2_linear(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculates the R^2 (coefficient of determination) for a linear regression model
    fitting the relationship: y_pred ~ a * y_true + b.

    This function uses torch.linalg.lstsq, which replaces the deprecated torch.lstsq.
    """
    yp = y_pred.reshape(-1).to(torch.float32)
    yt = y_true.reshape(-1).to(torch.float32)

    # Remove non-finite values (NaN, infinity) to ensure valid computation
    mask = torch.isfinite(yp) & torch.isfinite(yt)
    yp = yp[mask]
    yt = yt[mask]

    if yt.numel() < 2:
        return 0.0

    yt_mean = yt.mean()
    ss_tot = ((yt - yt_mean) ** 2).sum()

    # If the true values are nearly constant, R^2 is undefined, so return 0.
    if ss_tot <= 1e-12:
        return 0.0

    # Set up the design matrix A = [yt, 1] and the target vector b = yp
    # Use float64 for improved numerical stability during the least squares calculation.
    A = torch.stack([yt, torch.ones_like(yt)], dim=1).to(torch.float64)  # Shape: [N, 2]
    b = yp.to(torch.float64).unsqueeze(1)                               # Shape: [N, 1]

    # Note the argument order for lstsq: lstsq(input, driver) where input is A and driver is b
    solution = torch.linalg.lstsq(A, b).solution.squeeze(1)  # Shape: [2]
    a, c = solution[0].to(torch.float32), solution[1].to(torch.float32)

    # Calculate the sum of squared residuals
    yp_hat = a * yt + c
    ss_res = ((yp - yp_hat) ** 2).sum()

    r2 = 1.0 - (ss_res / ss_tot)
    
    # Clamp the R^2 value to the valid range [0.0, 1.0].
    return float(r2.clamp(min=0.0, max=1.0).item())

def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calculates Cohen's d, a measure of effect size between two groups.
    
    Args:
        a: A tensor containing samples from the first group.
        b: A tensor containing samples from the second group.
        
    Returns:
        The calculated Cohen's d value as a float.
    """
    a = a.float().view(-1)
    b = b.float().view(-1)
    
    # Calculate unbiased variance for each group
    va = a.var(unbiased=True)
    vb = b.var(unbiased=True)
    
    # Calculate the pooled standard deviation
    sp = torch.sqrt(0.5 * (va + vb) + 1e-6)
    
    # Calculate Cohen's d
    d = (a.mean() - b.mean()) / sp
    return float(d.item())

def downsample_mask(M: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """
    Downsamples a boolean mask from pixel resolution to token resolution using
    adaptive average pooling followed by thresholding.

    Args:
        M: The input boolean mask at pixel resolution, shape [B, H, W].
        Ht: The target height of the token grid.
        Wt: The target width of the token grid.

    Returns:
        A downsampled boolean mask at token resolution, shape [B, Ht, Wt].
    """
    # Use adaptive average pooling to downscale the mask
    mt = torch.nn.functional.adaptive_avg_pool2d(M.float().unsqueeze(1), (Ht, Wt)).squeeze(1)
    
    # Convert back to a boolean mask by thresholding at 0.5
    return (mt > 0.5)
