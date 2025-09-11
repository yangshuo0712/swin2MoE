# src/probe/capture_hooks.py
from typing import Any, Dict, List, Tuple

import torch

from super_res.distortionAware import DistortionAwareFrequencyExtractor


def _norm_token_hw_from_xsize(x_size) -> Tuple[int, int]:
    """
    Normalizes and returns the token grid dimensions (Height, Width) from the x_size input.

    Args:
        x_size: The input size, which can be a list, tuple, or tensor.

    Returns:
        A tuple containing the height and width of the token grid.

    Raises:
        RuntimeError: If x_size is None.
        TypeError: If x_size is of an unsupported type.
    """
    if x_size is None:
        raise RuntimeError("ProbeCollector: x_size is None, cannot determine token grid size.")
    if isinstance(x_size, (list, tuple)):
        Ht, Wt = int(x_size[0]), int(x_size[1])
    elif torch.is_tensor(x_size):
        xs = x_size.detach().cpu().tolist()
        Ht, Wt = int(xs[0]), int(xs[1])
    else:
        raise TypeError(f"Unknown type for x_size: {type(x_size)}")
    return Ht, Wt


def _infer_B_from_tokens(tokens: torch.Tensor, Ht: int, Wt: int) -> int:
    """
    Infers the batch size (B) from the tokens tensor shape.
    It supports tensors with 2 dimensions [N, C] where N = B*Ht*Wt,
    or 3 dimensions [B, T, C].

    Args:
        tokens: The input token tensor.
        Ht: The height of the token grid.
        Wt: The width of the token grid.

    Returns:
        The inferred batch size.
    """
    if tokens.dim() == 2:
        # Assumes tokens are in [N, C] format, where N = B * Ht * Wt
        N = tokens.shape[0]
        T = Ht * Wt
        if N % T != 0:
            raise AssertionError(
                f"The first dimension N={N} of tokens is not divisible by Ht*Wt={T}. "
                f"Cannot recover batch size."
            )
        return N // T
    elif tokens.dim() == 3:
        # Assumes tokens are in [B, T, C] format
        return tokens.shape[0]
    else:
        raise ValueError(f"Unsupported shape for tokens: {tuple(tokens.shape)}")


def _to_flat_yfreq(y_freq: torch.Tensor, B: int, Ht: int, Wt: int) -> torch.Tensor:
    """
    Reshapes various frequency-domain feature formats into a standard [N, K] tensor,
    where N = B*Ht*Wt.

    Args:
        y_freq: The frequency-domain feature tensor.
        B: The batch size.
        Ht: The height of the token grid.
        Wt: The width of the token grid.

    Returns:
        The flattened feature tensor of shape [N, K].
    """
    if y_freq.dim() == 2:
        # Expected shape is [N, K]
        N = y_freq.shape[0]
        expected_N = B * Ht * Wt
        if N != expected_N:
            raise AssertionError(f"y_freq first dimension N={N} does not match B*Ht*Wt={expected_N}")
        return y_freq

    if y_freq.dim() == 4:
        B0 = y_freq.shape[0]
        if B0 != B:
            raise AssertionError(f"Batch size mismatch in y_freq: {B0} vs {B}")
        
        # Case 1: [B, Ht, Wt, K]
        if y_freq.shape[1] == Ht and y_freq.shape[2] == Wt:
            _, _, _, K = y_freq.shape
            return y_freq.reshape(B * Ht * Wt, K)
        
        # Case 2: [B, K, Ht, Wt]
        if y_freq.shape[2] == Ht and y_freq.shape[3] == Wt:
            _, K, _, _ = y_freq.shape
            return y_freq.permute(0, 2, 3, 1).reshape(B * Ht * Wt, K)
        
        raise ValueError(f"Unsupported 4D shape for y_freq: {tuple(y_freq.shape)}")

    if y_freq.dim() == 3:
        # Case 3: [B, T, K] where T is expected to be Ht*Wt
        B0, T, K = y_freq.shape
        expected_T = Ht * Wt
        if not (B0 == B and T == expected_T):
             raise AssertionError(
                f"Expected y_freq shape [B, {expected_T}, K], but got {tuple(y_freq.shape)}"
            )
        return y_freq.reshape(B * Ht * Wt, K)

    raise ValueError(f"Unsupported shape for y_freq: {tuple(y_freq.shape)}")


def _to_bhw_map(t: torch.Tensor, B: int, Ht: int, Wt: int) -> torch.Tensor:
    """
    Normalizes an edge or noise map to the shape [B, Ht, Wt].

    Compatible input shapes:
      - [B, Ht, Wt]
      - [B, 1, Ht, Wt]
      - [B, Ht*Wt]
      - [N] where N = B*Ht*Wt
      - [N, 1] where N = B*Ht*Wt

    Args:
        t: The input tensor (edge or noise map).
        B: The batch size.
        Ht: The height of the token grid.
        Wt: The width of the token grid.

    Returns:
        The normalized tensor of shape [B, Ht, Wt].
    """
    N = B * Ht * Wt

    # Already in [B, Ht, Wt] format
    if t.dim() == 3 and t.shape == (B, Ht, Wt):
        return t

    # Shape: [B, 1, Ht, Wt]
    if t.dim() == 4 and t.shape[1] == 1 and t.shape[2:] == (Ht, Wt):
        return t.squeeze(1)

    # Shape: [B, Ht*Wt]
    if t.dim() == 2 and t.shape == (B, Ht * Wt):
        return t.view(B, Ht, Wt)

    # Shape: [N]
    if t.dim() == 1 and t.numel() == N:
        return t.view(B, Ht, Wt)

    # Shape: [N, 1]
    if t.dim() == 2 and t.shape[0] == N and t.shape[1] == 1:
        return t.view(B, Ht, Wt)

    # A check for multi-channel tensors that cannot be mapped
    if t.dim() == 2 and t.shape[0] == N and t.shape[1] > 1:
        raise ValueError(f"Cannot map tensor of shape [N, C] to [B, Ht, Wt] when C={t.shape[1]} > 1.")

    raise ValueError(f"Cannot normalize tensor to [B, Ht, Wt]; current shape is {tuple(t.shape)}.")


class ProbeCollector:
    """
    A collector class that uses PyTorch hooks to capture intermediate outputs
    from DistortionAwareFrequencyExtractor modules during a forward pass.
    """
    def __init__(self):
        self.records: Dict[str, List[Dict[str, Any]]] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        # The hook function to be registered.
        def _hook(mod, args, kwargs, output):
            # `with_kwargs=True` in register_forward_hook allows us to access keyword arguments.
            tokens = args[0]  # Note: tokens are expected to be of shape [N, C]
            x_size = kwargs.get("x_size", None)

            with torch.no_grad():
                Ht, Wt = _norm_token_hw_from_xsize(x_size)
                # Infer the batch size B from the total number of tokens.
                B = _infer_B_from_tokens(
                    tokens, Ht, Wt
                )

                # Get outputs from internal branches
                y_freq = mod._freq_branch(tokens)
                edge, noise = mod._edge_noise_branch(tokens, x_size)

                # Normalize shapes for consistent data collection
                y_freq_flat = _to_flat_yfreq(y_freq, B, Ht, Wt).detach().cpu()  # Shape: [N, K]
                edge_bhw = _to_bhw_map(edge, B, Ht, Wt).detach().cpu()        # Shape: [B, Ht, Wt]
                noise_bhw = _to_bhw_map(noise, B, Ht, Wt).detach().cpu()      # Shape: [B, Ht, Wt]

            # Store the collected data
            self.records.setdefault(name, []).append(
                {
                    "y_freq": y_freq_flat,
                    "edge": edge_bhw,
                    "noise": noise_bhw,
                    "x_size": (Ht, Wt),
                }
            )

        return _hook

    def attach(self, model: torch.nn.Module):
        """
        Attaches forward hooks to all DistortionAwareFrequencyExtractor modules in the model.
        """
        self.detach() # Ensure no old hooks are present
        for name, module in model.named_modules():
            if isinstance(module, DistortionAwareFrequencyExtractor):
                print(f"Attaching hook to: {name}")
                handle = module.register_forward_hook(
                    self._make_hook(name), with_kwargs=True
                )
                self.handles.append(handle)

    def detach(self):
        """
        Removes all attached hooks.
        """
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear_records(self):
        """
        Clears all captured data.
        """
        self.records.clear()
