# probe/capture_hooks.py
import re
from typing import Any, Dict

import torch


class ProbeCollector:
    """Collect DAFE internals via forward hooks without touching forward() or the graph."""

    def __init__(self, include_regex: str = r"dafe", exclude_regex: str | None = None):
        self.include = re.compile(include_regex) if include_regex else None
        self.exclude = re.compile(exclude_regex) if exclude_regex else None
        self.records: Dict[str, Dict[str, Any]] = {}
        self.handles = []

    def _looks_like_dafe(self, m) -> bool:
        cname = m.__class__.__name__.lower()
        if "dafe" in cname or "distortionaware" in cname:
            return True
        return hasattr(m, "freq_res") and hasattr(m, "intensity_head")

    def _make_hook(self, name: str):
        def _hook(mod, inputs, output):
            # DAFE.forward(tokens, x_size=...) -> we use inputs[0], inputs[1] if present
            tokens = inputs[0]  # [N, C]
            x_size = inputs[1] if len(inputs) > 1 else None
            with torch.no_grad():
                y_freq = mod._freq_branch(tokens)  # [N, K]
                if x_size is not None:
                    edge, noise = mod._edge_noise_branch(tokens, x_size)  # [N,1],[N,1]
                else:
                    edge, noise = mod._edge_noise_fallback(tokens)
            # store CPU copies; no attribute writes to the module
            self.records[name] = {
                "y_freq": y_freq.detach().cpu(),
                "edge": edge.detach().cpu(),
                "noise": noise.detach().cpu(),
                "x_size": x_size,
            }

        return _hook

    def attach(self, model):
        for qn, m in model.named_modules():
            if self.include and not self.include.search(qn):
                continue
            if self.exclude and self.exclude.search(qn):
                continue
            if self._looks_like_dafe(m):
                h = m.register_forward_hook(self._make_hook(qn))
                self.handles.append(h)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
