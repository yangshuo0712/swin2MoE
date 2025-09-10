
import torch
from typing import Any, List, Dict

__all__ = ["DAFEHook", "resolve_module_by_path"]

def resolve_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Resolve a nested module given a dotted path like "backbone.blocks[3].moe.dafe".
    Supports integer indexing using [idx].
    """
    cur = root
    if not path:
        return cur
    parts = path.split(".")
    for p in parts:
        if "[" in p and p.endswith("]"):
            name, idx = p[:p.index("[")], int(p[p.index("[")+1:-1])
            cur = getattr(cur, name)[idx]
        else:
            cur = getattr(cur, p)
    return cur

class DAFEHook:
    """
    Register a forward hook on a DAFE module that stores its internal debug dict.
    Expected that the DAFE module populates `self.last_debug` during forward:
        self.last_debug = {"y_freq":..., "edge":..., "noise":..., "x_size": (H_t,W_t)}
    Fallback: if `last_debug` is absent, the hook will raise an informative error.
    """
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.records: List[Dict[str, torch.Tensor]] = []

    def _hook(self, mod, inp, out):
        if not hasattr(mod, "last_debug"):
            raise RuntimeError("DAFEHook: module has no attribute `last_debug`. "
                               "Please add `self.last_debug={...}` inside DAFE.forward as discussed.")
        rec = {}
        for k in ["y_freq", "edge", "noise", "x_size"]:
            if k not in mod.last_debug:
                raise RuntimeError(f"DAFEHook: `last_debug` missing key `{k}`.")
            rec[k] = mod.last_debug[k]
        self.records.append(rec)

    def __enter__(self):
        self.handles.append(self.module.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
