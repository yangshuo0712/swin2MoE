import torch
from collections import defaultdict
from utils import cv, cv2, gini

class MoEHook:
    def __init__(self, model):
        self.hooks = []
        self.stats = defaultdict(list)
        self.model = model
        self.attached = []

    def _hook_fn(self, name):
        def hook(module, inputs, output):
            dbg = getattr(module, 'debug_outputs', None)
            if not isinstance(dbg, dict):
                return
            for key in ['cv_importance','cv_load','cv2_importance','cv2_load','gini_importance','gini_load']:
                if key in dbg:
                    self.stats[f'{name}_{key}'].append(float(dbg[key]))
            imp = dbg.get('importance', None)
            load = dbg.get('load', None)
            if isinstance(imp, torch.Tensor) and imp.numel() > 0:
                self.stats[f'{name}_cv_imp'].append(float(cv(imp)))
                self.stats[f'{name}_cv2_imp'].append(float(cv2(imp)))
                self.stats[f'{name}_gini_imp'].append(float(gini(imp)))
            if isinstance(load, torch.Tensor) and load.numel() > 0:
                self.stats[f'{name}_cv_load'].append(float(cv(load)))
                self.stats[f'{name}_cv2_load'].append(float(cv2(load)))
                self.stats[f'{name}_gini_load'].append(float(gini(load)))
        return hook

    def attach(self, name_filter: str | None = None):
        for name, module in self.model.named_modules():
            is_moe_like = hasattr(module, 'experts') and (
                hasattr(module, 'gating_network') or
                hasattr(module, 'w_gate') or
                ('MoE' in module.__class__.__name__)
            )
            if is_moe_like and (name_filter is None or name_filter in name):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
                self.attached.append((name, module.__class__.__name__))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.attached.clear()

    def get_stats(self):
        return {k: sum(v)/len(v) for k, v in self.stats.items() if len(v) > 0}
