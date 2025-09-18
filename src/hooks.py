from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch._dynamo as dynamo

try:
    # 复用你 utils 里的统计（如果没有也可用内置fallback）
    from utils import cv as _cv_utils
    from utils import cv2 as _cv2_utils
    from utils import gini as _gini_utils
except Exception:
    _cv_utils = _cv2_utils = _gini_utils = None


def _cv(x: torch.Tensor) -> float:
    if _cv_utils is not None:
        return float(_cv_utils(x.detach()))
    x = x.detach().float()
    m = x.mean().item()
    if m <= 1e-12:
        return 0.0
    return float(x.std(unbiased=False).item() / (m + 1e-12))


def _cv2(x: torch.Tensor) -> float:
    if _cv2_utils is not None:
        return float(_cv2_utils(x.detach()))
    # CV^2
    x = x.detach().float()
    m = x.mean().item()
    if m <= 1e-12:
        return 0.0
    cv = x.std(unbiased=False).item() / (m + 1e-12)
    return float(cv * cv)


def _gini(x: torch.Tensor) -> float:
    if _gini_utils is not None:
        return float(_gini_utils(x.detach()))
    # 快速近似：gini ≈ mean(|x - mean|)/(2*mean)
    x = x.detach().float()
    m = x.mean().item()
    if m <= 1e-12:
        return 0.0
    mad = (x - m).abs().mean().item()
    return float(mad / (2.0 * m + 1e-12))


def _p95(vals: List[float]) -> float:
    if not vals:
        return 0.0
    vs = sorted(vals)
    k = int(0.95 * (len(vs) - 1))
    return float(vs[k])


class MoEHook:
    FINAL_KEYS = (
        "load_cv",
        "load_cv2",
        "load_gini",
        "imp_cv",
        "imp_cv2",
        "imp_gini",
        "top1_share",
    )

    def __init__(
        self,
        model,
        layer_attr_name: Optional[str] = None,  # 若提供，将自动选 {0, mid, last}
        record_layers: Optional[Sequence[int]] = None,  # 层索引（优先于 name_filter）
        name_filter: Optional[str] = None,  # 只 hook 名字里包含此子串的模块
        record_every: int = 100,
        flush_every: int = 400,
        keep_recent: int = 20,
    ):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.attached: List[Tuple[str, str]] = []

        self.record_every = int(record_every)
        self.flush_every = int(flush_every)
        self.keep_recent = int(keep_recent)

        self.layer_attr_name = layer_attr_name
        self.name_filter = name_filter
        self.record_layers = list(record_layers) if record_layers is not None else None

        self._acc: Dict[str, List[float]] = defaultdict(list)

        self._buf: Dict[str, deque] = {
            k: deque(maxlen=self.keep_recent) for k in self.FINAL_KEYS
        }

        self._step = 0

    def _pick_layers(self) -> Optional[List[int]]:
        if self.record_layers is not None:
            return self.record_layers
        if self.layer_attr_name is None:
            return None
        blocks = getattr(self.model, self.layer_attr_name, None)
        if blocks is None:
            return None
        n = len(blocks)
        mids = [n // 2] if n >= 3 else []
        idxs = [0] + mids + [max(0, n - 1)]
        idxs = sorted(set([i for i in idxs if 0 <= i < n]))
        return idxs

    def _is_moe_like(self, m) -> bool:
        return hasattr(m, "experts") and (
            hasattr(m, "gating_network")
            or hasattr(m, "w_gate")
            or ("MoE" in m.__class__.__name__)
        )

    def attach(self):
        picked_layers = set(self._pick_layers() or [])
        layer_counter = -1

        for name, module in self.model.named_modules():
            if not self._is_moe_like(module):
                continue
            if (self.name_filter is not None) and (self.name_filter not in name):
                continue

            use_this = True
            if picked_layers:
                layer_counter += 1
                use_this = layer_counter in picked_layers

            if use_this:
                h = module.register_forward_hook(self._hook_fn())
                self.hooks.append(h)
                self.attached.append((name, module.__class__.__name__))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.attached.clear()
        self._acc.clear()
        for k in self._buf:
            self._buf[k].clear()

    @dynamo.disable()
    def _hook_fn(self):
        def hook(module, inputs, output):
            dbg = getattr(module, "debug_outputs", None)
            if not isinstance(dbg, dict):
                return

            # 1) 直接读取现成标量（若你在模块里已经计算过）
            for k in (
                "cv_importance",
                "cv_load",
                "cv2_importance",
                "cv2_load",
                "gini_importance",
                "gini_load",
            ):
                if k in dbg:
                    v = dbg[k]
                    if torch.is_tensor(v):
                        v = v.detach().float().mean().cpu().item()
                    self._acc[k].append(float(v))

            # 2) 若只有 raw tensor，就临时算一下
            imp = dbg.get("importance", None)
            load = dbg.get("load", None)

            if isinstance(imp, torch.Tensor) and imp.numel() > 0:
                t = imp.detach().float().cpu()
                self._acc["imp_cv"].append(_cv(t))
                self._acc["imp_cv2"].append(_cv2(t))
                self._acc["imp_gini"].append(_gini(t))

            if isinstance(load, torch.Tensor) and load.numel() > 0:
                t = load.detach().float().cpu()
                self._acc["load_cv"].append(_cv(t))
                self._acc["load_cv2"].append(_cv2(t))
                self._acc["load_gini"].append(_gini(t))

                # 最热专家份额（归一化后max）
                s = t
                if s.dim() == 2:  # [B,E] -> 聚合到 [E]
                    s = s.sum(0)
                s = s.clamp_min(1e-12)
                frac = (s / (s.sum() + 1e-12)).max().item()
                self._acc["top1_share"].append(float(frac))

        return hook

    # ====== 训练循环里在“更新步”调用 ======
    def step(self, global_step: int):
        self._step = int(global_step)
        if (self._step % self.record_every) != 0:
            # 不到采样步，仅清一次（保证下一步不会混入旧数据）
            self._acc.clear()
            return

        # 把“层内值列表”→“层间聚合标量”（均值），推进环形缓冲
        for final_key in self.FINAL_KEYS:
            # 兼容 dbg 的历史名字到统一名字
            candidates = {
                "load_cv": ["load_cv", "cv_load"],
                "load_cv2": ["load_cv2", "cv2_load"],
                "load_gini": ["load_gini", "gini_load"],
                "imp_cv": ["imp_cv", "cv_importance"],
                "imp_cv2": ["imp_cv2", "cv2_importance"],
                "imp_gini": ["imp_gini", "gini_importance"],
                "top1_share": ["top1_share"],
            }[final_key]

            vals = []
            for cand in candidates:
                if cand in self._acc and len(self._acc[cand]) > 0:
                    vals.extend(self._acc[cand])

            if vals:
                mean_v = float(sum(vals) / len(vals))
                self._buf[final_key].append(mean_v)

        self._acc.clear()

    # ====== 训练循环里低频调用：真正写 TensorBoard ======
    def flush(self, writer, tag_prefix: str = "moe"):
        if (self._step % self.flush_every) != 0:
            return

        for k, dq in self._buf.items():
            vals = list(dq)
            if not vals:
                continue
            mean_v = float(sum(vals) / len(vals))
            max_v = float(max(vals))
            p95_v = _p95(vals)
            writer.add_scalar(f"{tag_prefix}/{k}/mean", mean_v, self._step)
            writer.add_scalar(f"{tag_prefix}/{k}/max", max_v, self._step)
            writer.add_scalar(f"{tag_prefix}/{k}/p95", p95_v, self._step)
