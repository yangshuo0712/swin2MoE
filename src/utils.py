import torch
import importlib
import numpy as np
import random
import torch.distributed as dist
import os

from enum import Enum
from torchmetrics.classification import MulticlassF1Score

# --- DDP ---
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

# def is_main_process():
#     return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0


def is_main_process():
    if not is_dist_avail_and_initialized():
        return os.environ.get("RANK") is None or os.environ.get("RANK") == "0"
    return dist.get_rank() == 0

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def w_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_fun(fullname):
    path, name = fullname.rsplit('.', 1)
    return getattr(importlib.import_module(path), name)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # self.val = val
        # self.sum += val * n
        # self.count += n
        # self.avg = self.sum / self.count
        v = val.item() if torch.is_tensor(val) else float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # self.avg_item = self.avg.tolist()
        # fmtstr = "{avg_item" + self.fmt + "}"
        # try:
        #     return fmtstr.format(**self.__dict__)
        # except TypeError:
        #     # print a list of elements
        #     fmtstr = "{" + self.fmt + "}"
        #     return ' '.join([
        #         fmtstr for _ in range(len(self.avg_item))
        #     ]).format(*self.avg_item)
        if torch.is_tensor(self.avg):
                avg_val = self.avg.item() if self.avg.ndim == 0 else self.avg.tolist()
        else:
                avg_val = self.avg

        fmtstr = "{avg" + self.fmt + "}"
        return fmtstr.format(avg=avg_val)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(
                "Invalid summary type {}".format(self.summary_type))

        return fmtstr.format(**self.__dict__)


class F1AverageMeter(AverageMeter):
    def __init__(self, cfg, average, **kwargs):
        self.cfg = cfg
        self._cfg_average = average
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._to_update = True
        self.fun = MulticlassF1Score(
            len(self.cfg.classes), average=self._cfg_average
        ).to(self.cfg.device)

    @property
    def avg(self):
        if self._to_update:
            self._avg = self.fun.compute()
            self._to_update = False
        return self._avg

    @avg.setter
    def avg(self, value):
        self._avg = value
        self._to_update = False

    def update(self, val, n=1):
        pred, gt = val
        self.fun.update(pred, gt)
        self._to_update = True


def set_required_grad(model, value):
    for parameters in model.parameters():
        parameters.requires_grad = value

def sync_average_meters(meters, device):
    """Collect all meters' (sum,count) once, then recompute avg locally."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    buf = torch.stack([
        torch.tensor([m.sum, m.count], dtype=torch.float64, device=device)
        for m in meters.values()
    ])                                   # [num_meters, 2]
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    for (m, stats) in zip(meters.values(), buf):
        m.sum = stats[0].item()
        m.count = max(int(stats[1].item()), 1)
        m.avg = m.sum / m.count

def calculate_apc_spc(cfg):
    """
    Calculates and prints a comprehensive parameter report for the model,
    including total parameters, a breakdown of MoE vs. standard MLP layers,
    SPC/APC for the MoE components, and per-block MoE statistics.
    """
    from super_res.model import build_model

    model = build_model(cfg)
    moe_config = cfg.super_res.model.get('MoE_config')

    total_params = sum(p.numel() for p in model.parameters())
    total_mlp_params = 0
    total_moe_params = 0
    total_experts_params = 0
    num_moe_blocks = 0

    per_block_stats = []

    # --- Iterate through transformer blocks ---
    for rstb_layer in model.layers:
        for block_idx, block in enumerate(rstb_layer.residual_group.blocks):
            if hasattr(block, 'is_moe') and block.is_moe:
                num_moe_blocks += 1
                moe_layer = block.mlp
                total_moe_params += sum(p.numel() for p in moe_layer.parameters())

                block_experts_params = 0
                for expert in moe_layer.experts:
                    block_experts_params += sum(p.numel() for p in expert.parameters())
                total_experts_params += block_experts_params

                # --- every block SPC/APC ---
                if moe_config:
                    k = moe_config.get('k', 2)
                    num_experts = moe_config.get('num_experts', 8)
                    params_per_expert = block_experts_params / num_experts
                    shared_params = sum(p.numel() for p in moe_layer.parameters()) - block_experts_params
                    block_spc = block_experts_params
                    block_apc = shared_params + (k * params_per_expert)
                    per_block_stats.append({
                        "block_index": block_idx,
                        "SPC": block_spc,
                        "APC": block_apc
                    })
            else:
                total_mlp_params += sum(p.numel() for p in block.mlp.parameters())

    transformer_blocks_params = total_mlp_params + total_moe_params
    other_params = total_params - transformer_blocks_params

    print("\n" + "="*50)
    print("      Model Parameter Analysis Report")
    print("="*50)

    print(f"\n[ Overall Model Statistics ]")
    print(f"  - Total Parameters:          {total_params / 1e6:.4f} M")
    print(f"  - Transformer Blocks Params: {transformer_blocks_params / 1e6:.4f} M")
    print(f"  - Other Params (Conv, etc.): {other_params / 1e6:.4f} M")

    print(f"\n[ Standard MLP Layers ]")
    if total_mlp_params > 0:
        print(f"  - Total MLP Params: {total_mlp_params / 1e6:.4f} M")
    else:
        print("  - No standard MLP layers found.")

    print(f"\n[ Mixture of Experts (MoE) Layers ]")
    if total_moe_params > 0 and moe_config:
        shared_params_total = total_moe_params - total_experts_params
        params_per_expert = total_experts_params / (num_experts * num_moe_blocks)
        total_spc = total_experts_params
        total_apc = shared_params_total + (k * params_per_expert * num_moe_blocks)

        print(f"  - Total MoE Params (Shared + All Experts): {total_moe_params / 1e6:.4f} M")
        print(f"  - Sparse Parameter Count (SPC):            {total_spc / 1e6:.4f} M")
        print(f"  - Active Parameter Count (APC):            {total_apc / 1e6:.4f} M\n")

        print("  [Per-Block MoE Statistics]")
        for stat in per_block_stats:
            print(f"    - Block {stat['block_index']}: SPC = {stat['SPC']}, APC = {stat['APC']:.0f}")
    else:
        print("  - No MoE layers found or MoE_config is missing.")

    print("="*50 + "\n")
