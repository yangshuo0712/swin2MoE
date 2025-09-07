# chk_loader.py
# ---------------------------------------------------------------------
# Robust checkpoint save/load utilities for (DDP + torch.compile) setups
# ---------------------------------------------------------------------

import os
from typing import Any, Dict, Tuple, List, Optional
from collections import OrderedDict

import torch
import torch.distributed as dist

from utils import is_dist_avail_and_initialized, is_main_process


# ------------------------- Unwrap helpers -------------------------

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwrap a model from torch.compile (._orig_mod) and DDP (.module)
    wrappers until the base model is reached.
    """
    m = model
    changed = True
    # Iterate until no more wrappers
    while changed:
        changed = False
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
            changed = True
        if hasattr(m, "module"):
            m = m.module
            changed = True
    return m


# ---------------------- Checkpoint path utils ----------------------

def _extract_epoch_from_fname(fname: str) -> Optional[int]:
    """
    Extract epoch number from filename like 'model-18.pt'.
    Returns None if the pattern doesn't match.
    """
    if not (fname.startswith("model-") and fname.endswith(".pt")):
        return None
    try:
        core = fname[len("model-") : -len(".pt")]
        return int(core)
    except Exception:
        return None


def _select_last_checkpoint(filenames: List[str]) -> str:
    """
    From a list of filenames, select the one with the largest epoch number.
    Raises if none matches 'model-XX.pt'.
    """
    candidates: List[Tuple[int, str]] = []
    for fn in filenames:
        ep = _extract_epoch_from_fname(fn)
        if ep is not None:
            candidates.append((ep, fn))
    if not candidates:
        raise FileNotFoundError("No valid checkpoint files found (expected 'model-XX.pt').")
    candidates.sort(key=lambda x: x[0])  # ascending by epoch
    return candidates[-1][1]


# ---------------------- Prefix strip utilities ----------------------

def _strip_prefixes_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Return a copy of state_dict with known wrapper prefixes removed if present.
    Safe to call even if no such prefixes exist.
    """
    prefixes = ("module.", "_orig_mod.", "model.")
    need_strip = any(any(k.startswith(p) for p in prefixes) for k in sd.keys())
    if not need_strip:
        return sd

    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


# ---------------------- Public API: load/save ----------------------

def load_checkpoint(cfg: Any) -> Dict[str, Any]:
    """
    Load a checkpoint object from disk (not yet applied to model).
    - If cfg.epoch != -1, load that specific file.
    - Else, pick the latest 'model-XX.pt' under {cfg.output}/checkpoints.

    Returns: the object returned by torch.load (a dict with 'model_state_dict', etc.)
    """
    dir_chk = os.path.join(cfg.output, "checkpoints")
    if not os.path.isdir(dir_chk):
        raise FileNotFoundError(f"Checkpoint directory not found: {dir_chk}")

    if getattr(cfg, "epoch", -1) != -1:
        path = os.path.join(dir_chk, f"model-{cfg.epoch:02d}.pt")
    else:
        fnames = os.listdir(dir_chk)
        last = _select_last_checkpoint(fnames)
        path = os.path.join(dir_chk, last)

    if is_main_process():
        print(f"[ckpt] Loading checkpoint from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    map_location = getattr(cfg, "device", "cpu")
    return torch.load(path, map_location=map_location)

def load_state_dict_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    checkpoint: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Load model/optimizer/scheduler state from checkpoint.
    - Always load weights into the *unwrapped* base model.
    - Auto-strip known wrapper prefixes in old checkpoints.
    Returns: (next_epoch, index)
    """
    if is_main_process():
        print("[ckpt] Loading model state...")

    target = unwrap_model(model)
    model_state = target.state_dict()

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'.")

    ckpt_sd = _strip_prefixes_from_state_dict(checkpoint["model_state_dict"])

    # Optional sanity heads for quick visual comparison
    if is_main_process():
        model_head = list(model_state.keys())[:5]
        ckpt_head = list(ckpt_sd.keys())[:5]
        print(f"[ckpt][sanity] model keys head: {model_head}")
        print(f"[ckpt][sanity] ckpt  keys head: {ckpt_head}")

    missing, unexpected = target.load_state_dict(ckpt_sd, strict=False)

    if is_main_process():
        if missing:
            print(f"[ckpt][load] Missing keys (in model, not in ckpt) [<=15 shown]: {missing[:15]} | total={len(missing)}")
        if unexpected:
            print(f"[ckpt][load] Unexpected keys (in ckpt, not in model) [<=15 shown]: {unexpected[:15]} | total={len(unexpected)}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        if is_main_process():
            print("[ckpt] Loading optimizer state...")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if is_main_process():
            print("[ckpt] Loading scheduler state...")
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            if is_main_process():
                print(f"[ckpt][warn] Failed to load scheduler state: {e}")

    epoch = int(checkpoint.get("epoch", -1))
    index = int(checkpoint.get("index", 0))
    return epoch + 1, index


def load_state_dict_model_only(
    model: torch.nn.Module,
    checkpoint: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Load only the model weights from checkpoint into the unwrapped base model.
    Returns: (next_epoch, index)
    """
    if is_main_process():
        print("[ckpt] Loading model state ONLY...")

    target = unwrap_model(model)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'.")

    ckpt_sd = _strip_prefixes_from_state_dict(checkpoint["model_state_dict"])
    missing, unexpected = target.load_state_dict(ckpt_sd, strict=False)

    if is_main_process():
        if missing:
            print(f"[ckpt][load-only] Missing keys [<=15]: {missing[:15]} | total={len(missing)}")
        if unexpected:
            print(f"[ckpt][load-only] Unexpected keys [<=15]: {unexpected[:15]} | total={len(unexpected)}")

    epoch = int(checkpoint.get("epoch", -1))
    index = int(checkpoint.get("index", 0))
    return epoch + 1, index


def save_state_dict_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    index: int,
    cfg: Any,
) -> None:
    """
    Save a *clean* checkpoint (weights from unwrapped base model).
    - Only main process writes to disk.
    - Honors cfg.snapshot_interval (defaults to 1 if absent).
    """
    n_epoch = epoch + 1
    snapshot_interval = int(getattr(cfg, "snapshot_interval", 1))
    if snapshot_interval <= 0:
        snapshot_interval = 1

    if n_epoch % snapshot_interval != 0:
        return

    # Sync before saving to ensure all processes finish previous step
    if is_dist_avail_and_initialized():
        dist.barrier()

    if is_main_process():
        dir_chk = os.path.join(cfg.output, "checkpoints")
        os.makedirs(dir_chk, exist_ok=True)
        path = os.path.join(dir_chk, f"model-{n_epoch:02d}.pt")

        base_state = unwrap_model(model).state_dict()

        checkpoint = {
            "epoch": epoch,
            "index": index,
            "model_state_dict": base_state,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            try:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            except Exception as e:
                print(f"[ckpt][warn] scheduler.state_dict() failed: {e}")

        torch.save(checkpoint, path)
        print(f"[ckpt] Checkpoint saved to {path}")

    # Sync again to let non-main ranks proceed after the file is safely written
    if is_dist_avail_and_initialized():
        dist.barrier()
