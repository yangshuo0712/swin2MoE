import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist

from utils import is_dist_avail_and_initialized, is_main_process
from collections import OrderedDict

def unwrap_model(model):
    "Return the underlying model if wrapped by DDP/DataParallel."
    return model.module if hasattr(model, "module") else model

def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]

def load_checkpoint(cfg) -> Dict[str, Any]:
    """
    Return a checkpoint dict loaded to cfg.device.
    """
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, f'model-{cfg.epoch:02d}.pt')
    else:
        try:
            fnames = os.listdir(dir_chk)
            path = os.path.join(dir_chk, get_last_epoch(fnames))
        except (IndexError, FileNotFoundError):
            raise FileNotFoundError(f"No checkpoint found in {dir_chk}")

    if is_main_process():
        print(f'load file {path}')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    return torch.load(path, map_location=cfg.device)

def _strip_prefix_if_present(state_dict, prefix):
    """Strip a given prefix from state_dict keys if present."""
    if not any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            name = k[len(prefix):]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_state_dict_model(model, optimizer, scheduler, checkpoint) -> Tuple[int, int]:
    """
    Load both model & optimizer state. Returns (next_epoch, index).
    Handles 'module.' (DDP) and '_orig_mod.' (torch.compile) prefixes automatically.
    """
    if is_main_process():
        print('load model state')
    
    state_dict = checkpoint['model_state_dict']

    # Handle prefixes: first '_orig_mod.' from torch.compile, then 'module.' from DDP.
    state_dict = _strip_prefix_if_present(state_dict, '_orig_mod.')
    state_dict = _strip_prefix_if_present(state_dict, 'module.')
    
    # Now load the state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys while loading state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys while loading state_dict: {unexpected_keys}")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        if is_main_process():
            print('load optimizer state')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        if is_main_process():
            print('load scheduler state')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', -1)
    index = checkpoint.get('index', 0)

    return epoch + 1, index

def load_state_dict_model_only(model, checkpoint) -> Tuple[int, int]:
    """
    Load only the model weights. Returns (next_epoch, index).
    Handles DDP 'module.' and torch.compile '_orig_mod.' prefixes automatically and ignores missing keys.
    """
    if is_main_process():
        print('load model state only')
    
    state_dict = checkpoint['model_state_dict']
    
    # Handle prefixes: first '_orig_mod.' from torch.compile, then 'module.' from DDP.
    state_dict = _strip_prefix_if_present(state_dict, '_orig_mod.')
    state_dict = _strip_prefix_if_present(state_dict, 'module.')
    
    # Load the state dict with strict=False to be more robust against minor mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys while loading state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys while loading state_dict: {unexpected_keys}")

    epoch = checkpoint.get('epoch', -1)
    index = checkpoint.get('index', 0)

    return epoch + 1, index


def save_state_dict_model(model, optimizer, scheduler, epoch, index, cfg):
    """
    Save checkpoint on rank0 only.
    """
    n_epoch = epoch + 1
    if n_epoch % cfg.snapshot_interval != 0:
        return

    # Optionally sync to be safe when others continue after save
    if is_dist_avail_and_initialized():
        dist.barrier(device_ids=[cfg.local_rank])

    if is_main_process():
        dir_chk = os.path.join(cfg.output, 'checkpoints')
        os.makedirs(dir_chk, exist_ok=True)
        path = os.path.join(dir_chk, f'model-{n_epoch:02d}.pt')

        # When saving, unwrap model to get original state_dict
        # This handles both DDP and torch.compile cases
        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, '_orig_mod'):
             model_state_dict = unwrapped_model._orig_mod.state_dict()
        else:
             model_state_dict = unwrapped_model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'index': index,
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, path)

    if is_dist_avail_and_initialized():
        dist.barrier(device_ids=[cfg.local_rank])
