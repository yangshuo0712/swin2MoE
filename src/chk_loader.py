import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist

from utils import is_dist_avail_and_initialized, is_main_process
from collections import OrderedDict
# import pdb

def unwrap_model(model):
    "Return the underlying model if wrapped by DDP/DataParallel."
    return model.module if hasattr(model, "module") else model

def strip_module_prefix(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    """Remove 'module.' prefix added by (D)DP when needed."""
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()}

def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]

def load_checkpoint(cfg) -> Dict[str, Any]:
    """
    Return a checkpoint dict loaded to cfg.device.
    """
    # pdb.set_trace()
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, f'model-{cfg.epoch:02d}.pt')
    else:
        try:
            fnames = os.listdir(dir_chk)
            path = os.path.join(dir_chk, get_last_epoch(fnames))
        except (IndexError, FileNotFoundError):
            raise FileNotFoundError()

    if is_main_process():
        print(f'load file {path}')
    if not os.path.exists(path):
        raise FileNotFoundError()

    return torch.load(path, map_location=cfg.device)


# def load_state_dict_model(model, optimizer, checkpoint):
#     print('load model state')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print('load optimizer state')
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     return checkpoint['epoch'] + 1, checkpoint['index']

# def load_state_dict_model(model, optimizer, checkpoint) -> Tuple[int, int]:
#     """
#     Load both model & optimizer state. Returns (next_epoch, index).
#     Handles possible 'module.' prefixes automatically.
#     """
#     if is_main_process():
#         print('load model state')
#     state_dict = checkpoint['model_state_dict']
#     # Try direct load first
#     try:
#         model.load_state_dict(state_dict)
#     except RuntimeError:
#         # Strip prefix and try again
#         model.load_state_dict(strip_module_prefix(state_dict))
#
#     if is_main_process():
#         print('load optimizer state')
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     return checkpoint['epoch'] + 1, checkpoint['index']

def load_state_dict_model(model, optimizer, scheduler, checkpoint) -> Tuple[int, int]:
    """
    Load both model & optimizer state. Returns (next_epoch, index).
    Handles possible 'module.' prefixes automatically.
    """
    if is_main_process():
        print('load model state')
    
    state_dict = checkpoint['model_state_dict']
    
    # Check if the current model is a DDP model
    is_ddp_model = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    
    # Check if the checkpoint was saved from a DDP model
    is_ddp_checkpoint = all(k.startswith('module.') for k in state_dict.keys())

    if is_ddp_model and not is_ddp_checkpoint:
        # If loading a non-DDP checkpoint into a DDP model, add the "module." prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif not is_ddp_model and is_ddp_checkpoint:
        # If loading a DDP checkpoint into a non-DDP model, strip the "module." prefix
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    
    # Now load the state dict
    model.load_state_dict(state_dict)

    if is_main_process():
        print('load optimizer state')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        if is_main_process():
            print('load scheduler state')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'] + 1, checkpoint['index']

def load_state_dict_model_only(model, checkpoint) -> Tuple[int, int]:
    """
    Load only the model weights. Returns (next_epoch, index).
    """
    if is_main_process():
        print('load model state')
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(strip_module_prefix(state_dict))

    return checkpoint['epoch'] + 1, checkpoint['index']

# def load_state_dict_model_only(model, checkpoint):
#     print('load model state')
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     return checkpoint['epoch'] + 1, checkpoint['index']


# def save_state_dict_model(model, optimizer, epoch, index, cfg):
#     # save checkpoint
#     n_epoch = epoch + 1
#     if (n_epoch) % cfg.snapshot_interval == 0:
#         dir_chk = os.path.join(cfg.output, 'checkpoints')
#         os.makedirs(dir_chk, exist_ok=True)
#         path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(n_epoch))
#
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'index': index,
#         }
#
#         torch.save(checkpoint, path)

def save_state_dict_model(model, optimizer, scheduler, epoch, index, cfg):
    """
    Save checkpoint on rank0 only.
    """
    n_epoch = epoch + 1
    if n_epoch % cfg.snapshot_interval != 0:
        return

    # Optionally sync to be safe when others continue after save
    # ---------- Sync BEFORE saving ----------
    if is_dist_avail_and_initialized():
        dist.barrier(device_ids=[cfg.local_rank])

    if is_main_process():
        dir_chk = os.path.join(cfg.output, 'checkpoints')
        os.makedirs(dir_chk, exist_ok=True)
        path = os.path.join(dir_chk, f'model-{n_epoch:02d}.pt')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'index': index,
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, path)

    # ---------- Sync AFTER saving ----------
    if is_dist_avail_and_initialized():
        dist.barrier(device_ids=[cfg.local_rank])
