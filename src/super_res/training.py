import datetime
import torch
import torch.distributed as dist

import debug

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from validation import validate, do_save_metrics as save_metrics
from chk_loader import load_checkpoint, load_state_dict_model, \
        save_state_dict_model
from validation import build_eval_metrics
from losses import build_losses
from optim import build_optimizer
from .model import build_model

from utils import is_dist_avail_and_initialized, is_main_process

def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    rt /= dist.get_world_size()
    return rt
# -----------

def train(train_dloader, val_dloader, cfg):

    # Tensorboard
    # writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # only main process
    writer = None
    if is_main_process():
         writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(
             datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # eval every x
    eval_every = cfg.metrics.get('eval_every', 1)

    model = build_model(cfg)
    device = cfg.device
    model.to(device=device)
    
    # --- DDP ---
    if getattr(cfg, "distributed", False):
        model.to(cfg.device)
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
                broadcast_buffers=False,
                )
    # -----------

    losses = build_losses(cfg)
    optimizer = build_optimizer(model, cfg)

    # --- AMP ----
    scaler = torch.amp.GradScaler('cuda', enabled=getattr(cfg, 'AMP', False))
    # ------------

    begin_epoch = 0
    index = 0
    try:
        checkpoint = load_checkpoint(cfg)
        begin_epoch, index = load_state_dict_model(
            model, optimizer, checkpoint)
    except FileNotFoundError:
        print('no checkpoint found')

    # print('build eval metrics')
    # metrics = build_eval_metrics(cfg)

    if is_main_process():
        print('main_proc: build eval metrics')
        metrics = build_eval_metrics(cfg)

    for e in range(begin_epoch, cfg.epochs):

        # --- DDP ---
        if cfg.distributed and isinstance(train_dloader.sampler,
                                          torch.utils.data.DistributedSampler):
            train_dloader.sampler.set_epoch(e)
        # -----------
        model.train()
        index = train_epoch(
            model,
            train_dloader,
            losses,
            optimizer,
            e,
            writer,
            index,
            cfg,
            scaler)

    if (e + 1) % eval_every == 0:
        dist.barrier()
        result = validate(model, val_dloader, metrics, e,
                          writer if is_main_process() else None,
                          'test', cfg)
        if is_main_process():
            cfg.epoch = e + 1
            save_metrics(result, cfg)
        dist.barrier()

        # save_state_dict_model(model, optimizer, e, index, cfg)
        if is_main_process():
            save_state_dict_model(model, optimizer, e, index, cfg)

def train_epoch(model, train_dloader, losses, optimizer, epoch, writer,
                index, cfg, scaler=None):
    weights = cfg.losses.weights
    for index, batch in tqdm(
            enumerate(train_dloader, index), total=len(train_dloader),
            desc='Epoch: %d / %d' % (epoch + 1, cfg.epochs)):

        # Transfer in-memory data to CUDA devices to speed up training
        hr = batch["hr"].to(device=cfg.device, non_blocking=True)
        lr = batch["lr"].to(device=cfg.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=getattr(cfg, 'AMP', False)):

            sr = model(lr)

            loss_tracker = {}

            loss_moe = None
            if not torch.is_tensor(sr):
                sr, loss_moe = sr
                if torch.is_tensor(loss_moe):
                    loss_tracker['loss_moe'] = loss_moe * weights.moe

            sr = sr.contiguous()

            if 'pixel_criterion' in losses:
                loss_tracker['pixel_loss'] = weights.pixel * \
                    losses['pixel_criterion'](sr, hr)

            # cc loss
            if 'cc_criterion' in losses:
                loss_tracker['cc_loss'] = weights.cc * \
                    losses['cc_criterion'](sr, hr)

            # ssim loss
            if 'ssim_criterion' in losses:
                loss_tracker['ssim_loss'] = weights.ssim * \
                    losses['ssim_criterion'](sr, hr)

            loss_values: list[Tensor] = list(loss_tracker.values())            # List[Tensor]
            local_loss: Tensor  = torch.stack(loss_values).sum()          # Tensor
                
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(local_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            local_loss.backward()
            optimizer.step()

        # debug.log_hr_stats(lr, sr, hr, writer, index, cfg)
        # debug.log_losses(loss_tracker, 'train', writer, index)

        with torch.no_grad():
            global_loss = reduce_tensor(local_loss.detach())

        if writer is not None:
            debug.log_hr_stats(lr, sr, hr, writer, index, cfg)
            debug.log_losses({'train_loss': global_loss}, 'train', writer, index)
    return index
