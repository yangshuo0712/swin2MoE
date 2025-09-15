import datetime
import torch
import torch.distributed as dist
import torch.nn.functional as F

import debug

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
from torch import Tensor
from torch.profiler import profile, ProfilerActivity

from validation import validate, do_save_metrics as save_metrics
from chk_loader import load_checkpoint, load_state_dict_model, save_state_dict_model
from validation import build_eval_metrics
from losses import build_losses
from optim import build_optimizer
from hooks import MoEHook
from utils import is_dist_avail_and_initialized, is_main_process


def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    rt /= dist.get_world_size()
    return rt


def unwrap_model(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def train(train_dloader, val_dloader, cfg):
    # only main process writes TB
    writer = None
    if is_main_process():
        writer = SummaryWriter(cfg.output + "/tensorboard/train_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )

    eval_every = cfg.metrics.get('eval_every', 1)

    from .model import build_model
    model = build_model(cfg)

    if getattr(cfg, "debug_iters", None) is None:
        model = torch.compile(model)

    device = cfg.device
    model.to(device=device)

    # DDP
    if getattr(cfg, "distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    losses = build_losses(cfg)
    optimizer, scheduler = build_optimizer(model, cfg)

    # AMP
    scaler = torch.amp.GradScaler('cuda', enabled=getattr(cfg, 'AMP', False))

    begin_epoch = 0
    index = 0
    try:
        checkpoint = load_checkpoint(cfg)
        begin_epoch, index = load_state_dict_model(model, optimizer, scheduler, checkpoint)
    except FileNotFoundError:
        print('no checkpoint found')

    if is_main_process():
        print('build eval metrics')
    metrics = build_eval_metrics(cfg)

    moe_hook = MoEHook(unwrap_model(model))
    moe_hook.attach()

    for e in range(begin_epoch, cfg.epochs):
        # DDP sampler epoch
        if cfg.distributed and isinstance(train_dloader.sampler, torch.utils.data.DistributedSampler):
            train_dloader.sampler.set_epoch(e)

        model.train()

        use_accum = getattr(cfg, 'use_accum', False)

        if use_accum:
            index = acc_train_epoch(model, train_dloader, losses, optimizer, scheduler, e, writer, index, cfg, scaler, moe_hook)
        else:
            index = train_epoch(model, train_dloader, losses, optimizer, scheduler, e, writer, index, cfg, scaler, moe_hook)

        if scheduler:
            scheduler.step()

        if (e + 1) % eval_every == 0:
            if cfg.distributed:
                dist.barrier(device_ids=[cfg.local_rank])
            result = validate(model, val_dloader, metrics, e,
                              writer if is_main_process() else None,
                              'test', cfg)
            if is_main_process():
                cfg.epoch = e + 1
                save_metrics(result, cfg)
            if cfg.distributed:
                dist.barrier(device_ids=[cfg.local_rank])

            save_state_dict_model(model, optimizer, scheduler, e, index, cfg)


def train_epoch(model, train_dloader, losses, optimizer, scheduler, epoch, writer, index, cfg, scaler=None, moe_hook=None):
    debug_iters = getattr(cfg, "debug_iters", None)
    weights = cfg.losses.weights

    warm_e = getattr(cfg, 'train', {}).get('band_correction_warmup_epochs', 20)
    m = unwrap_model(model)
    if (not getattr(m, 'enable_band_correction', False)) and (epoch >= warm_e):
        m.enable_band_correction = True
        band_lr = float(getattr(cfg.optim, "band_corr_lr", cfg.optim.learning_rate * 0.1))
        band_wd = float(getattr(cfg.optim, "band_corr_weight_decay", 1e-4))
        for pg in optimizer.param_groups:
            if pg.get("name", "") == "band_correction":
                pg["lr"] = band_lr
                pg["weight_decay"] = band_wd
        if is_main_process():
            print(f"[Stage B] Enable band correction at epoch {epoch}: lr={band_lr}, wd={band_wd}")

    def _get_w(name, default=0.0):
        try:
            return float(getattr(weights, name))
        except Exception:
            try:
                return float(weights.get(name, default))
            except Exception:
                return float(default)

    w_moe = _get_w('moe', 0.0)
    w_kl = _get_w('kl', 0.0)
    w_ent = _get_w('entropy', 0.0)

    def run_one_iter(batch, index):
        hr = batch["hr"].to(device=cfg.device, non_blocking=True)
        lr = batch["lr"].to(device=cfg.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=getattr(cfg, 'AMP', False)):
            out = model(lr)
            loss_tracker = {}

            # MoE loss
            loss_moe = None
            if not torch.is_tensor(out):
                sr, loss_moe = out
                if torch.is_tensor(loss_moe) and w_moe > 0:
                    loss_tracker['loss_moe'] = w_moe * loss_moe
            else:
                sr = out

            sr = sr.contiguous()

            if 'pixel_criterion' in losses:
                loss_tracker['pixel_loss'] = _get_w('pixel', 0.0) * losses['pixel_criterion'](sr, hr)
            if 'cc_criterion' in losses:
                loss_tracker['cc_loss'] = _get_w('cc', 0.0) * losses['cc_criterion'](sr, hr)
            if 'ssim_criterion' in losses:
                loss_tracker['ssim_loss'] = _get_w('ssim', 0.0) * losses['ssim_criterion'](sr, hr)

            if getattr(m, 'enable_band_correction', False) and (w_kl > 0 or w_ent > 0):
                x = lr
                x = m.check_image_size(x)
                x_pre = (x - m.mean.type_as(x)) * m.img_range

                orig_flag = m.enable_band_correction
                m.enable_band_correction = True
                w_cur = m._compute_band_weights(x_pre)  # [B*L, C] with grad

                with torch.no_grad():
                    m.enable_band_correction = False
                    w_pri = m._compute_band_weights(x_pre)

                m.enable_band_correction = orig_flag

                w_cur = w_cur.clamp_min(1e-8)
                w_pri = w_pri.clamp_min(1e-8)

                # KL(stopgrad(prior) || current)  → F.kl_div input=log q, target=p
                kl_loss = F.kl_div(w_cur.log(), w_pri, reduction="batchmean")
                ent_loss = -(w_cur * w_cur.log()).sum(dim=1).mean()

                if w_kl > 0:
                    loss_tracker['kl_loss'] = w_kl * kl_loss
                if w_ent > 0:
                    loss_tracker['entropy_loss'] = w_ent * ent_loss

            loss_values: list[Tensor] = list(loss_tracker.values())
            local_loss: Tensor = torch.stack(loss_values).sum() if len(loss_values) else torch.tensor(0.0, device=sr.device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(local_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            local_loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            global_loss = reduce_tensor(local_loss.detach())

        if writer is not None:
            for pg in optimizer.param_groups:
                name = pg.get("name", "pg")
                writer.add_scalar(f"lr/{name}", pg["lr"], index)

        if getattr(m, 'enable_band_correction', False) and (w_kl > 0 or w_ent > 0) and writer is not None:
            with torch.no_grad():
                w_dot = (w_cur * w_pri).sum(dim=1).mean()                 # [0,1]
                hard_agree = (w_cur.argmax(dim=1) == w_pri.argmax(dim=1)).float().mean()
                ent_cur = (-(w_cur * w_cur.log()).sum(dim=1)).mean()
                ent_pri = (-(w_pri * w_pri.log()).sum(dim=1)).mean()

                writer.add_scalar("train/w_dot_prior", w_dot.item(), index)
                writer.add_scalar("train/w_hard_agree", hard_agree.item(), index)
                writer.add_scalar("train/entropy_cur", ent_cur.item(), index)
                writer.add_scalar("train/entropy_prior", ent_pri.item(), index)

                top1_cur = w_cur.argmax(dim=1)
                C = w_cur.size(1)
                for b in range(C):
                    share = (top1_cur == b).float().mean().item()
                    writer.add_scalar(f"train/w_top1_share/b{b}", share, index)

            try:
                ps_h, ps_w = m.patch_embed.patch_size
                Hp, Wp = (lr.shape[-2] + (ps_h - lr.shape[-2] % ps_h) % ps_h) // ps_h, \
                         (lr.shape[-1] + (ps_w - lr.shape[-1] % ps_w) % ps_w) // ps_w
                L = Hp * Wp
                C = w_cur.size(1)
                w_cur_map = w_cur[:L, :].T.reshape(C, 1, Hp, Wp)  # [C,1,Hp,Wp]
                w_pri_map = w_pri[:L, :].T.reshape(C, 1, Hp, Wp)
                w_cur_vis = (w_cur_map / (w_cur_map.max().clamp_min(1e-8))).clamp(0, 1)
                w_pri_vis = (w_pri_map / (w_pri_map.max().clamp_min(1e-8))).clamp(0, 1)
                writer.add_images("img/w_map_cur", w_cur_vis, index)
                writer.add_images("img/w_map_prior", w_pri_vis, index)
            except Exception:
                pass

        if writer is not None and getattr(m, 'enable_band_correction', False) and hasattr(m, "band_head") and m.band_head is not None:
            try:
                g2 = 0.0
                cnt = 0
                for p in m.band_head.parameters():
                    if p.grad is not None:
                        g2 += (p.grad.detach().norm(2).item() ** 2)
                        cnt += 1
                if cnt > 0:
                    writer.add_scalar("grad_norm/band_corr", (g2 ** 0.5), index)
            except Exception:
                pass

        if writer is not None and moe_hook is not None:
            try:
                stats = moe_hook.get_stats()
                for k, v in stats.items():
                    writer.add_scalar(f"moe/{k}", float(v), index)
                moe_hook.stats.clear() 
            except Exception:
                pass

        if writer is not None:
            debug.log_hr_stats(lr, sr, hr, writer, index, cfg)
            debug.log_losses({'train_loss': global_loss}, 'train', writer, index)
            for loss_name, loss_value in loss_tracker.items():
                if torch.is_tensor(loss_value):
                    writer.add_scalar(f"train/{loss_name}", loss_value.item(), index)

    # ----------- normal training ------------
    if debug_iters is None:
        for index, batch in tqdm(
                enumerate(train_dloader, index), total=len(train_dloader),
                desc='Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                disable=not is_main_process()):
            run_one_iter(batch, index)
        return index

    # ----------- debug + profiler ------------
    else:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
                     record_shapes=True) as prof:
            for i, (idx, batch) in enumerate(
                    tqdm(enumerate(train_dloader, index),
                         total=debug_iters,
                         desc='[DEBUG] Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                         disable=not is_main_process())):
                if i >= debug_iters:
                    break
                run_one_iter(batch, idx)

        print("\n" + "="*30)
        print("    PROFILER ANALYSIS RESULT    ")
        print("="*30)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print("="*30 + "\n")

        return index


def acc_train_epoch(model, train_dloader, losses, optimizer, schedule, epoch, writer, index, cfg, scaler=None, moe_hook=None):
    debug_iters = getattr(cfg, "debug_iters", None)
    accumulation_steps = getattr(cfg, 'accumulation_steps', 4)
    weights = cfg.losses.weights

    warm_e = getattr(cfg, 'train', {}).get('band_correction_warmup_epochs', 20)
    m = unwrap_model(model)
    if (not getattr(m, 'enable_band_correction', False)) and (epoch >= warm_e):
        m.enable_band_correction = True
        band_lr = float(getattr(cfg.optim, "band_corr_lr", cfg.optim.learning_rate * 0.1))
        band_wd = float(getattr(cfg.optim, "band_corr_weight_decay", 1e-4))
        for pg in optimizer.param_groups:
            if pg.get("name", "") == "band_correction":
                pg["lr"] = band_lr
                pg["weight_decay"] = band_wd
        if is_main_process():
            print(f"[Stage B] Enable band correction at epoch {epoch}: lr={band_lr}, wd={band_wd}")

    def _get_w(name, default=0.0):
        try:
            return float(getattr(weights, name))
        except Exception:
            try:
                return float(weights.get(name, default))
            except Exception:
                return float(default)

    w_moe = _get_w('moe', 0.0)
    w_kl = _get_w('kl', 0.0)
    w_ent = _get_w('entropy', 0.0)

    is_ddp = getattr(cfg, "distributed", False) and isinstance(
        model, torch.nn.parallel.DistributedDataParallel)

    optimizer.zero_grad(set_to_none=True)
    accum_counter = 0

    def run_one_iter(batch, idx):
        nonlocal accum_counter

        hr = batch["hr"].to(device=cfg.device, non_blocking=True)
        lr = batch["lr"].to(device=cfg.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=getattr(cfg, 'AMP', False)):
            out = model(lr)
            loss_tracker = {}

            loss_moe = None
            if not torch.is_tensor(out):
                sr, loss_moe = out
                if torch.is_tensor(loss_moe) and w_moe > 0:
                    loss_tracker['loss_moe'] = w_moe * loss_moe
            else:
                sr = out

            sr = sr.contiguous()

            if 'pixel_criterion' in losses:
                loss_tracker['pixel_loss'] = _get_w('pixel', 0.0) * losses['pixel_criterion'](sr, hr)
            if 'cc_criterion' in losses:
                loss_tracker['cc_loss'] = _get_w('cc', 0.0) * losses['cc_criterion'](sr, hr)
            if 'ssim_criterion' in losses:
                loss_tracker['ssim_loss'] = _get_w('ssim', 0.0) * losses['ssim_criterion'](sr, hr)

            if getattr(m, 'enable_band_correction', False) and (w_kl > 0 or w_ent > 0):
                x = lr
                x = m.check_image_size(x)
                x_pre = (x - m.mean.type_as(x)) * m.img_range

                orig_flag = m.enable_band_correction
                m.enable_band_correction = True
                w_cur = m._compute_band_weights(x_pre)   # with grad
                with torch.no_grad():
                    m.enable_band_correction = False
                    w_pri = m._compute_band_weights(x_pre)
                m.enable_band_correction = orig_flag

                w_cur = w_cur.clamp_min(1e-8)
                w_pri = w_pri.clamp_min(1e-8)

                kl_loss = F.kl_div(w_cur.log(), w_pri, reduction="batchmean")
                ent_loss = -(w_cur * w_cur.log()).sum(dim=1).mean()

                if w_kl > 0:
                    loss_tracker['kl_loss'] = w_kl * kl_loss
                if w_ent > 0:
                    loss_tracker['entropy_loss'] = w_ent * ent_loss

            loss_values: list[Tensor] = list(loss_tracker.values())
            local_loss: Tensor = torch.stack(loss_values).sum() if len(loss_values) else torch.tensor(0.0, device=sr.device)

        accum_counter += 1
        is_update_step = (accum_counter % accumulation_steps == 0)

        sync_ctx = nullcontext()
        if is_ddp and not is_update_step:
            sync_ctx = model.no_sync()

        with sync_ctx:
            if scaler is not None:
                scaler.scale(local_loss / accumulation_steps).backward()
            else:
                (local_loss / accumulation_steps).backward()

        if is_update_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                global_loss = reduce_tensor(local_loss.detach())

            if writer is not None:
                for pg in optimizer.param_groups:
                    name = pg.get("name", "pg")
                    writer.add_scalar(f"lr/{name}", pg["lr"], idx)

            if writer is not None and getattr(m, 'enable_band_correction', False) and hasattr(m, "band_head") and m.band_head is not None:
                try:
                    g2 = 0.0
                    cnt = 0
                    for p in m.band_head.parameters():
                        if p.grad is not None:
                            g2 += (p.grad.detach().norm(2).item() ** 2)
                            cnt += 1
                    if cnt > 0:
                        writer.add_scalar("grad_norm/band_corr", (g2 ** 0.5), idx)
                except Exception:
                    pass

            if writer is not None and moe_hook is not None:
                try:
                    stats = moe_hook.get_stats()
                    for k, v in stats.items():
                        writer.add_scalar(f"moe/{k}", float(v), idx)
                    moe_hook.stats.clear()
                except Exception:
                    pass

            if writer is not None:
                debug.log_hr_stats(lr, sr, hr, writer, idx, cfg)
                debug.log_losses({'train_loss': global_loss}, 'train', writer, idx)
                for loss_name, loss_value in loss_tracker.items():
                    if torch.is_tensor(loss_value):
                        writer.add_scalar(f"train/{loss_name}", loss_value.item(), idx)

    # ----------- normal training ------------
    if debug_iters is None:
        for index, batch in tqdm(
                enumerate(train_dloader, index), total=len(train_dloader),
                desc='Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                disable=not is_main_process()):
            run_one_iter(batch, index)

        if (accum_counter % accumulation_steps) != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return index

    # ----------- debug + profiler ------------
    else:
        if is_main_process():
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
                           record_shapes=True)
            with prof:
                for i, (idx, batch) in enumerate(
                        tqdm(enumerate(train_dloader, index),
                             total=debug_iters,
                             desc='[DEBUG] Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                             disable=not is_main_process())):
                    if i >= debug_iters:
                        break
                    run_one_iter(batch, idx)
                    prof.step()

            print("\n" + "="*30)
            print("    PROFILER ANALYSIS RESULT    ")
            print("="*30)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
            print("="*30 + "\n")
        else:
            for index, batch in tqdm(
                    enumerate(train_dloader, index), total=len(train_dloader),
                    desc='Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                    disable=not is_main_process()):
                run_one_iter(batch, index)

        return index
