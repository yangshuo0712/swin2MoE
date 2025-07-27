import os
import torch

import torch.distributed as dist
from tqdm import tqdm
from torch.nn import Upsample
from collections import OrderedDict

from utils import AverageMeter, load_fun, sync_average_meters, is_main_process
from metrics import CC, SAM, ERGAS, piq_psnr, piq_ssim, \
    piq_rmse
from chk_loader import load_checkpoint

def ddp_reduce_avg_metrics(avg_metrics, device):
    world = dist.get_world_size()
    for m in avg_metrics.values():
            t = torch.tensor([m.sum, m.count], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            m.sum = t[0].item()
            m.count = t[1].item()
            m.avg = m.sum / max(m.count, 1)

def validate(g_model, val_dloader, metrics, epoch, writer, mode, cfg):
    # Put the adversarial network model in validation mode
    g_model.eval()

    avg_metrics = build_avg_metrics()

    use_minmax = cfg.dataset.get('stats', {}).get('use_minmax', False)
    dset = cfg.dataset
    denorm = load_fun(dset.get('denorm'))(
            cfg,
            hr_name=cfg.dataset.hr_name,
            lr_name=cfg.dataset.lr_name)
    evaluable = load_fun(dset.get('printable'))(
            cfg,
            hr_name=cfg.dataset.hr_name,
            lr_name=cfg.dataset.lr_name,
            filter_outliers=False,
            use_minmax=use_minmax)

    amp_enabled = getattr(cfg, 'AMP', False)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=amp_enabled):
        for j, batch in tqdm(
                enumerate(val_dloader), total=len(val_dloader),
                desc='Val Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                disable=not is_main_process()):
            hr = batch["hr"].to(device=cfg.device, non_blocking=True)
            lr = batch["lr"].to(device=cfg.device, non_blocking=True)

            sr = g_model(lr)
            if not torch.is_tensor(sr):
                sr, _ = sr
            sr = sr.contiguous()

            # denormalize to original values
            hr, sr, lr = denorm(hr, sr, lr)
            # normalize [0, 1]
            hr, sr, lr = evaluable(hr, sr, lr)

            for k, fun in metrics.items():
                for i in range(len(sr)):
                    val = fun(sr[i][None], hr[i][None])
                    avg_metrics[k].update(val, n=1)

    # --- sync all metrics ---
    sync_average_meters(avg_metrics, cfg.device)
    if writer is not None and is_main_process():
             for k, v in avg_metrics.items():
                 val = v.avg.item() if isinstance(v.avg, torch.Tensor) else float(v.avg)
                 # writer.add_scalar("{}/{}".format(mode, k), v.avg.item(), epoch+1)
                 writer.add_scalar(f"{mode}/{k}", val, epoch+1)

    if cfg.get('eval_return_to_train', True):
        g_model.train()

    return avg_metrics
    # ------------------------

def build_eval_metrics(cfg):
    # Create an IQA evaluation model
    metrics = {
        'psnr_model': piq_psnr(cfg),
        'ssim_model': piq_ssim(cfg),
        'cc_model': CC(),
        'rmse_model': piq_rmse(cfg),
        'sam_model': SAM(),
        'ergas_model': ERGAS(),
    }

    for k in metrics.keys():
        metrics[k] = metrics[k].to(cfg.device)

    return metrics


def build_avg_metrics():
    return OrderedDict([
        ('psnr_model', AverageMeter("PIQ_PSNR", ":4.4f")),
        ('ssim_model', AverageMeter("PIQ_SSIM", ":4.4f")),
        ('cc_model', AverageMeter("CC", ":4.4f")),
        ('rmse_model', AverageMeter("PIQ_RMSE", ":4.4f")),
        ('sam_model', AverageMeter("SAM", ":4.4f")),
        ('ergas_model', AverageMeter("ERGAS", ":4.4f")),
    ])


def main(val_dloader, cfg, save_metrics=True):
    model = load_eval_method(cfg)
    if is_main_process():
        print('build eval metrics')
    metrics = build_eval_metrics(cfg)
    result = validate(
        model, val_dloader, metrics, cfg.epoch, None, 'test', cfg)
    if save_metrics and is_main_process():
        do_save_metrics(result, cfg)
    return result


def get_result_filename(cfg):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pt'.format(cfg.epoch))


def do_save_metrics(metrics, cfg):
    filename = get_result_filename(cfg)
    print('save results {}'.format(filename))
    torch.save({
        'epoch': cfg.epoch,
        'metrics': OrderedDict([
            (k, v.avg) for k, v in metrics.items()
        ])
    }, filename)


def load_metrics(cfg):
    filename = get_result_filename(cfg)
    print('load results {}'.format(filename))
    result = torch.load(filename)
    # check if epoch corresponds
    assert result['epoch'] == cfg.epoch
    # build AVG objects
    avg_metrics = build_avg_metrics()
    for k, v in result['metrics'].items():
        avg_metrics[k].avg = v
    return avg_metrics


def print_metrics(metrics):
    names = []
    values = []
    for i, v in enumerate(metrics.values()):
        try:
            names.append(v.name)
            values.append(v)
        except AttributeError:
            # skip for retrocompatibility
            pass
    print(*names)
    print(*values)


def load_eval_method(cfg):
    if cfg.eval_method is None:
        vis = cfg.visualize
        model = load_fun(vis.get('model'))(cfg)
        # Load model state dict
        try:
            checkpoint = load_checkpoint(cfg)
            _, _ = load_fun(vis.get('checkpoint'))(model, checkpoint)
        except Exception as e:
            print(e)
            exit(0)

        return model

    print('load non-dl upsampler: {}'.format(cfg.eval_method))
    return NonDLEvalMethod(cfg)


class NonDLEvalMethod(object):
    def __init__(self, cfg):
        self.upscale_factor = cfg.metrics.upscale_factor
        self.upsampler = Upsample(
            scale_factor=self.upscale_factor,
            mode=cfg.eval_method)

    def __call__(self, x):
        return self.upsampler(x)

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        return self
