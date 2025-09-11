import argparse
import torch
import os
import torch.distributed as dist
from thop import profile

from config import parse_config
from utils import is_main_process, load_fun, set_deterministic
from visualize import main as vis_main
from validation import main as val_main, print_metrics as val_print_metrics, \
        load_metrics
from debug import measure_avg_time
from super_res.model import build_model
from utils import calculate_apc_spc
from probe.run_probe import run_probe_pipeline
# import pdb

def parse_configs():
    parser = argparse.ArgumentParser(description='SuperRes model')

    # --- DDP ----
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank passed by distributed launcher')
    parser.add_argument('--distributed', type=bool, default=False,
                        help='denote whether to use DDP')
    # ------------

    # --- AMP ----
    parser.add_argument('--AMP', type=bool, default=False,
                        help='denote whether to use AMP')
    # ------------
    parser.add_argument('--debug_iters', type=int, default=None,
                        help='early exit when debuging')

    #-------------

    parser.add_argument('--use_accum',
                        type=bool,
                        default=True,
                        help='denote whether to use accum step')
    # For training and testing
    parser.add_argument('--config',
                        default="cfg_n/sen2venus_exp4_2x_v5.yml",
                        help='Configuration file.')
    parser.add_argument('--phase',
                        default='train',
                        choices=['train', 'test', 'mean_std', 'vis',
                                 'plot_data', 'avg_time', 'flops', 'apc', 'probe'],
                        help='Training or testing or play phase.')
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        metavar='B',
                        help='Batch size. If defined, overwrite cfg file.')
    help_num_workers = 'The number of workers to load dataset. Default: 0'
    parser.add_argument('--num_workers',
                        type=int,
                        default=32,
                        metavar='N',
                        help=help_num_workers)
    parser.add_argument('--output',
                        default='./output/sen2venus_v26_8',
                        help='Directory where save the output.')
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        metavar='N',
                        help='number of epoches (default: 50)')
    help_snapshot = 'The epoch interval of model snapshot (default: 10)'
    parser.add_argument('--snapshot_interval',
                        type=int,
                        default=1,
                        metavar='N',
                        help=help_snapshot)
    parser.add_argument("--num_images",
                        type=int,
                        default=10,
                        help="Number of images to plot")
    parser.add_argument('--eval_method',
                        default=None, type=str,
                        help='Non-DL method to use on evaluation.')
    parser.add_argument('--repeat_times',
                        type=int,
                        default=1000,
                        help='Measure times repeating model call')
    help_warm = 'Warm model calling it before starting the measure'
    parser.add_argument('--warm_times',
                        type=int,
                        default=10,
                        help=help_warm)
    parser.add_argument('--dpi',
                        type=int,
                        default=2400,
                        help="dpi in png output file.")

    args = parser.parse_args()
    return parse_config(args)


def init_distributed(cfg):
    """
    Initialize distributed environment.

    Logic:
    - If WORLD_SIZE env var > 1, treat as distributed (torchrun / launcher).
    - Else fall back to cfg.distributed flag (for explicit CLI).
    - Set CUDA device using local_rank BEFORE calling init_process_group.
    - Call dist.init_process_group with env:// and then barrier.
    """
    # Prefer environment variables set by torch.distributed launcher
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    rank_env = int(os.environ.get("RANK", 0))
    local_rank_env = int(os.environ.get("LOCAL_RANK", getattr(cfg, "local_rank", 0)))

    use_distributed = world_size_env > 1 or getattr(cfg, "distributed", False)

    if not use_distributed:
        # Single-process (or user explicitly disabled distributed)
        cfg.distributed = False
        cfg.rank = 0
        cfg.world_size = 1
        cfg.local_rank = 0
        cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    # Otherwise initialize distributed process group using env://
    cfg.distributed = True
    cfg.rank = rank_env
    cfg.world_size = world_size_env
    cfg.local_rank = local_rank_env

    # Set CUDA visible device for this process BEFORE init_process_group
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(cfg.local_rank)
            cfg.device = torch.device(f"cuda:{cfg.local_rank}")
        except Exception as e:
            # fallback
            cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Warning setting cuda device by local_rank failed: {e}")
    else:
        cfg.device = torch.device("cpu")

    # Initialize process group. Using env:// so rank/world_size come from env vars.
    # Note: some PyTorch versions accept device_id kwarg; we rely on setting device before init.
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=cfg.rank,
        world_size=cfg.world_size,
    )

    # synchronize
    dist.barrier()


def cleanup_distributed(cfg):
    if getattr(cfg, "distributed", False) and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            # ignore barrier errors during shutdown
            pass
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning destroying process group: {e}")


def main(cfg):
    try:
        init_distributed(cfg)
        load_dataset_fun = load_fun(cfg.dataset.get(
            'load_dataset', 'datasets.sen2venus.load_dataset'))

        # Training or evaluation phases
        if cfg.phase == 'avg_time':
            print(measure_avg_time(cfg))
        elif cfg.phase == 'train':
            train_fun = load_fun(cfg.get('train', 'srgan.training.train'))
            train_dloader, val_dloader, _ = load_dataset_fun(cfg)
            train_fun(train_dloader, val_dloader, cfg)
        elif cfg.phase == 'mean_std':
            if 'stats' in cfg.dataset.keys():
                cfg.dataset.pop('stats')
            _, _, concat_dloader = load_dataset_fun(cfg, concat_datasets=True)
            fun = load_fun(cfg.get(cfg.phase))
            fun(concat_dloader, cfg)
        elif cfg.phase == 'plot_data':
            _, _, concat_dloader = load_dataset_fun(cfg, concat_datasets=True)
            fun = load_fun(cfg.get(cfg.phase))
            fun(concat_dloader, cfg)
        elif cfg.phase == 'vis':
            # cfg may be dict-like or object. handle both safely.
            try:
                cfg.batch_size = 1
            except Exception:
                try:
                    cfg['batch_size'] = 1
                except Exception:
                    pass
            vis_main(cfg)
        elif cfg.phase == 'test':
            try:
                if cfg.eval_method is not None:
                    raise FileNotFoundError()
                metrics = load_metrics(cfg)
            except FileNotFoundError:
                _, val_dloader, _ = load_dataset_fun(cfg, only_test=True)
                metrics = val_main(
                    val_dloader, cfg, save_metrics=cfg.eval_method is None)
            if is_main_process():
                    val_print_metrics(metrics)
        elif cfg.phase == 'apc':
            calculate_apc_spc(cfg)
        elif cfg.phase == 'flops':
            model = build_model(cfg)
            input_shape = cfg.visualize.get('input_shape', [4, 128, 128])
            # Create a dummy input tensor with the correct shape
            dummy_input = torch.randn(1, *input_shape).to(cfg.device)
            # Calculate FLOPs and Params
            macs, params = profile(model, inputs=(dummy_input, ))
            # FLOPs is approximately 2 * MACs
            flops = macs * 2
            print(f"Input shape: {dummy_input.shape}")
            print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
            print(f"Parameters: {params / 1e6:.2f} M")
        elif cfg.phase == 'probe':
            run_probe_pipeline(cfg)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even in case of error
        try:
            cleanup_distributed(cfg)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    # parse input arguments
    cfg = parse_configs()
    # fix random seed
    set_deterministic(cfg.seed)
    # run main
    main(cfg)
