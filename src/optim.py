from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

def build_optimizer(model, cfg):
    o_cfg = cfg

    optimizer = optim.Adam(model.parameters(),
                           o_cfg.optim.learning_rate,
                           o_cfg.optim.model_betas,
                           o_cfg.optim.model_eps,
                           o_cfg.optim.model_weight_decay)

    scheduler = None
    if 'scheduler' in o_cfg.optim:
        scheduler_cfg = o_cfg.optim.scheduler
        if scheduler_cfg.type == 'CosineAnnealingLR':
            warmup_epochs = getattr(scheduler_cfg, "warmup_epochs", 0)

            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_cfg.T_max - warmup_epochs,
                    eta_min=scheduler_cfg.eta_min
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_cfg.T_max,
                    eta_min=scheduler_cfg.eta_min
                )

    return optimizer, scheduler
