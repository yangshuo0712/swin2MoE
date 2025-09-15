from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

def _unwrap(m):
    return m.module if hasattr(m, "module") else m

def build_optimizer(model, cfg):
    o = cfg
    m = _unwrap(model)

    base_lr = float(o.optim.learning_rate)
    betas   = tuple(o.optim.model_betas)
    eps     = float(o.optim.model_eps)
    wd      = float(o.optim.model_weight_decay)

    corr_params = []
    if hasattr(m, "band_logit_bias") and m.band_logit_bias is not None:
        corr_params.append(m.band_logit_bias)
    if hasattr(m, "band_head") and m.band_head is not None:
        corr_params += list(m.band_head.parameters())

    corr_ids = {id(p) for p in corr_params}
    base_params = [p for p in m.parameters() if id(p) not in corr_ids]

    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": base_lr, "weight_decay": wd, "name": "base"},
            {"params": corr_params, "lr": 0.0, "weight_decay": 0.0, "name": "band_correction"},
        ],
        lr=base_lr, betas=betas, eps=eps, weight_decay=wd,
    )

    scheduler = None
    if "scheduler" in o.optim:
        sch = o.optim.scheduler
        warm = int(getattr(sch, "warmup_epochs", 0))
        if warm > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm)
            cosine = CosineAnnealingLR(optimizer, T_max=sch.T_max - warm, eta_min=sch.eta_min)
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warm])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=sch.T_max, eta_min=sch.eta_min)

    return optimizer, scheduler
