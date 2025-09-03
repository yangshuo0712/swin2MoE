from utils import is_main_process

def build_model(cfg):
    """
    Builds the super-resolution model based on the configuration file.

    Args:
        cfg (EasyDict): Configuration object containing model parameters.

    Returns:
        torch.nn.Module: The instantiated model.
    """
    version = cfg.super_res.get('version', 'v1')
    if is_main_process():
        print(f'load super_res {version}')
        if version == 'FEC-DPS-MOE':
            moe_config = cfg.super_res.model.get('MoE_config', {})
            dct_extractor = moe_config.get('dct_extractor', 'linear')
            print(f"Using DCT extractor: {dct_extractor}")

    if version == 'v1':
        from .network_swinir import SwinIR as SRModel
    elif version == 'v2':
        from .network_swin2sr import Swin2SR as SRModel
    elif version in ('PS', 'CAEC-MoE', 'CADR'):
        # Added new versions to this model class, now mapping to the new file.
        from .network_swin2srDmoe import Swin2SR as SRModel
    elif version == 'swinfir':
        from .swinfir_arch import SwinFIR as SRModel
    elif version == 'FEC-DPS-MOE':
        from .network_swin2srDmoe import Swin2SR as SRModel
    else:
        raise ValueError(f"Unknown model version: {version}")

    model = SRModel(**cfg.super_res.model, model_version=version).to(cfg.device)

    return model
