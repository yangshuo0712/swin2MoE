from utils import is_main_process
def build_model(cfg):
    version = cfg.super_res.get('version', 'v1')
    if is_main_process():
        print('load super_res {}'.format(version))

    if version == 'v1':
        from .network_swinir import SwinIR as SRModel
    elif version == 'v2':
        from .network_swin2sr import Swin2SR as SRModel
    elif version == 'PS':
        from .network_swin2srForPS import Swin2SR as SRModel
    elif version == 'swinfir':
        from .swinfir_arch import SwinFIR as SRModel

    model = SRModel(**cfg.super_res.model).to(cfg.device)

    return model
