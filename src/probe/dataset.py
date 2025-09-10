
import torch
from torch.utils.data import Dataset
import random
from .probe_data import make_stripe_probe, make_checker_probe

__all__ = ["ProbeDataset"]

class ProbeDataset(Dataset):
    """
    Dataset that generates synthetic stripe/checker probes with local ROI distortions.
    Output shapes per sample:
      x: [C,H,W]  (float in [0,1])
      f_true: [H,W]  (float frequency per pixel; for checker it's constant)
      masks: dict[str]->[H,W] bool  with keys: 'noise', 'blur', 'clean'
    """
    def __init__(self, n=128, kind='stripe', device='cpu', seed=0):
        self.n, self.kind, self.device = n, kind, device
        self.rng = random.Random(seed)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        if self.kind == 'stripe':
            theta = self.rng.choice([0.0, 15.0, 30.0, 45.0])
            fmin  = 0.02
            fmax  = self.rng.uniform(0.20, 0.28)
            snr   = self.rng.uniform(24.0, 30.0)
            sigma = self.rng.choice([0.6, 0.8, 1.0])
            X, f_true, masks = make_stripe_probe(
                H=128, W=128, C=4, theta_deg=theta,
                fmin=fmin, fmax=fmax,
                snr_db=snr, blur_sigma=sigma,
                device=self.device, seed=123+idx
            )
        else:
            fx = self.rng.choice([0.06, 0.10, 0.14, 0.18])
            fy = self.rng.choice([0.06, 0.10, 0.14, 0.18])
            snr   = self.rng.uniform(24.0, 30.0)
            sigma = self.rng.choice([0.6, 0.8, 1.0])
            X, f_true, masks = make_checker_probe(
                H=128, W=128, C=4, fx=fx, fy=fy,
                snr_db=snr, blur_sigma=sigma,
                device=self.device, seed=123+idx
            )

        sample = {
            'x': X[0],            # [4,128,128]
            'f_true': f_true,     # [128,128]
            'masks': masks,       # dict[str]->[128,128] bool
        }
        return sample
