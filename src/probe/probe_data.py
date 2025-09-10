
import math
import torch
import torch.nn.functional as F

__all__ = ["make_stripe_probe", "make_checker_probe"]

def _gaussian_kernel2d(ks: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks-1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
    k = k / (k.sum() + 1e-8)
    return k

def _apply_blur_roi(img, mask, sigma=1.0, ksize=7):
    # img: [1,1,H,W] or [1,C,H,W]; mask: [1,1,H,W] in {0,1}
    device, dtype = img.device, img.dtype
    k = _gaussian_kernel2d(ksize, sigma, device, dtype).view(1,1,ksize,ksize)
    pad = ksize // 2
    blurred = F.conv2d(img, k.expand(img.shape[1],1,ksize,ksize), padding=pad, groups=img.shape[1])
    return img * (1 - mask) + blurred * mask

def _add_noise_roi(img, mask, snr_db=28.0):
    # Rough RMS estimate for per-image scaling
    s = img.flatten(2).std(dim=-1, unbiased=False).mean()
    noise_rms = s / (10.0**(snr_db/20.0) + 1e-8)
    noise = torch.randn_like(img) * noise_rms
    return img + noise * mask

def make_stripe_probe(
    H=128, W=128, C=4, theta_deg=0.0,
    fmin=0.03, fmax=0.24,
    roi_size=32,
    snr_db=28.0, blur_sigma=1.0,
    device='cpu', dtype=torch.float32,
    seed=123
):
    g = torch.Generator(device=device); g.manual_seed(seed)

    x = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, W).expand(H, W)
    y = torch.linspace(0, 1, H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    th = math.radians(theta_deg)
    t_raw = x*math.cos(th) + y*math.sin(th)
    t = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min() + 1e-8)

    f = fmin + (fmax - fmin) * t
    phase = 2*math.pi * f * t_raw
    I = 0.5 + 0.5 * torch.sin(phase)                 

    X = I.unsqueeze(0).repeat(C, 1, 1)                
    X = X.unsqueeze(0)                                 

    M_noise = torch.zeros((1,1,H,W), device=device, dtype=dtype)
    M_blur  = torch.zeros((1,1,H,W), device=device, dtype=dtype)

    def _rand_xy():
        x0 = torch.randint(low=0, high=max(1, W-roi_size), size=(1,), generator=g, device=device).item()
        y0 = torch.randint(low=0, high=max(1, H-roi_size), size=(1,), generator=g, device=device).item()
        return x0, y0

    x0n, y0n = _rand_xy()
    x0b, y0b = _rand_xy()
    # prevent overlap (simple one-time retry is enough here)
    if abs(x0n-x0b) < roi_size and abs(y0n-y0b) < roi_size:
        x0b, y0b = _rand_xy()

    M_noise[0,0, y0n:y0n+roi_size, x0n:x0n+roi_size] = 1.0
    M_blur [0,0, y0b:y0b+roi_size, x0b:x0b+roi_size] = 1.0
    M_clean = (1.0 - torch.clamp(M_noise + M_blur, 0, 1))

    X = X.to(device=device, dtype=dtype).clamp(0,1)
    X = _add_noise_roi(X, M_noise, snr_db=snr_db)
    X = _apply_blur_roi(X, M_blur, sigma=blur_sigma, ksize=7)
    X = X.clamp(0,1)

    f_true = f  # [H,W]
    masks = {
        "noise": M_noise[0,0].bool(),
        "blur":  M_blur [0,0].bool(),
        "clean": M_clean[0,0].bool(),
    }
    return X, f_true, masks

def make_checker_probe(
    H=128, W=128, C=4,
    fx=0.12, fy=0.08,
    roi_size=32,
    snr_db=28.0, blur_sigma=1.0,
    device='cpu', dtype=torch.float32,
    seed=123
):
    _ = seed  # reserved
    x = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, W).expand(H, W)
    y = torch.linspace(0, 1, H, device=device, dtype=dtype).view(H, 1).expand(H, W)

    I = 0.5 + 0.5*torch.sin(2*math.pi*fx*x) * torch.sin(2*math.pi*fy*y)
    X = I.unsqueeze(0).repeat(C,1,1).unsqueeze(0)  # [1,C,H,W]

    M_noise = torch.zeros((1,1,H,W), device=device, dtype=dtype)
    M_blur  = torch.zeros((1,1,H,W), device=device, dtype=dtype)

    # fixed corners for ROI for reproducibility
    x0n, y0n = 8, 8
    x0b, y0b = W-8-roi_size, H-8-roi_size
    M_noise[0,0, y0n:y0n+roi_size, x0n:x0n+roi_size] = 1.0
    M_blur [0,0, y0b:y0b+roi_size, x0b:x0b+roi_size] = 1.0
    M_clean = (1.0 - torch.clamp(M_noise + M_blur, 0, 1))

    X = _add_noise_roi(X, M_noise, snr_db=snr_db)
    X = _apply_blur_roi(X, M_blur, sigma=blur_sigma, ksize=7)
    X = X.clamp(0,1)

    f_true = torch.sqrt(torch.tensor(fx**2 + fy**2, device=device, dtype=dtype)).repeat(H,W)

    masks = {
        "noise": M_noise[0,0].bool(),
        "blur":  M_blur [0,0].bool(),
        "clean": M_clean[0,0].bool(),
    }
    return X, f_true, masks
