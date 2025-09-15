import math

import torch
import torch.nn.functional as F

__all__ = ["make_stripe_probe", "make_checker_probe"]


def _gaussian_kernel2d(ks: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / (k.sum() + 1e-8)
    return k


def _apply_blur_roi(img, mask, sigma=1.0, ksize=7):
    # img: [1,1,H,W] or [1,C,H,W]; mask: [1,1,H,W] in {0,1}
    device, dtype = img.device, img.dtype
    k = _gaussian_kernel2d(ksize, sigma, device, dtype).view(1, 1, ksize, ksize)
    pad = ksize // 2
    blurred = F.conv2d(
        img, k.expand(img.shape[1], 1, ksize, ksize), padding=pad, groups=img.shape[1]
    )
    return img * (1 - mask) + blurred * mask


def _add_noise_roi(img, mask, snr_db=28.0, g=None):
    s = img.flatten(2).std(dim=-1, unbiased=False).mean()
    noise_rms = s / (10.0 ** (snr_db / 20.0) + 1e-8)

    noise = torch.empty_like(img).normal_(mean=0.0, std=noise_rms, generator=g)
    return img + noise * mask

def make_stripe_probe(
    H=128,
    W=128,
    C=4,
    theta_deg=0.0,
    fmin_px=0.03,
    fmax_px=0.24,
    roi_size=32,
    snr_db=28.0,
    blur_sigma=1.0,
    device="cpu",
    dtype=torch.float32,
    seed=123,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    x = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
    y = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    th = math.radians(theta_deg)
    u = x * math.cos(th) + y * math.sin(th)
    u_min, u_max = u.min(), u.max()
    L = (u_max - u_min).clamp_min(1e-8)

    beta = (fmax_px - fmin_px) / L
    alpha = fmin_px - beta * u_min
    phase = 2 * math.pi * (alpha * u + 0.5 * beta * u * u)
    I = 0.5 + 0.5 * torch.sin(phase)

    X = I.unsqueeze(0).repeat(C, 1, 1).unsqueeze(0).clamp(0, 1)

    # ROI
    M_noise = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    M_blur = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

    def _rand_xy(g):
        x0 = torch.randint(
            0, max(1, W - roi_size), (1,), generator=g, device=device
        ).item()
        y0 = torch.randint(
            0, max(1, H - roi_size), (1,), generator=g, device=device
        ).item()
        return x0, y0

    x0n, y0n = _rand_xy(g)
    x0b, y0b = _rand_xy(g)
    if abs(x0n - x0b) < roi_size and abs(y0n - y0b) < roi_size:
        x0b, y0b = _rand_xy(g)
    M_noise[0, 0, y0n : y0n + roi_size, x0n : x0n + roi_size] = 1
    M_blur[0, 0, y0b : y0b + roi_size, x0b : x0b + roi_size] = 1
    M_clean = 1.0 - torch.clamp(M_noise + M_blur, 0, 1)

    X = _add_noise_roi(X, M_noise, snr_db=snr_db, g=g)
    X = _apply_blur_roi(X, M_blur, sigma=blur_sigma, ksize=7)
    X = X.clamp(0, 1)

    f_true = alpha + beta * u  # **cycles/pixel 的瞬时频率场**
    masks = {
        "noise": M_noise[0, 0].bool(),
        "blur": M_blur[0, 0].bool(),
        "clean": M_clean[0, 0].bool(),
    }
    return X, f_true, masks


def make_checker_probe(
    H=128,
    W=128,
    C=4,
    fx=8.0,
    fy=8.0,
    roi_size=32,
    snr_db=28.0,
    blur_sigma=1.0,
    device="cpu",
    dtype=torch.float32,
    seed=123,
    zero_mean_for_fft=True,
):
    px = max(1, int(round(W / (2.0 * max(fx, 1e-8)))))
    py = max(1, int(round(H / (2.0 * max(fy, 1e-8)))))

    y_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
    x_idx = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    board = ((y_idx // py) + (x_idx // px)) % 2  # 0/1 方块
    if zero_mean_for_fft:
        base = (board * 2 - 1).to(dtype)
        I = 0.5 + 0.5 * base
    else:
        I = board.to(dtype)

    X = I.unsqueeze(0).repeat(C, 1, 1).unsqueeze(0)

    M_noise = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    M_blur = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    x0n, y0n = 8, 8
    x0b, y0b = W - 8 - roi_size, H - 8 - roi_size
    M_noise[0, 0, y0n : y0n + roi_size, x0n : x0n + roi_size] = 1
    M_blur[0, 0, y0b : y0b + roi_size, x0b : x0b + roi_size] = 1
    M_clean = 1.0 - torch.clamp(M_noise + M_blur, 0, 1)

    X = _add_noise_roi(X, M_noise, snr_db=snr_db)
    X = _apply_blur_roi(X, M_blur, sigma=blur_sigma, ksize=7)
    X = X.clamp(0, 1)

    fx_px = 1.0 / (2.0 * px)
    fy_px = 1.0 / (2.0 * py)
    f_true = torch.full(
        (H, W), math.sqrt(fx_px**2 + fy_px**2), device=device, dtype=dtype
    )

    masks = {
        "noise": M_noise[0, 0].bool(),
        "blur": M_blur[0, 0].bool(),
        "clean": M_clean[0, 0].bool(),
    }
    return X, f_true, masks
