
"""
End-to-end script:
- build dataset
- run forward through your model
- hook DAFE to capture (y_freq, edge, noise, x_size)
- compute R^2 (frequency tracking) + Cohen's d (ROI separation)
- save plots

Usage (demo, no external model needed):
    python run_probe.py --demo --kind stripe --n 64 --out /tmp/probe_out

Usage (with your model):
    python run_probe.py --model-path your_model_package.module:make_model --dafe "backbone.blocks[3].moe.dafe" --kind stripe --n 128 --out ./out
Where `your_model_package.module:make_model` should point to a callable returning an initialized model.
"""
import argparse, os, importlib, sys
import torch
from torch.utils.data import DataLoader
from .dataset import ProbeDataset
from .hooks import resolve_module_by_path, DAFEHook
from .metrics import spectral_summary, r2_linear, cohens_d, downsample_mask
from .plotting import plot_regression, plot_distributions

def _build_demo_model(device='cpu'):
    """
    Minimal demo model that contains a DAFE-like module with `last_debug`.
    It projects input to features, flattens, calls DAFE, then returns identity.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class DistortionAwareFrequencyExtractor(nn.Module):
        def __init__(self, input_size, out_dim, dct_K=16):
            super().__init__()
            self.C=input_size; self.out_dim=out_dim; self.K=dct_K
            # fixed DCT basis
            n = torch.arange(self.C, dtype=torch.float32).unsqueeze(0)
            k = torch.arange(self.K, dtype=torch.float32).unsqueeze(1)
            basis = torch.cos(torch.pi * (n + 0.5) * k / self.C)
            basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-6)
            self.register_buffer("dct_basis", basis, persistent=True)
            self.alpha = nn.Parameter(torch.ones(self.K))
            self.freq_res = nn.Linear(self.C, self.K, bias=False)
            nn.init.zeros_(self.freq_res.weight)

            sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
            sobel_y = torch.tensor([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])
            sobel = torch.stack([sobel_x, sobel_y]).unsqueeze(1).float()
            self.register_buffer("sobel_kernel", sobel, persistent=True)
            avg = torch.ones((1,1,3,3))/9.0
            self.register_buffer("avg_kernel", avg.float(), persistent=True)
            self.intensity_head = nn.Linear(self.C, 1, bias=False)
            nn.init.normal_(self.intensity_head.weight, std=1e-3)
            self.beta_edge = nn.Parameter(torch.tensor(1.0))
            self.beta_noise = nn.Parameter(torch.tensor(1.0))
            self.norm = nn.LayerNorm(self.K+2)
            self.proj = nn.Linear(self.K+2, out_dim)

        def _like(self, t, like): return t.to(device=like.device, dtype=like.dtype)

        def _freq_branch(self, tokens):
            basis = self._like(self.dct_basis, tokens)
            y_fixed = torch.nn.functional.linear(tokens, basis)
            alpha = self._like(self.alpha, tokens)
            y_fixed = y_fixed * alpha.view(1,-1)
            y_res = self.freq_res(tokens)
            return y_fixed + y_res

        @torch.no_grad()
        def _infer_B(self, N, H, W):
            assert (H*W) > 0 and (N % (H*W) == 0)
            return N // (H*W)

        def _edge_noise_branch(self, tokens, x_size):
            H, W = x_size; N, C = tokens.shape
            B = self._infer_B(N, H, W)
            intensity = self.intensity_head(tokens).view(B, H, W, 1).permute(0,3,1,2).contiguous()
            sobel_k = self._like(self.sobel_kernel, intensity)
            avg_k   = self._like(self.avg_kernel, intensity)
            eps = torch.tensor(1e-6, device=intensity.device, dtype=intensity.dtype)
            grad = torch.nn.functional.conv2d(intensity, sobel_k, padding=1)
            gx, gy = grad[:,0:1], grad[:,1:2]
            edge = torch.sqrt(gx*gx + gy*gy + eps)
            smooth = torch.nn.functional.conv2d(intensity, avg_k, padding=1)
            resid = (intensity - smooth).abs()
            noise = torch.relu(resid - edge)
            edge = edge.view(B, H*W, 1).reshape(N,1)
            noise = noise.view(B, H*W, 1).reshape(N,1)
            edge = self._like(self.beta_edge, tokens) * edge
            noise = self._like(self.beta_noise, tokens) * noise
            return edge, noise

        def forward(self, tokens, x_size):
            y_freq = self._freq_branch(tokens)
            edge, noise = self._edge_noise_branch(tokens, x_size)
            y = torch.cat([y_freq, edge, noise], dim=-1)
            y = self.norm(y)
            out = self.proj(y)
            self.last_debug = {"y_freq": y_freq.detach(), "edge": edge.detach(),
                               "noise": noise.detach(), "x_size": x_size}
            return out

    class ToyModel(nn.Module):
        def __init__(self, in_ch=4, feat=32, dct_K=16, out_dim=16):
            super().__init__()
            self.enc = nn.Conv2d(in_ch, feat, kernel_size=1, bias=False)
            self.dafe = DistortionAwareFrequencyExtractor(feat, out_dim, dct_K=dct_K)

        def forward(self, x):
            # x: [B,4,128,128]
            B, C, H, W = x.shape
            f = self.enc(x)  # [B,feat,H,W]
            tokens = f.permute(0,2,3,1).reshape(B*H*W, -1)
            _ = self.dafe(tokens, x_size=(H,W))
            return x  # identity, we only care about dafe.last_debug

    m = ToyModel().to(device)
    m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", type=str, default="stripe", choices=["stripe","checker"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=str, default="./probe_out")
    ap.add_argument("--demo", action="store_true", help="use built-in toy model")
    ap.add_argument("--model-path", type=str, default="", help="pythonpath:factory e.g. pkg.mod:make_model")
    ap.add_argument("--dafe", type=str, default="", help='module path to DAFE, e.g., "backbone.blocks[3].moe.dafe"')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    if args.demo:
        model = _build_demo_model(device=device)
        dafe_mod = model.dafe
    else:
        if not args.model_path or not args.dafe:
            raise SystemExit("Provide --model-path and --dafe when not using --demo.")
        pkg, fac = args.model_path.split(":")
        factory = getattr(importlib.import_module(pkg), fac)
        model = factory()
        model.to(device).eval()
        dafe_mod = resolve_module_by_path(model, args.dafe)

    # Dataset & loader
    ds = ProbeDataset(n=args.n, kind=args.kind, device=device, seed=42)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # Hook DAFE
    with DAFEHook(dafe_mod) as hk:
        # Forward
        for batch in dl:
            X = batch["x"].to(device)  # [B,4,128,128]
            _ = model(X)

    # Collect tensors
    # We concatenate across batches (records per forward call)
    y_freq_list = []
    edge_list = []
    noise_list = []
    Ht = Wt = None
    for rec in hk.records:
        y_freq_list.append(rec["y_freq"])
        edge_list.append(rec["edge"])
        noise_list.append(rec["noise"])
        if Ht is None:
            Ht, Wt = rec["x_size"]
    y_freq = torch.cat(y_freq_list, dim=0)  # [N_total,K]
    edge = torch.cat(edge_list, dim=0)      # [N_total,1]
    noise = torch.cat(noise_list, dim=0)    # [N_total,1]

    # Rebuild per-batch B,Ht,Wt (all batches same sizes)
    # N_total = sum(B_i * Ht * Wt). We'll compute B from dataset len & batch size for plotting only.
    B_total = len(ds)
    # Compute spectral summaries
    mu, k_peak = spectral_summary(y_freq, B_total, Ht, Wt)  # [B,Ht,Wt] each
    edge_map = edge.view(B_total, Ht, Wt)
    noise_map = noise.view(B_total, Ht, Wt)

    # Downsample ground-truth frequency and masks to token grid
    import torch.nn.functional as F
    def pool_to_tokens(t):  # t: [B,H,W] -> [B,Ht,Wt]
        return F.adaptive_avg_pool2d(t.unsqueeze(1), (Ht, Wt)).squeeze(1)
    f_true_stack = []
    noise_mask_stack = []
    blur_mask_stack = []
    clean_mask_stack = []
    # Need to re-run the dataset iteration to grab f_true/masks in order
    for i in range(len(ds)):
        s = ds[i]
        f_true_stack.append(s["f_true"])
        noise_mask_stack.append(s["masks"]["noise"].float())
        blur_mask_stack.append(s["masks"]["blur"].float())
        clean_mask_stack.append(s["masks"]["clean"].float())
    f_true = torch.stack(f_true_stack, dim=0).to(y_freq.device)       # [B,H,W]
    m_noise = torch.stack(noise_mask_stack, dim=0).to(y_freq.device)  # [B,H,W]
    m_blur  = torch.stack(blur_mask_stack , dim=0).to(y_freq.device)
    m_clean = torch.stack(clean_mask_stack, dim=0).to(y_freq.device)

    f_t = pool_to_tokens(f_true)                         # [B,Ht,Wt]
    noise_t = (pool_to_tokens(m_noise) > 0.5)            # bool
    blur_t  = (pool_to_tokens(m_blur ) > 0.5)
    clean_t = (pool_to_tokens(m_clean) > 0.5)

    # Compute metrics
    from .metrics import r2_linear, cohens_d
    r2_mu = r2_linear(mu, f_t)
    r2_kp = r2_linear(k_peak.float(), f_t)
    d_noise = cohens_d(noise_map[noise_t], noise_map[clean_t])
    delta_mu = (mu[blur_t].mean() - mu[clean_t].mean()).item()

    # Save metrics
    metrics = {
        "R2_centroid_vs_truefreq": r2_mu,
        "R2_peakidx_vs_truefreq": r2_kp,
        "Cohens_d_noise_vs_clean_on_noiseproxy": d_noise,
        "Delta_mu_blur_minus_clean": float(delta_mu),
        "Ht": int(Ht), "Wt": int(Wt), "B": int(B_total),
    }
    import json
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    import numpy as np
    from .plotting import plot_regression, plot_distributions
    plot_regression(mu.cpu().numpy(), f_t.cpu().numpy(),
                    os.path.join(args.out, "regression_mu_vs_ftrue.png"),
                    title="Spectral centroid vs true frequency")
    plot_regression(k_peak.cpu().numpy().astype(np.float32), f_t.cpu().numpy(),
                    os.path.join(args.out, "regression_kpeak_vs_ftrue.png"),
                    title="Spectral peak index vs true frequency")
    plot_distributions(noise_map[noise_t].cpu().numpy(), noise_map[clean_t].cpu().numpy(),
                       os.path.join(args.out, "noise_proxy_noise_vs_clean.png"),
                       label_a="Noise ROI", label_b="Clean", title="Noise proxy distributions")
    plot_distributions(mu[blur_t].cpu().numpy(), mu[clean_t].cpu().numpy(),
                       os.path.join(args.out, "mu_blur_vs_clean.png"),
                       label_a="Blur ROI", label_b="Clean", title="Spectral centroid (blur vs clean)")

    print("Saved metrics and figures to:", args.out)

if __name__ == "__main__":
    main()
