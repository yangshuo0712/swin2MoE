# src/probe/run_probe_multi.py
import argparse
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import parse_config
from super_res.model import build_model as super_res_build_model
from chk_loader import load_checkpoint, load_state_dict_model_only

from .dataset import ProbeDataset
from .metrics import spectral_summary, r2_linear, cohens_d, downsample_mask
from .capture_hooks import ProbeCollector
from .plotting import plot_regression, plot_distributions

def downsample_map_continuous(M: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """
    Downsamples a continuous-valued map using adaptive average pooling.

    Args:
        M (torch.Tensor): The input tensor of shape [B, H, W].
        Ht (int): The target height for downsampling.
        Wt (int): The target width for downsampling.

    Returns:
        torch.Tensor: The downsampled tensor of shape [B, Ht, Wt].
    """
    if M.dim() != 3:
        raise ValueError(f"downsample_map_continuous expects input of shape [B, H, W], but got {list(M.shape)}")
    return F.adaptive_avg_pool2d(M.unsqueeze(1).float(), (Ht, Wt)).squeeze(1)

def run_probe_pipeline(cfg):
    """
    Executes the complete probe evaluation pipeline:
    1) Builds a probe dataset (e.g., checkerboards, stripes).
    2) Runs model inference and captures intermediate features using hooks, while caching
       the corresponding ground truth data to ensure alignment.
    3) Calculates metrics for frequency, edge, and noise proxies.
    4) Visualizes results and saves metrics to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output, exist_ok=True)

    print("Loading model...")
    model = super_res_build_model(cfg)
    checkpoint = load_checkpoint(cfg)
    load_state_dict_model_only(model, checkpoint)
    model.to(device).eval()
    print("Model loaded successfully.")

    collector = ProbeCollector()
    collector.attach(model)

    print("Building probe dataset...")
    ds = ProbeDataset(n=cfg.probe.n_samples, kind=cfg.probe.kind, device=device, seed=42)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.probe.batch_size, shuffle=False)

    # -- Cache ground truth and masks during the forward pass to ensure perfect alignment --
    f_true_cache = []
    m_noise_cache, m_blur_cache, m_clean_cache = [], [], []

    with torch.inference_mode():
        for batch in tqdm(dl, desc="Running probe evaluation"):
            x = batch["x"].to(device, non_blocking=True)
            _ = model(x)

            # Cache ground truth/masks to CPU
            f_true_cache.append(batch["f_true"].detach().float().cpu())
            m = batch["masks"]
            m_noise_cache.append(m["noise"].detach().float().cpu())
            m_blur_cache.append(m["blur"].detach().float().cpu())
            m_clean_cache.append(m["clean"].detach().float().cpu())

    # -- Sanity Check #1: Smoke test a single record for shape and grid consistency --
    if not collector.records:
        raise RuntimeError(
            "ProbeCollector did not capture any records. "
            "Please check if DistortionAwareFrequencyExtractor modules were found and hooked correctly."
        )

    any_layer = next(iter(collector.records.keys()))
    first_rec = collector.records[any_layer][0]
    Ht, Wt = first_rec["x_size"]
    B0 = first_rec["edge"].shape[0]

    assert first_rec["y_freq"].dim() == 2, "y_freq should be [N, K]"
    N0, K0 = first_rec["y_freq"].shape
    assert N0 == B0 * Ht * Wt, f"y_freq N={N0} does not match B*Ht*Wt={B0*Ht*Wt}"
    assert first_rec["edge"].shape == (B0, Ht, Wt), f"edge shape should be [B, Ht, Wt], but got {first_rec['edge'].shape}"
    assert first_rec["noise"].shape == (B0, Ht, Wt), f"noise shape should be [B, Ht, Wt], but got {first_rec['noise'].shape}"
    print(f"✅ Sanity check #1 passed for a single record: layer={any_layer}, B={B0}, Ht×Wt={Ht}×{Wt}, K={K0}")

    # -- Sanity Check #2: Global consistency (total samples / token grid / num frequency bands K) --
    B_total = sum(t.shape[0] for t in f_true_cache)
    for lname, recs in collector.records.items():
        sum_B = sum(r["edge"].shape[0] for r in recs)
        assert sum_B == B_total, f"{lname}: Number of recorded samples {sum_B} != number of inference samples {B_total}"

        hw_set = {tuple(r["x_size"]) for r in recs}
        assert len(hw_set) == 1, f"{lname}: Inconsistent x_size found across different records: {hw_set}"
        (Ht_chk, Wt_chk) = next(iter(hw_set))

        k_set = {r["y_freq"].shape[1] for r in recs}
        assert len(k_set) == 1, f"{lname}: Inconsistent number of frequency bands (K) in y_freq across records: {k_set}"
    print("✅ Sanity check #2 passed for global consistency (total samples / token grid / num frequency bands K).")

    print("Data collection complete.")

    # -- Aggregate results from all batches --
    aggregated_data = {}
    for layer_name, records in collector.records.items():
        aggregated_data[layer_name] = {
            "y_freq": torch.cat([r["y_freq"] for r in records], dim=0),   # [B_total*Ht*Wt, K]
            "edge":   torch.cat([r["edge"]   for r in records], dim=0),   # [B_total, Ht, Wt]
            "noise":  torch.cat([r["noise"]  for r in records], dim=0),   # [B_total, Ht, Wt]
            "x_size": records[0]["x_size"],                                # (Ht, Wt)
        }

    # -- Concatenate and downsample the ground truth/masks to the token grid --
    f_true_full = torch.cat(f_true_cache, dim=0)  # Shape: [B_total, H, W] or [B_total, 1, H, W]
    if f_true_full.dim() == 4 and f_true_full.size(1) == 1:
        f_true_full = f_true_full.squeeze(1)      # -> [B_total, H, W]

    m_noise_full = torch.cat(m_noise_cache, dim=0)
    m_blur_full  = torch.cat(m_blur_cache,  dim=0)
    m_clean_full = torch.cat(m_clean_cache, dim=0)
    if m_noise_full.dim() == 4 and m_noise_full.size(1) == 1:
        m_noise_full = m_noise_full.squeeze(1)
        m_blur_full  = m_blur_full.squeeze(1)
        m_clean_full = m_clean_full.squeeze(1)

    # Use a consistent token grid size (taken from the first layer's records)
    (Ht, Wt) = next(iter(aggregated_data.values()))["x_size"]

    f_true_token  = downsample_map_continuous(f_true_full, Ht, Wt)  # Continuous values
    m_noise_token = downsample_mask(m_noise_full, Ht, Wt)           # Boolean mask
    m_blur_token  = downsample_mask(m_blur_full,  Ht, Wt)           # Boolean mask
    m_clean_token = downsample_mask(m_clean_full, Ht, Wt)           # Boolean mask

    # -- Calculate metrics and generate plots for each layer --
    print("Calculating metrics...")
    results = {}
    for layer_name, data in aggregated_data.items():
        y_freq = data["y_freq"]                                     # [B_total*Ht*Wt, K]
        mu, k_peak = spectral_summary(y_freq, B_total, Ht, Wt)       # -> [B_total, Ht, Wt]

        edge_map  = data["edge"]                                    # [B_total, Ht, Wt]
        noise_map = data["noise"]                                   # [B_total, Ht, Wt]

        # R^2 for regression fit
        r2_mu_vs_f = r2_linear(mu, f_true_token)
        r2_k_vs_f  = r2_linear(k_peak.float(), f_true_token)

        # Effect size (Noise ROI vs. Clean ROI)
        if torch.any(m_noise_token) and torch.any(m_clean_token):
            cohen_d_noise = cohens_d(noise_map[m_noise_token], noise_map[m_clean_token])
        else:
            cohen_d_noise = float('nan')

        # Difference in spectral centroid between blur and clean regions
        if torch.any(m_blur_token) and torch.any(m_clean_token):
            delta_mu_blur_clean = (mu[m_blur_token].mean() - mu[m_clean_token].mean()).item()
        else:
            delta_mu_blur_clean = float('nan')

        results[layer_name] = {
            "R2_mu_vs_f": float(r2_mu_vs_f),
            "R2_k_vs_f": float(r2_k_vs_f),
            "Cohens_d_noise": float(cohen_d_noise),
            "Delta_mu_blur_clean": float(delta_mu_blur_clean),
        }

        # Visualization (only plot distributions if ROIs are not empty)
        plot_regression(
            mu.cpu().numpy(), f_true_token.cpu().numpy(),
            os.path.join(cfg.output, f"{layer_name}_regr_mu.png"),
            title=f"{layer_name}: Spectral Centroid vs True Frequency"
        )
        if torch.any(m_noise_token) and torch.any(m_clean_token):
            plot_distributions(
                noise_map[m_noise_token].cpu().numpy(),
                noise_map[m_clean_token].cpu().numpy(),
                os.path.join(cfg.output, f"{layer_name}_dist_noise.png"),
                label_a="Noise ROI", label_b="Clean ROI",
                title=f"{layer_name}: Noise Proxy Distribution"
            )

    # -- Save results to disk --
    json_path = os.path.join(cfg.output, "probe_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {json_path}")

    df = pd.DataFrame.from_dict(results, orient="index")
    csv_path = os.path.join(cfg.output, "probe_metrics.csv")
    df.to_csv(csv_path)
    print(f"Metrics saved to {csv_path}")

    collector.detach()
    print("Evaluation complete.")
