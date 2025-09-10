
# Probe pipeline (DAFE verification)

This folder contains a small, self-contained pipeline to verify that **DAFE detects distortions** using controllable **stripe/checker probes** with local ROI noise/blur.

## Layout

- `probe_data.py` — functions to generate stripe/checker images with ROI noise/blur (shape: `[1, 4, 128, 128]`).
- `dataset.py` — `ProbeDataset` for batching.
- `hooks.py` — utility to resolve your DAFE module by path and a `DAFEHook` that records `last_debug`.
- `metrics.py` — compute spectral centroid/peak, R², Cohen's d, and mask downsampling.
- `plotting.py` — matplotlib plotting helpers (no seaborn).
- `run_probe.py` — end-to-end script: dataset → forward → capture DAFE → metrics → plots.

## Quick start (demo)

> The demo uses a tiny built-in model that contains a DAFE-like extractor and populates `last_debug`.

```bash
python -m probe.run_probe --demo --kind stripe --n 64 --out ./probe_out
```

Outputs:
- `metrics.json`
- `regression_mu_vs_ftrue.png`
- `regression_kpeak_vs_ftrue.png`
- `noise_proxy_noise_vs_clean.png`
- `mu_blur_vs_clean.png`

## Use with your model

1. Ensure your DAFE module stores a debug dict during forward, for example:

```python
# inside DistortionAwareFrequencyExtractor.forward(...)
self.last_debug = {
    "y_freq": y_freq.detach(),        # [N, K]
    "edge": edge.detach(),            # [N, 1]
    "noise": noise.detach(),          # [N, 1]
    "x_size": x_size                  # (Ht, Wt) token grid
}
```

2. Run the probe with your model factory and the module path to DAFE:

```bash
python -m probe.run_probe \
  --model-path your_package.model_factory:make_model \
  --dafe "backbone.blocks[3].moe.dafe" \
  --kind stripe --n 128 --out ./probe_out
```

`--model-path` must be of the form `pkg.subpkg.module:factory_function`, which returns an initialized model. The `--dafe` string is a dotted path to your DAFE module; indices are supported, e.g. `blocks[3]`.

## How many samples?

- Fast sanity check: `--n 48` (≈5 minutes)
- Paper-level stats: `--n 160` (stripe) and/or `--n 80` (checker)

Keep noise/blur light (SNR 24–30 dB; σ≤1.0) to probe detection rather than break the backbone.

## Notes

- All tensors are generated at `[1, 4, 128, 128]` to match your training size.
- We downsample the pixel masks and true frequency to the token grid `(Ht, Wt)` before computing metrics.
- Plots use matplotlib defaults, one chart per figure (no subplots), and no explicit colors, per constraints.
