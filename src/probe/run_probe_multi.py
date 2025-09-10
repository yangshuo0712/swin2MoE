# probe/run_probe_multi.py
import argparse, os, re, json, torch, torch.nn.functional as F
from .dataset import ProbeDataset
from .metrics import r2_linear, cohens_d
from .capture_hooks import ProbeCollector

def _pool2d(x2d, Ht, Wt):
    return F.adaptive_avg_pool2d(x2d.unsqueeze(0).unsqueeze(0), (Ht, Wt)).squeeze()

def _spectral_summary(y_freq, Ht, Wt):
    N, K = y_freq.shape
    y = y_freq.abs().view(Ht, Wt, K)
    w = y / (y.sum(-1, keepdim=True) + 1e-6)
    k = torch.arange(K, device=y.device).view(1,1,K)
    mu = (w * k).sum(-1)         # [Ht,Wt]
    kpk = y.argmax(-1)           # [Ht,Wt]
    return mu, kpk

def _build_model(args, device):
    if args.demo:
        from .run_probe import _build_demo_model
        return _build_demo_model(device=device)
    if not args.model_path:
        raise SystemExit("非 demo 模式需要 --model-path 形如 pkg.mod:make_model")
    pkg, fac = args.model_path.split(":")
    factory = getattr(__import__(pkg, fromlist=[fac]), fac)
    return factory().to(device).eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", type=str, default="stripe", choices=["stripe","checker"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", type=str, default="./probe_out_multi")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model-path", type=str, default="")
    ap.add_argument("--include", type=str, default="dafe", help="正则，匹配 DAFE 模块名")
    ap.add_argument("--exclude", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(args, device)
    model.eval()

    # 1) 挂钩所有 DAFE（只读，不改 forward）
    collector = ProbeCollector(include_regex=args.include, exclude_regex=args.exclude)
    collector.attach(model)

    # 2) 数据集（逐样本评估，保证对齐）
    ds = ProbeDataset(n=args.n, kind=args.kind, device=device, seed=42)

    # 3) 聚合器
    agg = {}
    with torch.inference_mode():
        for i in range(len(ds)):
            s = ds[i]
            X = s["x"].unsqueeze(0).to(device)    # [1,4,128,128]
            _ = model(X)                          # 触发 hooks，数据进 collector.records

            for name, rec in collector.records.items():
                y_freq = rec["y_freq"]            # CPU tensor
                edge   = rec["edge"]
                noise  = rec["noise"]
                x_size = rec["x_size"] or (16, 16)  # 若没有 x_size，就假定一个合理 token 网格

                Ht, Wt = x_size
                mu, kpk = _spectral_summary(y_freq, Ht, Wt)

                f_t  = _pool2d(s["f_true"].cpu(), Ht, Wt)
                mN   = (_pool2d(s["masks"]["noise"].float().cpu(), Ht, Wt) > 0.5)
                mB   = (_pool2d(s["masks"]["blur"] .float().cpu(), Ht, Wt) > 0.5)
                mC   = (_pool2d(s["masks"]["clean"].float().cpu(), Ht, Wt) > 0.5)

                if name not in agg:
                    agg[name] = {"mu":[], "k":[], "f":[], "noise_in":[], "noise_clean":[], "mu_blur":[], "mu_clean":[]}

                a = agg[name]
                a["mu"].append(mu.flatten())
                a["k"].append(kpk.flatten())
                a["f"].append(f_t.flatten())
                a["noise_in"].append(noise.view(Ht, Wt)[mN].flatten())
                a["noise_clean"].append(noise.view(Ht, Wt)[mC].flatten())
                a["mu_blur"].append(mu[mB].flatten())
                a["mu_clean"].append(mu[mC].flatten())

    collector.detach()

    # 4) 逐层指标
    layer_metrics = {}
    for name, A in agg.items():
        cat = lambda L: torch.cat(L, 0) if len(L) else None
        mu_all, k_all, f_all = cat(A["mu"]), cat(A["k"]), cat(A["f"])
        n_in, n_cl = cat(A["noise_in"]), cat(A["noise_clean"])
        mu_bl, mu_cl = cat(A["mu_blur"]), cat(A["mu_clean"])

        r2_mu = float("nan") if (mu_all is None or f_all is None) else r2_linear(mu_all, f_all)
        r2_k  = float("nan") if (k_all  is None or f_all is None) else r2_linear(k_all.float(), f_all)
        d_n   = None
        if n_in is not None and n_cl is not None and n_in.numel()>0 and n_cl.numel()>0:
            d_n = cohens_d(n_in, n_cl)
        dmu   = None
        if mu_bl is not None and mu_cl is not None and mu_bl.numel()>0 and mu_cl.numel()>0:
            dmu = (mu_bl.mean() - mu_cl.mean()).item()

        layer_metrics[name] = {
            "R2_mu_vs_f": r2_mu,
            "R2_k_vs_f": r2_k,
            "Cohens_d_noiseproxy": (float(d_n) if d_n is not None else None),
            "Delta_mu_blur_minus_clean": (float(dmu) if dmu is not None else None),
            "count_tokens": int(f_all.numel()) if f_all is not None else 0,
        }

    # 5) 保存
    with open(os.path.join(args.out, "layer_metrics.json"), "w") as f:
        json.dump(layer_metrics, f, indent=2)

    lines = ["layer,R2_mu,R2_k,d_noise,Delta_mu,ntokens"]
    for name, m in layer_metrics.items():
        dnoise = "NA" if m["Cohens_d_noiseproxy"] is None else f"{m['Cohens_d_noiseproxy']:.4f}"
        dmu    = "NA" if m["Delta_mu_blur_minus_clean"] is None else f"{m['Delta_mu_blur_minus_clean']:.4f}"
        lines.append(f"{name},{m['R2_mu_vs_f']:.4f},{m['R2_k_vs_f']:.4f},{dnoise},{dmu},{m['count_tokens']}")
    with open(os.path.join(args.out, "layer_metrics.csv"), "w") as f:
        f.write("\n".join(lines))

    print(f"[done] per-layer metrics saved to {args.out}")

if __name__ == "__main__":
    main()
