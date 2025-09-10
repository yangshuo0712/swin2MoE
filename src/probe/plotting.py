
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_regression", "plot_distributions"]

def plot_regression(k_or_mu, f_true, out_path, title="Frequency regression"):
    # Single scatter with linear fit
    x = k_or_mu.reshape(-1)
    y = f_true.reshape(-1)
    plt.figure()
    plt.scatter(x, y, s=4, alpha=0.5)
    # linear fit
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    plt.plot(x, yhat, linewidth=2)
    plt.xlabel("Predicted spectral index / centroid")
    plt.ylabel("True frequency (cycles/pixel)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_distributions(a, b, out_path, label_a="A", label_b="B", title="Distribution comparison"):
    plt.figure()
    plt.hist(a.reshape(-1), bins=50, alpha=0.6, label=label_a)
    plt.hist(b.reshape(-1), bins=50, alpha=0.6, label=label_b)
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
