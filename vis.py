from __future__ import annotations

import os
from typing import Optional, Set, Tuple

import numpy as np
import torch


def plot_gmm_histogram_tri_dual(
    loss1: np.ndarray,
    clean_mask1: np.ndarray,
    id_rest_mask1: np.ndarray,
    ood_mask1: np.ndarray,
    loss2: np.ndarray,
    clean_mask2: np.ndarray,
    id_rest_mask2: np.ndarray,
    ood_mask2: np.ndarray,
    *,
    epoch: int,
    outdir: str = "./figure_vis",
) -> None:
    """
    Plot a 3-way histogram (clean / ID-rest / OOD) for two models and overlay a 2-GMM fit.

    This function is used for debugging/visualization only. It was originally implemented
    inline in `train_cifar100.py` and extracted here.

    Args:
        loss1/loss2: 1D numpy arrays of length N, normalized to roughly [0, 1].
        clean_mask*: bool masks (length N) for clean intersection subset.
        id_rest_mask*: bool masks (length N) for non-OOD and non-clean subset.
        ood_mask*: bool masks (length N) for OOD subset.
        epoch: epoch number used in the output filename.
        outdir: directory for saved figures.
    """
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture

    os.makedirs(outdir, exist_ok=True)

    def _fit_and_plot(ax, loss, clean_m, id_rest_m, ood_m, title: str):
        ax.hist(loss[clean_m], bins=100, density=True, alpha=0.55, label="Clean subset")
        ax.hist(loss[id_rest_m], bins=100, density=True, alpha=0.55, label="ID (non-clean)")
        ax.hist(loss[ood_m], bins=100, density=True, alpha=0.55, label="OOD subset")

        x = loss.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, n_init=10, tol=1e-2, reg_covar=5e-4, random_state=0)
        gmm.fit(x)

        x_range = np.linspace(0.0, 1.0, 1000).reshape(-1, 1)
        mix_pdf = np.exp(gmm.score_samples(x_range))
        resp = gmm.predict_proba(x_range)
        comp_pdf = resp * mix_pdf[:, None]

        ax.plot(x_range, mix_pdf, "-", lw=2.0, c="k", label="Mixture")
        ax.plot(x_range, comp_pdf[:, 0], "--", lw=1.8, label="Component")
        ax.plot(x_range, comp_pdf[:, 1], "--", lw=1.8, label="Component")

        means = np.sort(gmm.means_.reshape(-1))
        for m in means:
            ax.axvline(float(m), ymin=0, ymax=1, ls=":", lw=1.5, c="k")

        ax.set_xlabel("Normalized loss")
        ax.set_ylabel("Estimated pdf")
        ax.set_title(title)
        ax.legend(loc="upper right", prop={"size": 10})

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    _fit_and_plot(ax1, loss1, clean_mask1, id_rest_mask1, ood_mask1, f"Epoch-{epoch}:Model_1")
    ax2 = plt.subplot(1, 2, 2)
    _fit_and_plot(ax2, loss2, clean_mask2, id_rest_mask2, ood_mask2, f"Epoch-{epoch}:Model_2")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"gmm_hist_epoch{epoch}_tri.png"), dpi=300)
    plt.close()


@torch.no_grad()
def collect_features_for_tsne(model, loader, *, device: str = "cuda", max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (feature, label) arrays for t-SNE visualization.

    Assumes the loader yields batches like (inputs, targets, index) or compatible.

    Args:
        model: PyTorch model returning (features, logits).
        loader: DataLoader over the evaluation set.
        device: "cuda" or "cpu".
        max_points: Optional cap for number of points collected (for speed).

    Returns:
        features: numpy array [K, D]
        labels: numpy array [K]
    """
    model.eval()
    feats = []
    labels = []
    for xb, yb, _ in loader:
        xb = xb.to(device)
        fb, _ = model(xb)
        feats.append(fb.detach().cpu())
        labels.append(yb.detach().cpu())
        if max_points is not None and sum(len(a) for a in labels) >= max_points:
            break
    F = torch.cat(feats, 0).numpy()
    Y = torch.cat(labels, 0).numpy()
    if max_points is not None and len(Y) > max_points:
        idx = np.random.choice(len(Y), max_points, replace=False)
        F, Y = F[idx], Y[idx]
    return F, Y


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epoch: int,
    outdir: str = "./figure_vis",
    only_id: bool = True,
    id_set: Optional[Set[int]] = None,
    max_points_per_class: Optional[int] = 800,
) -> None:
    """
    Run t-SNE and save a scatter plot.

    Args:
        features: numpy array [N, D]
        labels: numpy array [N]
        epoch: epoch number used in output filename.
        outdir: directory to save images.
        only_id: if True, restrict to labels in `id_set`.
        id_set: set of ID class labels in the *original label space*.
        max_points_per_class: optional downsampling cap per class to keep plots readable.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    os.makedirs(outdir, exist_ok=True)

    X, Y = features, labels
    if only_id and id_set is not None:
        mask = np.isin(Y, list(id_set))
        X, Y = X[mask], Y[mask]

    if max_points_per_class is not None:
        idx_keep = []
        for c in np.unique(Y):
            idx_c = np.where(Y == c)[0]
            if len(idx_c) > max_points_per_class:
                idx_c = np.random.choice(idx_c, max_points_per_class, replace=False)
            idx_keep.append(idx_c)
        idx_keep = np.concatenate(idx_keep)
        X, Y = X[idx_keep], Y[idx_keep]

    Z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(X)

    plt.figure(figsize=(6, 6))
    for c in np.unique(Y):
        m = (Y == c)
        plt.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.7, label=str(c))
    plt.xticks([])
    plt.yticks([])
    plt.title(f"t-SNE @ epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"tsne_epoch{epoch}.png"), dpi=300)
    plt.close()

