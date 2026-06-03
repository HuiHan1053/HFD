from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def initialize_prototypes(features: torch.Tensor, labels: torch.Tensor, *, num_classes: int, num_prototypes: int) -> torch.Tensor:
    """
    Initialize `num_prototypes` prototypes per class.

    This is extracted from `train_cifar100.py` (prototype initialization during the
    first semi-supervised epoch).

    Args:
        features: Tensor of shape [N, D]. Typically L2-normalized features.
        labels: Tensor of shape [N]. Integer class labels in [0, num_classes).
        num_classes: Number of ID classes.
        num_prototypes: Number of prototypes per class (M).

    Returns:
        prototypes: Tensor of shape [num_classes, M, D].

    Notes:
        - If a class has fewer than M samples, we fall back to using the class mean
          and repeating it M times.
        - Otherwise, we run KMeans within the class to obtain M centers.
    """
    M = int(num_prototypes)
    D = int(features.shape[1])
    device = features.device
    prototypes = torch.zeros(num_classes, M, D, device=device)

    for c in range(num_classes):
        feats_c = features[labels == c]
        if feats_c.size(0) < M:
            proto = F.normalize(feats_c.mean(dim=0), dim=-1)
            prototypes[c] = proto.unsqueeze(0).repeat(M, 1)
        else:
            km = KMeans(n_clusters=M, n_init="auto", random_state=0)
            km.fit(feats_c.detach().cpu().numpy())
            centers = torch.tensor(km.cluster_centers_, device=device)
            prototypes[c] = F.normalize(centers, dim=-1)
    return prototypes


@torch.no_grad()
def update_prototypes(
    *,
    prototypes: torch.Tensor,
    sample_features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Update prototypes using an EMA-style assignment-and-mean step.

    This is the lightweight version of the update logic from `train_cifar100.py`.
    The caller is responsible for filtering OOD samples; this function assumes
    `sample_features` and `labels` are already the subset you want to use.

    Args:
        prototypes:
            Either [C, D] (single prototype per class) or [C, M, D] (multi-prototype).
        sample_features: [B, D] batch features used to update prototypes.
        labels: [B] integer labels for the batch features.
        alpha: EMA step size. Higher means faster adaptation.

    Returns:
        Updated `prototypes` tensor with the same shape as input.

    Behavior:
        - Single-prototype case ([C, D]):
          each class prototype is updated towards the mean of its samples.
        - Multi-prototype case ([C, M, D]):
          samples are assigned to the closest prototype (by Euclidean distance),
          and each prototype is updated towards its assigned-sample mean.
    """
    device = sample_features.device
    prototypes = prototypes.to(device)
    labels = labels.to(device)

    if sample_features.numel() == 0:
        return prototypes

    x = F.normalize(sample_features, dim=1)
    C = int(prototypes.shape[0])

    if prototypes.ndim == 2:
        for c in range(C):
            cls_feats = x[labels == c]
            if cls_feats.numel() == 0:
                continue
            proto_mean = cls_feats.mean(dim=0)
            prototypes[c] = (1 - alpha) * prototypes[c] + alpha * proto_mean
        return F.normalize(prototypes, dim=1)

    M = int(prototypes.shape[1])
    for c in range(C):
        cls_feats = x[labels == c]
        if cls_feats.numel() == 0:
            continue
        d = torch.cdist(cls_feats, prototypes[c])
        assign = d.argmin(dim=1)
        for m in range(M):
            feats_m = cls_feats[assign == m]
            if feats_m.numel() == 0:
                continue
            mean_m = feats_m.mean(dim=0)
            prototypes[c, m] = (1 - alpha) * prototypes[c, m] + alpha * mean_m
    return F.normalize(prototypes, dim=2)

