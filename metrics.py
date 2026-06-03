from __future__ import annotations

import numpy as np


def compute_clean_precision_on_eval(eval_loader, pred_np: np.ndarray) -> float:
    """
    Compute "clean precision" over the evaluation set.

    This is used to log how pure the selected "clean/labeled" subset is.

    Args:
        eval_loader: DataLoader whose dataset exposes `noise_label` and `clean_label`.
        pred_np: 1D bool array (length N, aligned with dataset order).
            True means "predicted clean" by the method.

    Returns:
        Precision (%) = TP / (TP + FP) * 100, where:
        - TP: predicted-clean AND truly-clean (noise_label == clean_label)
        - FP: predicted-clean BUT not truly-clean
    """
    ds = eval_loader.dataset
    clean_gt = (np.array(ds.noise_label, dtype=np.int64) == np.array(ds.clean_label, dtype=np.int64))
    pred_pos = pred_np.astype(bool)
    denom = int(pred_pos.sum())
    if denom == 0:
        return 0.0
    tp = int((pred_pos & clean_gt).sum())
    return 100.0 * tp / denom

