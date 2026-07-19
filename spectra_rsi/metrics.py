"""Primary metrics (Sec. 12.5)."""
import numpy as np


def normalized_delta_error(delta_hat, delta_true, eps=1e-9):
    return float(np.linalg.norm(delta_hat - delta_true)
                 / (np.linalg.norm(delta_true) + eps))


def support_f1(pred_support, true_support):
    pred, true = set(pred_support), set(true_support)
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(true) if true else 0.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def regression_recall(delta_hat, delta_true, tau, weights=None):
    """Weighted recall of materially regressed slices (Delta c < -tau)."""
    n = len(delta_true)
    w = np.ones(n) if weights is None else np.asarray(weights)
    regressed = delta_true < -tau
    if not regressed.any():
        return 1.0
    detected = delta_hat < -tau / 2          # detection threshold at tau/2
    num = float(np.sum(w * detected * regressed))
    den = float(np.sum(w * regressed))
    return num / den
