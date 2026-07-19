"""Weighted sparse-group recovery (Eq. recovery) via FISTA, plus
debiased refit and bootstrap support stability.

  min_x 0.5 ||y - M x||^2 + lambda1 ||x||_1 + lambda_g sum_e w_e ||x_Ge||_2
"""
import numpy as np


def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _group_soft(x, groups, thresholds):
    out = x.copy()
    for g, t in zip(groups, thresholds):
        nrm = np.linalg.norm(x[g])
        out[g] = 0.0 if nrm <= t else (1 - t / nrm) * x[g]
    return out


def _lipschitz(M, iters=60, rng=None):
    rng = rng or np.random.default_rng(0)
    v = rng.normal(size=M.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = M.T @ (M @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-18:
            return 1.0
        v /= nv
    return nv


def sparse_group_recover(y, M, groups, group_w, lambda_l1, lambda_group,
                         n_iters=600):
    """FISTA with elementwise + group soft-thresholding prox."""
    p = M.shape[1]
    L = _lipschitz(M) + 1e-9
    step = 1.0 / L
    x = np.zeros(p)
    z = x.copy()
    t = 1.0
    for _ in range(n_iters):
        grad = M.T @ (M @ z - y)
        u = z - step * grad
        u = _soft(u, step * lambda_l1)
        u = _group_soft(u, groups, [step * lambda_group * w for w in group_w])
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = u + ((t - 1) / t_next) * (u - x)
        x, t = u, t_next
    return x


def debias(y, M, x_hat, tol=1e-8):
    """Least-squares refit restricted to the selected support (Sec. 5.1)."""
    support = np.nonzero(np.abs(x_hat) > tol)[0]
    if support.size == 0:
        return x_hat, support
    Ms = M[:, support]
    coef, *_ = np.linalg.lstsq(Ms, y, rcond=None)
    out = np.zeros_like(x_hat)
    out[support] = coef
    return out, support


def bootstrap_support(y, M, groups, group_w, lambda_l1, lambda_group,
                      reps=60, n_iters=250, rng=None):
    """Row-resampling bootstrap: per-group selection frequencies (Sec. 8.1)."""
    rng = rng or np.random.default_rng(0)
    m = len(y)
    freq = np.zeros(len(groups))
    for _ in range(reps):
        idx = rng.integers(0, m, m)
        xb = sparse_group_recover(y[idx], M[idx], groups, group_w,
                                  lambda_l1, lambda_group, n_iters=n_iters)
        for e, g in enumerate(groups):
            if np.linalg.norm(xb[g]) > 1e-8:
                freq[e] += 1
    return freq / reps


def active_groups(x_hat, groups, rel_tol=0.05):
    """Groups carrying more than rel_tol of the maximum group energy."""
    energies = np.array([np.linalg.norm(x_hat[g]) for g in groups])
    if energies.max() <= 0:
        return [], energies
    keep = np.nonzero(energies > rel_tol * energies.max())[0]
    return list(keep), energies
