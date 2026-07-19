"""Paired randomized sketches and measurement design (Secs. 3.1, 6).

Rows are sparse, signed, variance-normalized task weights. Measurements are
weighted sums of task-level paired mean differences (realizable composites),
with Neyman item allocation N_ij ~ |a_ij| sigma_j.
"""
import numpy as np


def design_matrix(m, n, row_density, sigma, rng, group_slices=None,
                  focus_groups=None, explore_fraction=0.15):
    """Sparse Rademacher design, variance-normalized (Sec. 6.1).

    If `focus_groups` is given (stage II), row support concentrates on the
    nominated groups' slices with an exploration floor elsewhere (Sec. 6.2).
    """
    A = np.zeros((m, n))
    if focus_groups is not None and group_slices is not None:
        focus_mask = np.zeros(n, dtype=bool)
        for g in focus_groups:
            focus_mask[group_slices[g]] = True
        p_focus = row_density
        p_off = row_density * explore_fraction
        probs = np.where(focus_mask, p_focus, p_off)
    else:
        probs = np.full(n, row_density)

    for i in range(m):
        s = rng.random(n) < probs
        if not s.any():
            s[rng.integers(n)] = True
        xi = rng.choice([-1.0, 1.0], size=n)
        row = np.where(s, xi / np.maximum(sigma, 1e-9), 0.0)
        norm = np.sqrt((row ** 2).sum())
        A[i] = row / max(norm, 1e-12)
    return A


def neyman_allocation(A_row, sigma, budget):
    """N_ij ~ |a_ij| sigma_j over the row support, integer allocation >= 2."""
    support = np.nonzero(A_row)[0]
    w = np.abs(A_row[support]) * sigma[support]
    w = w / w.sum()
    N = np.maximum(2, np.floor(budget * w).astype(int))
    return support, N


def paired_sketch(world, cand, A, sigma, items_per_row, rng):
    """Collect y_i = sum_j a_ij * mean paired difference on slice j.

    Returns (y, total_items, per_slice_means) where per_slice_means caches
    reusable slice estimates for residual monitoring.
    """
    m = A.shape[0]
    y = np.zeros(m)
    total_items = 0
    slice_sums = {}
    slice_counts = {}
    for i in range(m):
        support, N = neyman_allocation(A[i], sigma, items_per_row)
        acc = 0.0
        for j, nj in zip(support, N):
            diffs = world.paired_scores(cand, int(j), int(nj), rng=rng)
            mean_j = diffs.mean()
            acc += A[i, j] * mean_j
            total_items += int(nj)
            slice_sums[j] = slice_sums.get(j, 0.0) + diffs.sum()
            slice_counts[j] = slice_counts.get(j, 0) + int(nj)
        y[i] = acc
    per_slice = {j: slice_sums[j] / slice_counts[j] for j in slice_sums}
    return y, total_items, per_slice


def group_slice_map(dictionary, top_frac=0.25):
    """Map each dictionary group to the capability slices it loads on most,
    used to focus stage-II measurement rows."""
    n = dictionary.Psi.shape[0]
    out = []
    k = max(1, int(n * top_frac / max(1, len(dictionary.groups))))
    for g in dictionary.groups:
        load = np.abs(dictionary.Psi[:, g]).sum(axis=1)
        out.append(np.argsort(load)[::-1][:max(k, 8)])
    return out
