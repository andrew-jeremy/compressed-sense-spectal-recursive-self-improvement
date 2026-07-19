"""Sparse-group recovery: exact-support recovery on clean synthetic data."""
import numpy as np
from spectra_rsi.recovery import sparse_group_recover, debias, active_groups


def test_group_recovery_identifies_support():
    rng = np.random.default_rng(0)
    p, n_groups = 32, 8
    groups = [np.arange(i * 4, (i + 1) * 4) for i in range(n_groups)]
    x_true = np.zeros(p)
    x_true[groups[2]] = rng.normal(0, 1, 4)
    x_true[groups[5]] = rng.normal(0, 1, 4)
    M = rng.normal(0, 1 / np.sqrt(60), (60, p))
    y = M @ x_true + 0.01 * rng.normal(size=60)
    gw = np.ones(n_groups)
    x_hat = sparse_group_recover(y, M, groups, gw, 1e-3, 5e-3, n_iters=800)
    x_deb, _ = debias(y, M, x_hat)
    act, _ = active_groups(x_deb, groups, rel_tol=0.1)
    assert set(act) == {2, 5}
    assert np.linalg.norm(x_deb - x_true) / np.linalg.norm(x_true) < 0.1
