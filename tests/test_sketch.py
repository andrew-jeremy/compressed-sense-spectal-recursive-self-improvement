"""Paired sketch: unbiasedness and CRN variance reduction (Prop. 1)."""
import numpy as np
import pytest
from spectra_rsi import SyntheticWorld, SpectraConfig
from spectra_rsi.sketch import design_matrix, paired_sketch


@pytest.fixture
def world():
    return SyntheticWorld(n_slices=100, n_experts=4, rank_per_expert=2, seed=3)


def test_paired_sketch_unbiased(world):
    rng = np.random.default_rng(0)
    cand = world.make_candidate("single_gain", experts=[1], scale=0.5, rng=rng)
    A = design_matrix(20, world.n, 0.2, world.sigma, rng)
    true = A @ world.true_delta(cand)
    reps = 40
    est = np.zeros((reps, 20))
    for r in range(reps):
        y, _, _ = paired_sketch(world, cand, A, world.sigma, 150, rng)
        est[r] = y
    bias = np.abs(est.mean(axis=0) - true)
    se = est.std(axis=0) / np.sqrt(reps)
    # bias within 4 standard errors on at least 95% of rows
    assert np.mean(bias < 4 * se + 1e-3) > 0.9


def test_crn_variance_reduction():
    """Common random numbers must shrink paired-difference variance."""
    w_crn = SyntheticWorld(100, 4, 2, seed=5, cnr_rho=0.9)
    w_ind = SyntheticWorld(100, 4, 2, seed=5, cnr_rho=0.0)
    rng = np.random.default_rng(1)
    cand_c = w_crn.make_candidate("single_gain", experts=[0], rng=rng)
    cand_i = w_ind.make_candidate("single_gain", experts=[0],
                                  rng=np.random.default_rng(1))
    var_crn = np.var(w_crn.paired_scores(cand_c, 3, 5000))
    var_ind = np.var(w_ind.paired_scores(cand_i, 3, 5000))
    assert var_crn < 0.5 * var_ind
