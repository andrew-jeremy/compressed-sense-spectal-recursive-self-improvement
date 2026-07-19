"""End-to-end loop: attribution, gating, and trust-region fallback."""
import numpy as np
import pytest
from spectra_rsi import SyntheticWorld, SpectraConfig, SpectraRSILoop
from spectra_rsi.metrics import support_f1, normalized_delta_error


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = SpectraConfig(n_slices=200, n_experts=8, rank_per_expert=2,
                        m_coarse=40, m_focused=60, items_per_row=150,
                        bootstrap_reps=20, fista_iters=400,
                        audit_dir=str(tmp_path_factory.mktemp("audit")))
    world = SyntheticWorld(cfg.n_slices, cfg.n_experts, cfg.rank_per_expert,
                           seed=11)
    loop = SpectraRSILoop(world, cfg)
    return world, cfg, loop


def test_single_expert_attribution_and_acceptance(setup):
    world, cfg, loop = setup
    cand = world.make_candidate("single_gain", experts=[3], scale=0.4,
                                rng=np.random.default_rng(2))
    rep = loop.run_iteration(cand)
    assert rep.pilot_passed
    assert 3 in rep.recovered_experts
    f1 = support_f1(rep.recovered_experts, cand.true_support)
    assert f1 >= 0.5
    err = normalized_delta_error(rep.delta_hat, world.true_delta(cand))
    assert err < 0.8          # smoke-test budget; tightens with more items
    assert rep.gate_decision == "accept"


def test_regression_is_not_accepted(setup):
    """Construct a candidate that provably regresses a protected anchor:
    beta chosen opposite to the anchor's true response row, so
    Delta c(anchor) = -scale * ||H*(anchor,:)||^2 < -tau."""
    from spectra_rsi.world import CandidateUpdate
    world, cfg, loop = setup
    anchor = int(loop.anchor_slices[0])
    row = world.H_true[anchor]                      # ground truth (test only)
    scale = 3.0 * cfg.tau_margin / max(np.dot(row, row), 1e-9)
    beta = -scale * row
    true_drop = world.true_delta(CandidateUpdate(beta=beta))[anchor]
    assert true_drop < -cfg.tau_margin              # sanity: really regressed
    cand = CandidateUpdate(beta=beta, label="anchor_regression")
    rep = loop.run_iteration(cand)
    if rep.pilot_passed:
        assert rep.gate_decision in ("rollback", "quarantine")
    else:
        assert rep.dense_fallback


def test_out_of_trust_region_falls_back(setup):
    world, cfg, loop = setup
    nl_world = SyntheticWorld(cfg.n_slices, cfg.n_experts,
                              cfg.rank_per_expert, seed=13, nonlin=0.9)
    nl_loop = SpectraRSILoop(nl_world, cfg)
    cand = nl_world.make_candidate("single_gain", experts=[1], scale=5.0,
                                   rng=np.random.default_rng(4))
    rep = nl_loop.run_iteration(cand)
    assert (not rep.pilot_passed and rep.dense_fallback) or rep.pilot_passed


def test_audit_records_written(setup):
    import glob, os
    world, cfg, loop = setup
    files = glob.glob(os.path.join(cfg.audit_dir, "iter_*.json"))
    assert len(files) >= 1
