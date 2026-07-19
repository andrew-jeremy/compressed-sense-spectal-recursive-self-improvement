"""End-to-end SPECTRA-RSI demo on the synthetic ground-truth world.

Runs the four canonical candidate interventions from the manuscript's
experimental protocol (Sec. 12.2) through full loop iterations and reports
attribution, reconstruction, budget, and gate outcomes against ground truth.
"""
import numpy as np
from spectra_rsi import SyntheticWorld, SpectraConfig, SpectraRSILoop
from spectra_rsi.metrics import (support_f1, normalized_delta_error,
                                 regression_recall)


def main():
    cfg = SpectraConfig(n_slices=400, n_experts=16, rank_per_expert=2,
                        m_coarse=80, m_focused=120, items_per_row=600,
                        lambda_l1=8e-3, lambda_group=3e-2,
                        bootstrap_reps=30, audit_dir="audit_logs")
    world = SyntheticWorld(cfg.n_slices, cfg.n_experts, cfg.rank_per_expert,
                           seed=cfg.seed, offband_leak=0.01)
    loop = SpectraRSILoop(world, cfg)
    rng = np.random.default_rng(99)

    candidates = [
        world.make_candidate("single_gain", experts=[5], scale=0.4, rng=rng),
        world.make_candidate("single_regression", experts=[9], scale=0.6, rng=rng),
        world.make_candidate("canceling_mixture", experts=[2, 11], scale=0.5, rng=rng),
        world.make_candidate("broad_noncompressible", scale=0.4, rng=rng),
    ]

    dense_budget = cfg.n_slices * cfg.items_per_row  # dense per-slice reference
    print(f"{'candidate':<24}{'gate':<12}{'F1':>5}{'d-err':>7}"
          f"{'reg-recall':>11}{'sense items':>12}{'vs dense':>9}")
    print("-" * 80)
    for cand in candidates:
        rep = loop.run_iteration(cand)
        if rep.dense_fallback:
            print(f"{cand.label:<24}{'DENSE-FB':<12}{'--':>5}{'--':>7}"
                  f"{'--':>11}{'--':>12}{'--':>9}"
                  f"   (pilot residual {rep.pilot_residual:.2f})")
            continue
        truth = world.true_delta(cand)
        f1 = support_f1(rep.recovered_experts, cand.true_support)
        derr = normalized_delta_error(rep.delta_hat, truth)
        rrec = regression_recall(rep.delta_hat, truth, cfg.tau_margin)
        frac = (rep.items_sense) / dense_budget
        print(f"{cand.label:<24}{rep.gate_decision:<12}{f1:>5.2f}{derr:>7.2f}"
              f"{rrec:>11.2f}{rep.items_sense:>12,}{frac:>8.1%}")
        print(f"    true experts={sorted(cand.true_support)}  "
              f"recovered={sorted(rep.recovered_experts)}  "
              f"anchors used={rep.items_anchor:,}  "
              f"probe items={rep.items_probe:,}")
    print("\nAudit records written to ./audit_logs/")


if __name__ == "__main__":
    main()
