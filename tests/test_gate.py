"""E-process gate: type-I control under adversarial optional stopping,
and power under genuine non-inferiority."""
import numpy as np
from spectra_rsi.gate import EProcess


def test_type1_error_under_optional_stopping():
    """Adversary peeks after every observation and stops at the most
    favorable moment. False rejection of a TRUE null (mean == -tau, i.e.
    exactly materially regressed) must stay below alpha."""
    alpha, tau = 0.05, 0.05
    rng = np.random.default_rng(42)
    n_sims, horizon = 400, 2000
    false_rejects = 0
    for _ in range(n_sims):
        ep = EProcess(tau=tau, alpha=alpha, bound=1.0)
        rejected = False
        for _ in range(horizon // 50):
            obs = np.clip(rng.normal(-tau, 0.3, 50), -1, 1)  # H0 boundary
            ep.update(obs)
            if ep.rejects_null:          # adversarial peeking
                rejected = True
                break
        false_rejects += rejected
    rate = false_rejects / n_sims
    assert rate <= alpha + 2 * np.sqrt(alpha * (1 - alpha) / n_sims), rate


def test_power_under_true_improvement():
    """A genuinely improved anchor should be certified with high probability."""
    alpha, tau = 0.05, 0.02
    rng = np.random.default_rng(7)
    certified = 0
    n_sims = 100
    for _ in range(n_sims):
        ep = EProcess(tau=tau, alpha=alpha, bound=1.0)
        for _ in range(60):
            obs = np.clip(rng.normal(+0.08, 0.25, 50), -1, 1)
            ep.update(obs)
            if ep.rejects_null:
                break
        certified += ep.rejects_null
    assert certified / n_sims > 0.9
