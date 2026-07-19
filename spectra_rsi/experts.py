"""Spectral mixture of low-rank experts: router statistics and support priors.

Implements Sec. 5: occupancy pi_e, gate entropy H_e, update energy g_e,
and the clipped prior activity score p_e with an exploration floor.
Router statistics are a PRIOR, never evidence of improvement.
"""
import numpy as np


class Router:
    """Tracks routing statistics over the candidate-generation corpus."""

    def __init__(self, n_experts, seed=0):
        self.E = n_experts
        self._rng = np.random.default_rng(seed)

    def occupancy_stats(self, cand_beta, rank_per_expert, corpus_size=2000,
                        fidelity=0.8):
        """Simulate router occupancy correlated (imperfectly) with which
        experts the candidate actually touched. `fidelity` < 1 injects router
        noise / gaming so the prior is fallible, as the paper requires."""
        R = rank_per_expert
        energy = np.array([np.linalg.norm(cand_beta[e * R:(e + 1) * R])
                           for e in range(self.E)])
        base = energy / (energy.max() + 1e-12) if energy.max() > 0 else energy
        noise = self._rng.random(self.E)
        occ = fidelity * base + (1 - fidelity) * noise
        occ = occ / (occ.sum() + 1e-12)
        # gate entropy per expert (high entropy = diffuse routing)
        p = np.clip(occ, 1e-9, 1)
        ent = -p * np.log(p)
        return occ, ent, energy


def expert_priors(occupancy, entropy, energy, clip=(0.05, 0.95),
                  a0=-1.0, a1=0.8, a2=1.2, a3=0.5):
    """Prior activity score p_e = sigma(a0 + a1 log occ + a2 g - a3 H),
    clipped to [p_min, p_max] to preserve exploration (Eq. prior)."""
    g = energy / (np.max(energy) + 1e-12) if np.max(energy) > 0 else energy
    z = (a0 + a1 * np.log(occupancy + 1e-6) + a2 * g - a3 * entropy)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, clip[0], clip[1])


def group_weights(priors, gamma=0.5, eps=1e-3):
    """w_e = (p_e + eps)^(-gamma): low-prior groups pay a higher penalty."""
    return (priors + eps) ** (-gamma)
