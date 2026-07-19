"""Counterfactual spectral response dictionary (Sec. 4).

Signed central-difference probes on expert directions estimate the
interventional response matrix H_t; per-expert truncated SVD yields the
grouped dictionary Psi_t = [U_1, ..., U_E].
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Dictionary:
    Psi: np.ndarray            # (n, p) column-orthonormal within groups
    groups: list               # list of index arrays into columns of Psi
    H: np.ndarray              # (n, d) raw fingerprint matrix
    expert_maps: list          # per-expert (Sigma V^T) mapping dict coords -> beta coords

    @property
    def p(self):
        return self.Psi.shape[1]


def build_dictionary(world, cfg, rng=None) -> Dictionary:
    """Apply +/- h probes for every expert direction and factorize per expert."""
    rng = rng or np.random.default_rng(cfg.seed)
    E, R = cfg.n_experts, cfg.rank_per_expert
    d = E * R
    cols = []
    for idx in range(d):
        direction = np.zeros(d)
        direction[idx] = 1.0
        cols.append(world.probe_delta(direction, cfg.probe_magnitude,
                                      cfg.probe_items, rng=rng))
    H = np.stack(cols, axis=1)                      # (n, d)

    Psi_blocks, groups, expert_maps = [], [], []
    col0 = 0
    for e in range(E):
        He = H[:, e * R:(e + 1) * R]                # (n, R)
        U, S, Vt = np.linalg.svd(He, full_matrices=False)
        r = min(R, (S > 1e-8).sum())
        r = max(r, 1)
        Psi_blocks.append(U[:, :r])
        groups.append(np.arange(col0, col0 + r))
        expert_maps.append((S[:r], Vt[:r]))
        col0 += r
    Psi = np.concatenate(Psi_blocks, axis=1)
    return Dictionary(Psi=Psi, groups=groups, H=H, expert_maps=expert_maps)


def pilot_linearity_check(world, cand, dictionary, cfg, rng=None):
    """Trust-region pilot (Assumption 1): compare probe-superposition
    prediction H beta with a small paired evaluation on random slices.
    Returns (relative_residual, passed)."""
    rng = rng or np.random.default_rng(cfg.seed + 7)
    pred = dictionary.H @ cand.beta
    # test where the superposition prediction claims signal exists: the
    # top-|pred| slices; a linearity violation shows up exactly there.
    k = min(world.n, 40)
    idx = np.argsort(np.abs(pred))[::-1][:k]
    obs = np.array([world.paired_scores(cand, int(j), cfg.pilot_items,
                                        rng=rng).mean() for j in idx])
    denom = max(np.linalg.norm(obs), np.linalg.norm(pred[idx])) + 1e-9
    rel = np.linalg.norm(obs - pred[idx]) / denom
    return rel, rel <= cfg.trust_region_residual
