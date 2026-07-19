"""Synthetic ground-truth world.

Implements the paper's problem formulation (Sec. 3) with a known
capability-response operator so that reconstruction, attribution, and gating
can be validated against ground truth.

The world exposes exactly the interface a real evaluation harness would:
paired scoring of (base, candidate) on sampled items with common decoding
randomness. Everything downstream (probes, sketches, gate) only uses this
interface, so a real LLM harness can be dropped in behind it.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CandidateUpdate:
    """A frozen candidate: coefficients over expert low-rank directions."""
    beta: np.ndarray                 # (d,) coefficients over stacked expert directions
    label: str = "candidate"
    # ground-truth bookkeeping (for benchmarking only; never read by the loop)
    true_support: set = field(default_factory=set)   # expert indices actually perturbed


class SyntheticWorld:
    """Ground truth: Delta c = H* beta + nonlinearity + heteroscedastic noise.

    - H* (n x d): true interventional response matrix, block-structured so
      that each expert's directions load mainly on a band of slices (this is
      the group-compressibility assumption made explicit and controllable).
    - Paired scoring with common random numbers: base and candidate item
      scores share a noise component with correlation `cnr_rho`, so paired
      differences have variance reduced by (1 - rho) relative to independent
      evaluation (Prop. 1 in the paper).
    """

    def __init__(self, n_slices, n_experts, rank_per_expert, seed=0,
                 band_overlap=0.15, offband_leak=0.05, nonlin=0.0,
                 noise_scale=0.25, cnr_rho=0.85):
        self.n = n_slices
        self.E = n_experts
        self.R = rank_per_expert
        self.d = n_experts * rank_per_expert
        self.nonlin = nonlin
        self.cnr_rho = cnr_rho
        rng = np.random.default_rng(seed)
        self._rng = rng

        # block-structured true response matrix
        band = n_slices // n_experts
        H = np.zeros((self.n, self.d))
        for e in range(n_experts):
            lo, hi = e * band, (e + 1) * band
            # widen band with overlap
            pad = int(band * band_overlap)
            lo2, hi2 = max(0, lo - pad), min(self.n, hi + pad)
            for r in range(rank_per_expert):
                col = e * rank_per_expert + r
                v = np.zeros(self.n)
                # in-band responses are positive: pushing an expert "up"
                # improves the capabilities it owns (sign semantics), while
                # off-band leakage stays signed and diffuse.
                v[lo2:hi2] = np.abs(rng.normal(0, 1, hi2 - lo2))
                v += offband_leak * rng.normal(0, 1, self.n)
                H[:, col] = v / np.linalg.norm(v)
        self.H_true = H

        # per-slice score noise scale (heteroscedastic)
        self.sigma = noise_scale * (0.5 + rng.random(self.n))

        # current deployed parameter state in expert-coefficient coordinates
        self.theta_offset = np.zeros(self.d)

    # ------------------------------------------------------------------ #
    # ground truth (benchmark use only)
    # ------------------------------------------------------------------ #
    def true_delta(self, cand: CandidateUpdate) -> np.ndarray:
        lin = self.H_true @ cand.beta
        if self.nonlin > 0:
            # quadratic saturation: response bends for large updates
            lin = lin - self.nonlin * np.sign(lin) * lin ** 2
        return lin

    # ------------------------------------------------------------------ #
    # evaluation interface (what a real harness would implement)
    # ------------------------------------------------------------------ #
    def paired_scores(self, cand: CandidateUpdate, slice_idx: int, n_items: int,
                      rng=None) -> np.ndarray:
        """Paired per-item score differences d_l for one capability slice,
        evaluated with common random numbers (shared item + decoding noise)."""
        rng = rng or self._rng
        mu = self.true_delta(cand)[slice_idx]
        s = self.sigma[slice_idx]
        shared = rng.normal(0, s, n_items)
        e_base = np.sqrt(1 - self.cnr_rho) * rng.normal(0, s, n_items)
        e_cand = np.sqrt(1 - self.cnr_rho) * rng.normal(0, s, n_items)
        base = shared + e_base
        candv = mu + shared + e_cand
        return candv - base           # variance ~ 2 (1 - rho) s^2

    def probe_delta(self, direction: np.ndarray, magnitude: float,
                    n_items_per_slice: int, rng=None) -> np.ndarray:
        """Central-difference fingerprint (Eq. fingerprint): evaluate
        +h and -h probes on ALL slices with finite item budget."""
        rng = rng or self._rng
        plus = CandidateUpdate(beta=magnitude * direction)
        minus = CandidateUpdate(beta=-magnitude * direction)
        mu_p = self.true_delta(plus)
        mu_m = self.true_delta(minus)
        noise = (self.sigma / np.sqrt(max(1, n_items_per_slice))
                 * np.sqrt(2 * (1 - self.cnr_rho)))
        obs_p = mu_p + noise * rng.normal(0, 1, self.n)
        obs_m = mu_m + noise * rng.normal(0, 1, self.n)
        return (obs_p - obs_m) / (2 * magnitude)

    # ------------------------------------------------------------------ #
    # candidate factory helpers (benchmark interventions, Sec. 12.2)
    # ------------------------------------------------------------------ #
    def make_candidate(self, kind="single_gain", experts=None, scale=0.5,
                       rng=None) -> CandidateUpdate:
        rng = rng or self._rng
        beta = np.zeros(self.d)
        R = self.R
        if experts is None:
            experts = [int(rng.integers(self.E))]
        if kind == "single_gain":
            for e in experts:
                beta[e * R:(e + 1) * R] = scale * np.abs(rng.normal(0.8, 0.2, R))
        elif kind == "single_regression":
            for e in experts:
                beta[e * R:(e + 1) * R] = -scale * np.abs(rng.normal(0.8, 0.2, R))
        elif kind == "canceling_mixture":
            e1, e2 = experts[0], experts[-1]
            beta[e1 * R:(e1 + 1) * R] = scale
            beta[e2 * R:(e2 + 1) * R] = -scale
            experts = [e1, e2]
        elif kind == "broad_noncompressible":
            beta = scale * rng.normal(0, 0.3, self.d)
            experts = list(range(self.E))
        else:
            raise ValueError(kind)
        return CandidateUpdate(beta=beta, label=kind, true_support=set(experts))
