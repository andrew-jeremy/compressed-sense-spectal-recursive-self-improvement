"""Anytime-valid modular acceptance gate (Sec. 8).

Non-inferiority testing on protected anchor streams via betting e-processes:

  H0_j : Delta c_j <= -tau   vs.   H1_j : Delta c_j > -tau

For bounded paired differences d in [-B, B], the wealth process

  E_s = prod_i ( 1 + lam_i * (d_i + tau) ),   0 <= lam_i < 1 / (B + tau)

is a nonnegative supermartingale under every P in H0 (E_P[d] <= -tau implies
E_P[1 + lam (d + tau)] <= 1). By Ville's inequality,
P(sup_s E_s >= 1/alpha) <= alpha under H0, at ANY stopping rule.
A grid mixture over lam keeps validity while adapting to unknown effect size.
"""
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class GateDecision(Enum):
    ACCEPT = "accept"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"


class EProcess:
    """Grid-mixture betting e-process for H0: mean <= -tau, obs in [-B, B]."""

    def __init__(self, tau, alpha=0.05, bound=1.0, n_grid=12):
        self.tau = tau
        self.alpha = alpha
        self.B = bound
        lam_max = 0.5 / (bound + tau)     # stay safely inside admissible range
        self.grid = np.linspace(lam_max / n_grid, lam_max, n_grid)
        self.log_wealth = np.zeros(n_grid)
        self.n = 0

    def update(self, observations):
        obs = np.clip(np.atleast_1d(observations), -self.B, self.B)
        for d in obs:
            self.log_wealth += np.log1p(self.grid * (d + self.tau))
            self.n += 1
        return self.value()

    def value(self):
        # mixture (uniform prior over grid) — still an e-process
        m = self.log_wealth.max()
        return float(np.exp(m) * np.mean(np.exp(self.log_wealth - m)))

    @property
    def rejects_null(self):
        """True iff evidence of non-inferiority has crossed 1/alpha."""
        return self.value() >= 1.0 / self.alpha


@dataclass
class AnchorReport:
    anchor_id: int
    e_value: float
    n_items: int
    non_inferior: bool


class AnchorGate:
    """Per-expert + composed-candidate acceptance on protected anchors.

    Each critical anchor stream gets an independent e-process with a
    Bonferroni share of the total risk budget alpha. Acceptance requires
    every critical anchor to establish non-inferiority; budget exhaustion
    without rejection sends the expert to quarantine, not acceptance.
    """

    def __init__(self, anchor_slices, tau, alpha_total, bound=1.0):
        self.anchor_slices = list(anchor_slices)
        self.tau = tau
        self.alpha_each = alpha_total / max(1, len(self.anchor_slices))
        self.bound = bound

    def run(self, world, cand, batch=50, max_items=4000, rng=None):
        """Sequentially stream anchor items until every e-process resolves
        or budget runs out. Anytime-valid: any stopping rule is fine."""
        rng = rng or np.random.default_rng(0)
        procs = {j: EProcess(self.tau, self.alpha_each, self.bound)
                 for j in self.anchor_slices}
        undecided = set(self.anchor_slices)
        used = 0
        while undecided and used < max_items:
            for j in list(undecided):
                d = world.paired_scores(cand, j, batch, rng=rng)
                procs[j].update(d)
                used += batch
                if procs[j].rejects_null:
                    undecided.discard(j)
        reports = [AnchorReport(j, procs[j].value(), procs[j].n,
                                procs[j].rejects_null)
                   for j in self.anchor_slices]
        all_pass = all(r.non_inferior for r in reports)
        decision = GateDecision.ACCEPT if all_pass else (
            GateDecision.ROLLBACK if any(
                (not r.non_inferior) and r.n_items >= max_items // max(1, len(self.anchor_slices))
                for r in reports) else GateDecision.QUARANTINE)
        if not all_pass and used >= max_items:
            decision = GateDecision.QUARANTINE if any(
                not r.non_inferior and r.e_value > 1.0 for r in reports) \
                else GateDecision.ROLLBACK
        return decision, reports, used
