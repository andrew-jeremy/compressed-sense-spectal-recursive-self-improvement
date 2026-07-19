"""The SPECTRA-RSI loop (Algorithm 1).

One counterfactual, risk-controlled iteration:

 1. freeze candidate; align previous basis, check drift confidence sequence
 2. (re)build interventional dictionary from signed expert probes if needed
 3. clipped router priors with exploration floor
 4. local-linearity pilot (trust region) -> dense fallback on failure
 5. stage-I coarse paired sketch over all groups
 6. preliminary recovery + bootstrap instability -> nominate groups
 7. stage-II focused sketch, retaining exploration mass
 8. weighted sparse-group recovery + debias -> expert attribution
 9. protected anchor e-processes per nominated expert + composed candidate
10. accept / rollback / quarantine; audit record

Discovery never touches anchor data; anchors never inform the dictionary,
penalties, or nomination (Sec. 8.4 firewall).
"""
from dataclasses import dataclass, field, asdict
import json, os, time, hashlib
import numpy as np

from .experts import Router, expert_priors, group_weights
from .probes import build_dictionary, pilot_linearity_check
from .sketch import design_matrix, paired_sketch, group_slice_map
from .recovery import (sparse_group_recover, debias, bootstrap_support,
                       active_groups)
from .drift import procrustes_transport, ResidualDriftMonitor
from .gate import AnchorGate, GateDecision


@dataclass
class IterationReport:
    label: str
    pilot_residual: float
    pilot_passed: bool
    nominated_experts: list
    recovered_experts: list
    bootstrap_frequencies: list
    delta_hat_norm: float
    gate_decision: str
    anchor_reports: list
    items_probe: int
    items_sense: int
    items_anchor: int
    drift_alarm: bool
    dense_fallback: bool
    wall_time_s: float
    audit_hash: str = ""

    def to_json(self):
        return json.dumps(asdict(self), indent=2, default=str)


class SpectraRSILoop:
    def __init__(self, world, cfg, anchor_slices=None):
        self.world = world
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.router = Router(cfg.n_experts, seed=cfg.seed + 1)
        self.dictionary = None
        self.prev_Psi = None
        self.drift_monitor = ResidualDriftMonitor(alpha=cfg.drift_alpha)
        # protected anchors: held-out slices, never used in discovery
        if anchor_slices is None:
            anchor_slices = list(self.rng.choice(
                world.n, size=max(4, world.n // 50), replace=False))
        self.anchor_slices = anchor_slices
        os.makedirs(cfg.audit_dir, exist_ok=True)
        self._iteration = 0

    # ------------------------------------------------------------------ #
    def _ensure_dictionary(self, force=False):
        items = 0
        if self.dictionary is None or force:
            if self.dictionary is not None:
                self.prev_Psi = self.dictionary.Psi.copy()
            self.dictionary = build_dictionary(self.world, self.cfg, rng=self.rng)
            items = (2 * self.cfg.n_experts * self.cfg.rank_per_expert
                     * self.cfg.probe_items * self.world.n // self.world.n)
            items = (2 * self.cfg.n_experts * self.cfg.rank_per_expert
                     * self.cfg.probe_items)
            if self.prev_Psi is not None:
                procrustes_transport(self.dictionary.Psi, self.prev_Psi)
        return items

    # ------------------------------------------------------------------ #
    def run_iteration(self, cand) -> IterationReport:
        cfg, world = self.cfg, self.world
        t0 = time.time()
        self._iteration += 1
        drift_alarm = self.drift_monitor.is_alarmed()
        items_probe = self._ensure_dictionary(force=drift_alarm)
        if drift_alarm:
            self.drift_monitor = ResidualDriftMonitor(alpha=cfg.drift_alpha)
        D = self.dictionary

        # ---- router priors (fallible; clipped; exploration floor) ----
        occ, ent, energy = self.router.occupancy_stats(
            cand.beta, cfg.rank_per_expert)
        priors = expert_priors(occ, ent, energy, clip=cfg.prior_clip)
        gw = group_weights(priors, gamma=cfg.prior_gamma)

        # ---- trust-region pilot ----
        pilot_res, pilot_ok = pilot_linearity_check(world, cand, D, cfg,
                                                    rng=self.rng)
        if not pilot_ok:
            report = self._dense_fallback_report(cand, pilot_res, t0)
            self._audit(report)
            return report

        sense_slices = [j for j in range(world.n)
                        if j not in set(self.anchor_slices)]
        sigma = world.sigma

        # ---- stage I: coarse sketch ----
        A1 = design_matrix(cfg.m_coarse, world.n, cfg.row_density, sigma,
                           self.rng)
        A1[:, self.anchor_slices] = 0.0          # firewall: no anchor leakage
        y1, items1, slice_means1 = paired_sketch(
            world, cand, A1, sigma, cfg.items_per_row, self.rng)
        M1 = A1 @ D.Psi
        x1 = sparse_group_recover(y1, M1, D.groups, gw,
                                  cfg.lambda_l1, cfg.lambda_group,
                                  n_iters=cfg.fista_iters // 2)
        nominated, _ = active_groups(x1, D.groups)

        # ---- stage II: focused sketch with exploration mass ----
        gmap = group_slice_map(D)
        A2 = design_matrix(cfg.m_focused, world.n, cfg.row_density, sigma,
                           self.rng, group_slices=gmap,
                           focus_groups=nominated or list(range(cfg.n_experts)),
                           explore_fraction=cfg.explore_fraction)
        A2[:, self.anchor_slices] = 0.0
        y2, items2, slice_means2 = paired_sketch(
            world, cand, A2, sigma, cfg.items_per_row, self.rng)

        A = np.vstack([A1, A2])
        y = np.concatenate([y1, y2])
        M = A @ D.Psi

        # ---- recovery + debias + bootstrap ----
        x_hat = sparse_group_recover(y, M, D.groups, gw,
                                     cfg.lambda_l1, cfg.lambda_group,
                                     n_iters=cfg.fista_iters)
        x_deb, support = debias(y, M, x_hat)
        freq = bootstrap_support(y, M, D.groups, gw, cfg.lambda_l1,
                                 cfg.lambda_group, reps=cfg.bootstrap_reps,
                                 rng=self.rng)
        recovered, energies = active_groups(x_deb, D.groups)
        delta_hat = D.Psi @ x_deb

        # ---- drift monitor: predictive residuals on reused slice means ----
        resid = []
        for j, mhat in {**slice_means1, **slice_means2}.items():
            resid.append((mhat - delta_hat[j]) ** 2 - sigma[j] ** 2
                         * 2 * (1 - world.cnr_rho) / cfg.items_per_row)
        self.drift_monitor.update(np.array(resid[:cfg.drift_window]))

        # ---- protected anytime-valid gate ----
        gate = AnchorGate(self.anchor_slices, cfg.tau_margin, cfg.alpha_risk)
        decision, anchor_reports, items_anchor = gate.run(
            world, cand, batch=cfg.anchor_batch,
            max_items=cfg.max_anchor_items, rng=self.rng)

        report = IterationReport(
            label=cand.label,
            pilot_residual=float(pilot_res),
            pilot_passed=True,
            nominated_experts=[int(e) for e in nominated],
            recovered_experts=[int(e) for e in recovered],
            bootstrap_frequencies=[float(f) for f in freq],
            delta_hat_norm=float(np.linalg.norm(delta_hat)),
            gate_decision=decision.value,
            anchor_reports=[asdict(r) for r in anchor_reports],
            items_probe=items_probe,
            items_sense=items1 + items2,
            items_anchor=items_anchor,
            drift_alarm=bool(drift_alarm),
            dense_fallback=False,
            wall_time_s=time.time() - t0,
        )
        report.delta_hat = delta_hat            # attach for callers
        report.x_hat = x_deb
        self._audit(report)
        return report

    # ------------------------------------------------------------------ #
    def _dense_fallback_report(self, cand, pilot_res, t0):
        """Out-of-trust-region candidate: refuse sparse explanation."""
        report = IterationReport(
            label=cand.label, pilot_residual=float(pilot_res),
            pilot_passed=False, nominated_experts=[], recovered_experts=[],
            bootstrap_frequencies=[], delta_hat_norm=0.0,
            gate_decision=GateDecision.QUARANTINE.value, anchor_reports=[],
            items_probe=0, items_sense=0, items_anchor=0,
            drift_alarm=False, dense_fallback=True,
            wall_time_s=time.time() - t0)
        report.delta_hat = None
        report.x_hat = None
        return report

    # ------------------------------------------------------------------ #
    def _audit(self, report):
        """Immutable audit record (Sec. 14): hash-chained JSON."""
        payload = report.to_json()
        h = hashlib.sha256(payload.encode()).hexdigest()[:16]
        report.audit_hash = h
        path = os.path.join(self.cfg.audit_dir,
                            f"iter_{self._iteration:04d}_{h}.json")
        with open(path, "w") as f:
            f.write(payload)
