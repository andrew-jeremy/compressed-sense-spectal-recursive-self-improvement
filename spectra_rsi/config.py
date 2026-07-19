"""Configuration for a SPECTRA-RSI loop instance."""
from dataclasses import dataclass, field


@dataclass
class SpectraConfig:
    # --- capability space ---
    n_slices: int = 400            # n: capability slices
    n_experts: int = 16            # E: experts / dictionary groups
    rank_per_expert: int = 2       # r_e: retained response rank per expert

    # --- probing (counterfactual dictionary) ---
    probe_magnitude: float = 0.4   # h: signed probe step (within trust region)
    probe_items: int = 400         # items per slice per probe evaluation

    # --- sketching ---
    m_coarse: int = 60             # stage-I sketch rows
    m_focused: int = 90            # stage-II sketch rows
    row_density: float = 0.10      # expected fraction of nonzero weights per row
    items_per_row: int = 200       # item budget per sketch row (Neyman-allocated)
    explore_fraction: float = 0.15 # measurement mass outside nominated groups

    # --- recovery ---
    lambda_l1: float = 2e-3
    lambda_group: float = 8e-3
    prior_gamma: float = 0.5       # w_e = (p_e + eps)^(-gamma)
    prior_clip: tuple = (0.05, 0.95)
    fista_iters: int = 600
    bootstrap_reps: int = 60

    # --- trust region / pilot ---
    pilot_items: int = 120
    trust_region_residual: float = 0.5    # relative pilot residual triggering fallback

    # --- drift ---
    drift_alpha: float = 0.05
    drift_window: int = 50

    # --- anytime-valid gate ---
    alpha_risk: float = 0.05       # total false-acceptance budget over critical anchors
    tau_margin: float = 0.02       # material-regression margin tau_j
    max_anchor_items: int = 24000  # total sequential anchor budget
    anchor_batch: int = 100

    # --- misc ---
    seed: int = 0
    audit_dir: str = "audit_logs"

    def __post_init__(self):
        assert 0 < self.row_density <= 1
        assert 0 <= self.explore_fraction < 1
