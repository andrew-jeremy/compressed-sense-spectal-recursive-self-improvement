"""SPECTRA-RSI: Counterfactual Spectral Sketching and Anytime-Valid Gating
for Modular Recursive Self-Improvement.

Reference implementation of the architecture in the SPECTRA-RSI manuscript.
All components operate against a pluggable `World` interface; a synthetic
ground-truth world is provided for validation and benchmarking.
"""

from .config import SpectraConfig
from .world import SyntheticWorld, CandidateUpdate
from .experts import Router, expert_priors
from .probes import build_dictionary, Dictionary
from .sketch import design_matrix, paired_sketch, neyman_allocation
from .recovery import sparse_group_recover, debias, bootstrap_support
from .drift import procrustes_transport, principal_angles, ResidualDriftMonitor
from .gate import EProcess, AnchorGate, GateDecision
from .loop import SpectraRSILoop, IterationReport
from . import metrics

__version__ = "0.1.0"

__all__ = [
    "SpectraConfig", "SyntheticWorld", "CandidateUpdate",
    "Router", "expert_priors",
    "build_dictionary", "Dictionary",
    "design_matrix", "paired_sketch", "neyman_allocation",
    "sparse_group_recover", "debias", "bootstrap_support",
    "procrustes_transport", "principal_angles", "ResidualDriftMonitor",
    "EProcess", "AnchorGate", "GateDecision",
    "SpectraRSILoop", "IterationReport", "metrics",
]
