# SPECTRA-RSI

Andrew Kiruluta, UC Berkeley, CA

Reference implementation of **SPECTRA-RSI: Counterfactual Spectral Sketching and
Anytime-Valid Gating for Modular Recursive Self-Improvement** (manuscript, July 2026).

The manuscript reframes recursive self-improvement as a measurement and control
problem: after a candidate update, estimate *which* capabilities changed, attribute
the change to *which* expert module caused it, and decide — with anytime-valid
statistical guarantees — whether the update may enter the next model generation.

This repo implements the full closed loop against a pluggable evaluation
interface, validated on a synthetic ground-truth world.

## Architecture → code map

| Manuscript component (section) | Module |
|---|---|
| Problem formulation, paired sketches (§3) | `spectra_rsi/world.py`, `spectra_rsi/sketch.py` |
| Counterfactual response dictionary, signed probes, trust-region pilot (§4) | `spectra_rsi/probes.py` |
| Dictionary transport (Procrustes), principal angles, residual drift confidence sequence (§4.3) | `spectra_rsi/drift.py` |
| Spectral mixture of low-rank experts, router priors p_e, exploration floor (§5) | `spectra_rsi/experts.py` |
| Measurement design: sparse Rademacher rows, variance normalization, two-stage focus, Neyman allocation (§6) | `spectra_rsi/sketch.py` |
| Weighted group-sparse recovery (FISTA), debiased refit, bootstrap support stability (§5.1, §8.1) | `spectra_rsi/recovery.py` |
| Anytime-valid non-inferiority gate: betting e-processes, Ville's inequality, Bonferroni risk budget, accept/rollback/quarantine (§8) | `spectra_rsi/gate.py` |
| Algorithm 1: the full iteration, discovery/acceptance firewall, hash-chained audit records (§9, §14) | `spectra_rsi/loop.py` |
| Primary metrics (§12.5) | `spectra_rsi/metrics.py` |

## Install & run

```bash
pip install -e ".[dev]"          # only dependency: numpy (pytest for tests)
pytest                            # validation suite
python experiments/run_demo.py    # full loop on 4 canonical interventions
python experiments/run_stopping_stress.py   # RQ3: e-process vs CLT peeking
```

## What the demo shows

`run_demo.py` builds a 400-slice, 16-expert synthetic world with a known
block-structured response operator, then runs four candidate types from the
manuscript's experimental protocol (§12.2):

- `single_gain` — one beneficial expert update → should be attributed to the
  right expert and **accepted** by the anchor gate;
- `single_regression` — a materially regressive expert → **rollback/quarantine**;
- `canceling_mixture` — two experts whose aggregate effect cancels (invisible
  to average-score evaluation) → both recovered with opposite signs;
- `broad_noncompressible` — violates the sparsity assumption → detected via
  pilot residual / support instability rather than force-fit.

Reported against ground truth: expert-support F1, normalized delta error,
regression recall, evaluated-item budget vs dense per-slice evaluation, and
gate decisions. Each iteration writes an immutable JSON audit record.

`run_stopping_stress.py` reproduces the paper's key statistical claim: under
adversarial peeking at a truly regressed anchor, naive repeated CLT testing
inflates type-I error several-fold, while the e-process gate stays below its
declared α at any stopping rule (Ville's inequality).

## Plugging in a real model

`SyntheticWorld` implements the only interface the loop uses:

- `paired_scores(candidate, slice_idx, n_items, rng)` — paired per-item score
  differences between base and candidate on one capability slice, ideally with
  common decoding randomness;
- `probe_delta(direction, magnitude, items_per_slice, rng)` — central-difference
  capability fingerprint of a signed low-rank expert probe;
- `sigma` — per-slice score-noise scales; `n` — number of slices.

Implement these against your evaluation harness (LoRA/MoE-LoRA adapters for
probe directions, benchmark slices for capabilities) and the rest of the loop —
dictionary, sketching, recovery, drift monitoring, gating, audit — carries over
unchanged.

## Honest limitations

This is a *reference implementation for the synthetic research benchmark*, not
a production evaluation system. The manuscript's empirical questions (RQ1–RQ6 - 
see reference here: [DOI: 10.13140/RG.2.2.20406.87360](https://doi.org)
concern real LLM update loops; nothing here claims those results. Known
simplifications: Bonferroni (not closed-testing) risk allocation; sub-Gaussian
mixture boundary for the drift monitor; router statistics simulated with a
fidelity knob rather than a trained router; no judge-model noise correlation.
