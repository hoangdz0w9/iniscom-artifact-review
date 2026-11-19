# Anonymous reproducibility artifact

This archive accompanies the anonymous manuscript *Operating Policies Shape
Confidence-Routed Intrusion Detection*. It provides analysis, audit, threshold,
routing, bootstrap, timing, and asset-building code together with compact
machine-readable result summaries and the exact reported table/figure assets.

## Quick verification

Create a Python 3.12 environment, install `requirements.txt`, then run:

```bash
python reproduce/verify_artifact.py
python -m unittest discover -s tests -v
```

The first command checks every archived member against
`PUBLIC_ARTIFACT_HASHES.json`, scans text for local identity/path markers, and
validates the dimensions and selected headline values of the supplied summary
CSVs. Unit tests use synthetic or bundled compact inputs; they do not download
datasets, fit the manuscript models, or rerun timing.

## Contents

- `configs/`: frozen seeds, model definitions, split rules, and policy names.
- `scripts/`: training/benchmark, threshold, routing, sensitivity, review, and
  deterministic table/figure builders.
- `vendor/benchmark_helpers/`: preprocessing helper used by the instrumented
  runners.
- `tests/`: synthetic regression tests for analysis and integrity rules.
- `results/`: compact summary CSVs supporting the reported claims.
- `results/benchmark_timing/raw_calls.csv`: recorded timings of the accepted
  benchmark run, one row per timed call (30,800 rows: seed, regime, model,
  batch size, rows reaching each stage, elapsed seconds). These are timing
  measurements only; they contain no network-traffic records.
- `results/cost_model_diagnostic/`: outputs of the post-hoc cost-model
  diagnostic behind Table 17 (per-seed fitted coefficients, held-out errors and
  a manifest with input/output hashes).
- `paper_assets/`: generated LaTeX tables and vector figures used by the paper.
- `reproduce/verify_artifact.py`: archive and headline-result verifier.

## Reproducing Table 17 (post-hoc cost-model diagnostic)

```bash
python scripts/cost_model_weighting_diagnostic.py
```

Run from the artifact root, the script reads `results/benchmark_timing/raw_calls.csv`
and `results/benchmark_review/overhead_assessment.csv`, and writes to
`reproduced/` (the shipped files are not modified; `verify_artifact.py` ignores
this directory, so the two commands can be run in either order). It then:

1. refits the pre-specified model (fixed per-call cost plus per-row cost,
   nonnegative least squares on seconds, equal weight per calibration batch
   size) and stops unless it reproduces the archived verdict that all 12
   held-out assessments fail (Table 16);
2. refits the same model on the same calibration sizes with one change of
   objective, minimizing `sum_k ((a + b*n_k - T_k) / T_k)^2` over `a, b >= 0`
   (Eq. 7 of the paper), and re-applies the same held-out batch sizes 32 and
   1,000, route accounting and adequacy rule (MAPE <= 20%, max APE <= 50%);
3. compares the regenerated CSVs with `results/cost_model_diagnostic/` (values
   to a relative tolerance of 1e-9) and the regenerated LaTeX table with
   `paper_assets/tables/cost_model_weighting.tex` (byte for byte), and fails
   on any difference.

The relative-error refit was chosen after the held-out failure had been seen.
It diagnoses sensitivity to the fitting objective; it is not independent
validation of a latency predictor and does not imply that recorded times
transfer to other hosts or execution regimes. Rerunning re-analyses recorded
measurements; it does not re-measure timing.

## Scope and omitted material

Provider network-traffic datasets, archived score arrays, fitted models, raw
packet captures and the machine-specific batch plans (row-index files used to
schedule timed calls) are not redistributed; the recorded call timings above
are included. They are excluded for
licensing, size, privacy, and anonymous-review hygiene. Consequently this is a
code-and-results artifact, not a claim that every model can be refitted from
this ZIP alone. Runners expose explicit data/output arguments for authorized
local copies; no script installs packages or downloads data automatically.

Timing results describe repeated calls on one recorded host and are not
cross-hardware estimates. Oracle-prior policies use aggregate held-out label
prevalence and are diagnostics, not deployable unlabelled-target estimators.
Row-bootstrap intervals assume IID rows. The same-pool random control is
post-hoc and may share duplicate or collection context across partitions.

## Anonymous-release note

This archive contains no author names, local user paths, network-traffic data,
model binaries, or repository history. Authors should confirm the venue's double-blind artifact
rules and select an explicit software/data license before public hosting. No
license is granted merely by distributing this review ZIP.
