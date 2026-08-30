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
- `paper_assets/`: generated LaTeX tables and vector figures used by the paper.
- `reproduce/verify_artifact.py`: archive and headline-result verifier.

## Scope and omitted material

Provider datasets, archived score arrays, fitted models, raw packet captures,
and machine-specific timing plans are not redistributed. They are excluded for
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

This draft contains no author names, local user paths, raw data, model binaries,
or repository history. Authors should confirm the venue's double-blind artifact
rules and select an explicit software/data license before public hosting. No
license is granted merely by distributing this review ZIP.
