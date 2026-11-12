# PhaseForge-SV

**Phase-only speaker verification and reproducible GMM-UBM experiments.**

Two isolated research tracks:

- **Kaggle competition:** learn UBM, feature transformations and cohorts only from competition background recordings. Adapt speaker models only with designated enrolment data. Score per-trial likelihood ratios. Never load original Google corpus in this track.
- **Google Speech Commands v0.02:** speaker-disjoint research benchmark, explicit duplicate/overlap auditing, completely separate model artifacts. Not competition-eligible training data.

Implemented platform: phase-derived features; GMM-UBM EM/covariance studies; MAP; normalization; temporal verification; phase-DTW; multi-seed ensembles; reproducible sweeps, tests and experiment records.

## Quick start

    pip install -e '.[dev]'
    python -m pytest -q
    python -m phaseforge.cli audit-kaggle --data /path/to/kaggle
    python -m phaseforge.cli run --data /path/to/kaggle --config configs/kaggle_baseline.json --output runs/baseline
    python -m phaseforge.cli sweep --data /path/to/kaggle --config configs/kaggle_sweep.json --output runs/sweep

GitHub Actions tests the code with synthetic audio; real training requires a CPU/GPU runner with dataset access. No audio or original corpus files belong in Git.

The previously reported 4.61% development EER is a historical external baseline, not a result newly reproduced here.

See docs/PROTOCOL.md and docs/EXPERIMENTS.md.