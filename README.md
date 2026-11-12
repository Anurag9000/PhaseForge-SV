# PhaseForge-SV

**PyTorch-native research platform for phase-derived text-dependent speaker verification.**

The project implements GMM-UBM speaker verification with controlled feature
ablation, covariance/EM studies, MAP adaptation, score normalization,
temporal verification models, phase-template DTW, learned background-only
phase encoders, hyperparameter search, diagnostics, and reproducible reports.

## Two isolated data tracks

- **Kaggle competition:** fit the UBM, PCA/ICA, learned phase encoders, and
  score cohorts using only recordings listed in Kaggle background.csv.
  Adapt speakers only from the corresponding dev/eval enrolment manifests.
  The required baseline outputs a per-trial average speaker-versus-UBM
  log-likelihood ratio. Google original recordings are prohibited as
  additional Kaggle training data.
- **Full Google Speech Commands v0.02:** research-only speaker-disjoint
  verification, independent stop-only/all-word UBM comparison, and overlap
  audits. Never export a Google-trained model as a Kaggle submission.

## Install and validate

    python -m pip install -e ".[dev,plots]"
    python -m pytest -q
    python -m phaseforge.cli audit-kaggle --data /data/kaggle --deep --crc

The supplied Kaggle data directory needs its six CSV manifests plus audio.zip
(or the extracted audio/ directory). The deep audit verifies every referenced
PCM WAV, archive structure, duplicate PCM hashes and optional ZIP CRC.

## Reproduce the baseline and compare methods

    python -m phaseforge.cli run --data /data/kaggle --config configs/kaggle_group.json --output runs/group
    python -m phaseforge.cli run --data /data/kaggle --config configs/kaggle_temporal.json --output runs/temporal
    python -m phaseforge.cli fuse --runs runs/group runs/temporal --output runs/fusion

    python -m phaseforge.cli proxy --data /data/kaggle --config configs/kaggle_group.json --output runs/proxy
    python -m phaseforge.cli gmm-study --data /data/kaggle --config configs/kaggle_group.json --output runs/gmm_study --seeds 20

    python -m phaseforge.cli sweep --data /data/kaggle --config configs/kaggle_normalization_sweep.json --output runs/normalization
    python -m phaseforge.cli optimize --data /data/kaggle --config configs/optimize_robust.json --output runs/robust_optimization

    python -m phaseforge.cli run-alt --method hmm --data /data/kaggle --config configs/kaggle_group.json --output runs/hmm
    python -m phaseforge.cli run-alt --method dtw --data /data/kaggle --config configs/kaggle_group.json --output runs/dtw

Other complete configurations include speaker-balanced UBM training,
supervised/background-identity phase-CNN encoding and self-supervised
contrastive phase-CNN encoding in configs/. All are trained from scratch
on competition-permitted background recordings.

## Diagnostics and reproducible reporting

    python -m phaseforge.cli analyze --scores runs/group/development_scores.csv --output runs/group_analysis
    python -m phaseforge.cli robustness --data /data/kaggle --run runs/group --output runs/group_robustness
    python -m phaseforge.cli report --runs runs/group runs/temporal --output runs/comparison --gmm-study runs/gmm_study/gmm_behavior.json

The saved run JSON includes complete config, seeds, EM history,
covariance/component statistics, device and code/dataset provenance.
A report contains ROC curves, development EER/AUC comparisons,
speaker-condition analysis and paired grouped-bootstrap intervals.
Real training outputs are not supplied or claimed until a run completes.

## Finalize a Kaggle-only submission

    python -m phaseforge.cli finalize --data /data/kaggle --runs runs/group --output runs/final
    python -m phaseforge.cli finalize --data /data/kaggle --runs runs/group runs/temporal --output runs/fused_final

The finalizer refuses Google-trained and research-only models and checks
trial IDs, data origin and finite scores before writing submission.csv.
The original Google corpus can be evaluated only in the separate research track.

## GPU and Google corpus

The model trainer automatically uses CUDA when available. A manual
Kaggle-only GitHub Actions workflow requires repository secrets
KAGGLE_USERNAME and KAGGLE_KEY, or use the self-hosted GPU workflow.
**Never commit raw audio or account credentials to this public repository.**

For the six original Google archive parts, see
scripts/reassemble_google.sh and scripts/run_google_research.sh.
Keep that corpus outside the Kaggle data path and do not transfer its
trained weights into competition submissions.

## Status

GitHub Actions validates the code using synthetic-audio unit/integration
tests. **Passing CI is not evidence of improved real-data EER.**
The previously reported 4.61% development EER is a historical result
from another notebook; this repository has not yet reproduced or
superseded that measurement on the real Kaggle corpus.

Read [protocol](docs/PROTOCOL.md),
[research experiment program](docs/EXPERIMENTS.md),
[runbook](docs/RUNBOOK.md), and
[references](docs/REFERENCES.md).
