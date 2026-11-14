# PhaseForge-SV

**An extensible, native-PyTorch research laboratory for phase-only, text-dependent speaker verification.**

Two independent tracks are enforced: **Kaggle competition** (background-only UBM, designated enrolment only, phase-derived features, per-frame GMM-UBM log likelihood ratios) and **Google Speech Commands v0.02 research** (speaker-disjoint benchmark, separate trained models and saved results). **Never train a Kaggle submission on the original Google archive.**

## Implemented model families

| Family | Available configurations |
|---|---|
| Phase representation | Group/relative-phase, temporal phase and IFD, CosPhase/direct phase, phase-group cepstra, signed-power phase-only group-delay surrogate, LP-residual phase, all-pole phase, multi-resolution |
| Signal processing | Energy VAD, masked circular aggregation, no/full/partial utterance CMVN, background-only global CMVN, PCA, whitened PCA, ICA, derivatives |
| GMM background models | Native torch EM, diagonal/spherical/tied/full covariance, initialization and occupancy studies, mixture-size/relevance sweeps, speaker-balanced or utterance-balanced sampling |
| Speaker models | Mean-only MAP (required baseline), optional variance/weight MAP, scratch GMM control, one/two/three enrolment recordings |
| Scoring | Mean framewise LLR, smoothing, top-component approximate LLR, Z/T/S background cohort normalization, multi-seed/multi-stream fusion |
| Alternatives | True left-to-right Baum-Welch HMM-UBM, time-binned GMM, phase-DTW; separately labeled experimental models |
| Learned features | Supervised background-speaker phase-CNN and background-only contrastive phase-CNN feeding the GMM-UBM |
| Experiments | Heldout background-speaker proxy, complete 20-seed GMM behavior study, finite search sweeps, stage-wise optimizer, optional Bayesian/TPE search, robustness perturbations |
| Reporting | ROC/AUC, interpolated EER, FAR/FRR, grouped bootstrap, per-speaker failures, enrolled-vs-unknown impostor comparison, saved configurations and model checkpoints |

## Install and test

    pip install -e '.[dev]'
    pytest -q

Install `pip install -e '.[search]'` only if using the optional Optuna Bayesian search. PyTorch and torchaudio must be installed with matching CPU or CUDA builds.

## Kaggle competition track

The dataset directory must contain `background.csv`, `dev_enrol.csv`, `dev_trials.csv`, `eval_enrol.csv`, `eval_trials.csv`, `sample_submission.csv`, and `audio.zip` (or an `audio/` directory).

    python -m phaseforge.cli audit-kaggle --data /data/kaggle
    python -m phaseforge.cli run --data /data/kaggle --config configs/kaggle_group.json --output runs/group
    python -m phaseforge.cli run --data /data/kaggle --config configs/kaggle_temporal.json --output runs/temporal
    python -m phaseforge.cli finalize --data /data/kaggle --runs runs/group runs/temporal --output runs/final

The `finalize` command verifies every constituent GMM run's Kaggle data origin, score/trial consistency, finite outputs and then creates a Kaggle-format `submission.csv` with a `provenance.json` audit. The exploratory HMM, DTW and four-claim-coupled scores are not silently eligible for this step.

## More experiments

    python -m phaseforge.cli proxy --data /data/kaggle --config configs/kaggle_group.json --output runs/proxy --rotations 4
    python -m phaseforge.cli gmm-study --data /data/kaggle --config configs/kaggle_group.json --output runs/gmm-study --seeds 20
    python -m phaseforge.cli sweep --data /data/kaggle --config configs/kaggle_normalization_sweep.json --output runs/norm
    python -m phaseforge.cli optimize --data /data/kaggle --config configs/optimize_stagewise.json --output runs/opt
    python -m phaseforge.cli bayes --data /data/kaggle --config configs/bayesian_phase_search.json --output runs/bayes
    python -m phaseforge.cli run-alt --method hmm --data /data/kaggle --config configs/kaggle_group.json --output runs/hmm
    python -m phaseforge.cli run-alt --method scratch --data /data/kaggle --config configs/kaggle_group.json --output runs/scratch
    python -m phaseforge.cli analyze --scores runs/group/development_scores.csv --output runs/analysis

Use `bash scripts/run_staged_suite.sh /data/kaggle runs/staged` for the predefined staged comparison.

## Full Google corpus research (strictly separate)

    bash scripts/reassemble_google.sh /data/google_parts /data/google_extracted
    bash scripts/run_google_research.sh /data/google_extracted runs/google

Original Google WAVs must never be committed. Exact hash mismatches do not guarantee Kaggle independence because competition files were renamed/lightly perturbed. Research results are **not** competition training data.

## Automated runners

- **Continuous integration:** `.github/workflows/ci.yml` runs synthetic torch/phase/speaker tests on each push.
- **Kaggle-only manual runs:** `.github/workflows/kaggle-manual-experiments.yml` downloads only competition data with Actions secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`; user must configure secrets and manually dispatch. The GitHub-hosted runner is CPU-only.
- **GPU manual runs:** `.github/workflows/gpu-experiments.yml` requires a self-hosted Linux runner with the `gpu` label and dataset paths on that machine. It never commits audio.

See `docs/RUNBOOK.md`, `docs/PROTOCOL.md`, `docs/EXPERIMENTS.md` and `docs/REFERENCES.md` for complete reproducibility and scientific context.

## Honest performance record

The historical **4.61% development EER** was measured in a previous separate notebook, **not** a new run completed by PhaseForge. Passing synthetic CI does not establish new competition EER. A result is verified only once the selected real-data run has completed with saved model, predictions and metrics. Hidden Kaggle evaluation labels are not available for independent local EER measurement.


## Historical 4.61% notebook reproduction (not yet measured in this repo)

The original 2026 notebook selected 32-component frequency/group-phase and
16-component temporal-phase UBMs, with PCA dimensions 24/32, MAP relevance
0.5/2, five GMM seeds and cohort S-normalization **after averaging seeds**.
It initialized k-means on at most 12,000 background frames and used
low-likelihood dead-component reinitialization. Separate compatible
configurations preserve these choices:

    bash scripts/reproduce_historical_best.sh /data/kaggle runs/historical

No claim of exactly 4.61% EER can be made until this implementation is run
on the original competition files and the resulting scores are audited.

## Complete heterogeneous experiment suite (36 registered jobs)

    python -m phaseforge.cli suite --data /data/kaggle --config configs/phaseforge_exhaustive_suite.json --output runs/registered --max-jobs 3
    python -m phaseforge.cli suite --data /data/kaggle --config configs/phaseforge_exhaustive_suite.json --output runs/registered --max-jobs 36

Each job uses only Kaggle background training and labelled development trials.
No evaluation recordings are scored during suite model selection. Failed jobs
and completed jobs have persistent ledgers; source revision and manifest
fingerprints prevent reusing stale outcomes. Development and speaker-excluded
proxy EER are reported in **separate** leaderboards.

The self-hosted GPU workflow requires a persistent absolute `results_path`
outside Git checkout; use the *same* path on subsequent dispatches to resume
the 36-job suite. Hosted GitHub CPU runners remain useful for small isolated
runs, not exhaustive GPU search.

## Public Google speech research

The manual `Original Google Speech Commands research` Actions workflow can
download the official full 2.3 GB corpus, inventory it and run the research
benchmark without Kaggle secrets. Its default CPU job caps background files
to reduce runtime; set 0 to include all research background files when
hardware capacity permits. This workflow **does not create a Kaggle model**.
The original corpus has a different repeated-take distribution from the
challenge; the 1-enrolment/1-heldout-take protocol must not be treated as the
same experiment as Kaggle's 3-enrolment verification.


## NEW — 126 previously missing-method research experiments

The original 36-job suite remains unchanged. The additional 126-job registry lives at
`configs/missing_research_suite.json`; see `docs/MISSING_RESEARCH_SUITE.md`.

Smoke test with the competition-only dataset:

    python3 -m phaseforge.cli suite-new --data /data/kaggle --track kaggle --config configs/new_missing_smoke.json --output runs/new_smoke

Full Kaggle-background-only research comparisons:

    bash scripts/run_missing_suite.sh /data/kaggle runs/new_missing 126

Separate original Google speaker-disjoint research:

    python3 -m phaseforge.cli suite-new --track google --data /data/google --benchmark /data/locked_google_benchmark --config configs/missing_research_suite.json --output runs/google_new --max-jobs 126

Additional models: braided BCNN 1D/2D, GE2E, ECAPA/Res2, TDNN/x-vector,
ResNet/SE-ResNet, DS-CNN, TC-ResNet, CRNN, KWT Transformer, Conformer,
gated SSM-lite, multiple supervised/self-supervised objectives, GMM
supervectors/i-vector/PLDA/SVM, passphrase backgrounds, adaptive S-norm,
quality-aware scores, minDCF/Cllr, grouped calibration, and MathWorks
phase-compatible ablations. Original MFCC/logmel references are Google-only.

All new neural, classical, keyword and calibrated models are research-only;
they do not produce eligible Kaggle GMM-UBM submissions. Registered jobs are
not completed training runs; real EER is unknown until actual execution.
