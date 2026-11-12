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
