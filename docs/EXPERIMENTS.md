# Exhaustive PhaseForge experiment agenda and implementation ledger

Experiment families: (1) source/duplicate audit and preprocessing/VAD;
(2) phase only group/temporal/CosPhase/direct/IFD/LP residual/all-pole,
multiwindow and signed-power group cepstral representations;
(3) partial mean and variance normalization, masked band means,
PCA/ICA and temporal delta;
(4) GMM EM with diag/spherical/tied/full covariances, 4-256 mixtures,
random/kmeans initialization and 20+ seeds;
(5) MAP means-only baseline versus variance/weight adaptation,
relevance 0.5-64 and one/two/three enrolment utterances;
(6) per-frame LLR, top-C scoring, smoothing, Z/T/S cohort norms,
seed and phase-stream fusion;
(7) true left-right Baum-Welch HMM-UBM, time-binned GMM, and phase-DTW,
each as a separate comparator;
(8) speaker-disjoint background proxy and Google research,
FAR/FRR/EER/ROC, grouped bootstrap;
(9) stage-wise diagnose -> modify -> retrain -> validate -> save best,
with all failed trials in the ledger.

Implemented Python modules:
- data.py: strict Kaggle and original Google research data isolation
- features.py: phase-only 10-mode waveform feature extraction
- transforms.py: background-fitted PCA/ICA
- gmm.py: Torch EM, covariance studies, optional variance/weight MAP
- hmm.py: left-right forward-backward Baum-Welch and MAP emission adaptation
- temporal.py: time-binned GMM and phase-DTW
- scoring.py: likelihood ratios, cohort normalization and fusion
- metrics.py: EER, FAR/FRR, AUC and grouped bootstrap
- engine.py: Kaggle and Google full train/eval job orchestration
- studies.py: speaker-heldout background proxy, 20-seed GMM behavior study
- experiments.py: bounded/resumable search, score fusion

Not yet implemented or verified: complete Google archive benchmark run,
new actual Kaggle EER, neural phase encoders, cuVS/TorchGMM hardware benchmark,
all literature reproduction targets, exhaustive GPU search over every combination.
Run status must be taken from actual saved experiment artifacts and CI, not
from source-code implementation alone.

The classical magnitude-normalized modified group delay is not used as a
competition input. The signed-power clipped phase derivative is a phase-only
surrogate, labeled explicitly as such in the source.

Independent per-speaker scratch EM with 1/2/3 takes is implemented in `scratch.py`
and executable with `run-alt --method scratch`; its scores are research-only.

Bayesian/TPE hyperparameter optimization is available via optional Optuna
(`pip install -e '.[search]'`) and `phaseforge bayes --data ... --config
configs/bayesian_phase_search.json --output runs/bayesian`; validation defaults
to background-speaker proxy EER, with SQLite-resumable study results.


## Notebook reproduction and empirical job registry (added)

The `historical_461_group.json` and `historical_461_temporal.json`
configurations reproduce the earlier notebook's *intended* experiment
structure: 32/16 mixture GMMs, PCA 24/32, r=0.5/2, group/temporal phase,
background-only cohort seed 2026, five seeds, shared cohort tables averaged
**before** S-normalization, 12,000-frame k-means init and hard-example resets.

The `phaseforge_exhaustive_suite.json` registry defines 36 executable
jobs spanning phase features, transformations, GMM covariance/mixture
variants, MAP strategies, speaker-balanced sampling, temporal alternatives,
and learned phase CNN encoders. A source-fingerprinted ledger supports
staged runs and reexecution after changes. The suite never accesses eval
recordings while comparing models.

No new real-corpus EER is claimed until the relevant data-backed job
completes and saves its score tables, validation protocol and run manifest.
