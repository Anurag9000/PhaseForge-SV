# New PhaseForge Research Suite — previously missing families

This suite is **separate** from the original 36-job GMM-UBM suite and is
created because the earlier repository omitted many published neural
speaker-verification, GMM-supervector/i-vector, keyword, and calibration ideas.

## Actual executable entrypoints

    python -m phaseforge.cli suite-new \
       --data /path/to/kaggle \
       --config configs/missing_research_suite.json \
       --output runs/missing_new --track kaggle --max-jobs 3

    python -m phaseforge.cli suite-new \
       --data /path/to/kaggle \
       --config configs/missing_research_suite.json \
       --output runs/missing_new --track kaggle --max-jobs 126

    python -m phaseforge.cli suite-new \
       --data /path/to/original-google \
       --benchmark /path/to/locked-google-benchmark \
       --config configs/missing_research_suite.json \
       --output runs/missing_google --track google --max-jobs 126

For quick smoke runs use:

    python -m phaseforge.cli suite-new \
       --data /path/to/kaggle \
       --config configs/new_missing_smoke.json \
       --output runs/missing_smoke --track kaggle

Individual runs:

    python -m phaseforge.cli run-neural --data /data/kaggle \
       --config configs/new_neural_base.json --output runs/neural_xvector

    python -m phaseforge.cli run-classical --data /data/kaggle \
       --config configs/new_classical_base.json --output runs/ivector

    python -m phaseforge.cli run-keyword --data /data/google \
       --benchmark /data/google_benchmark \
       --config configs/new_keyword_base.json --output runs/google_keyword

    python -m phaseforge.cli run-mathworks-google --data /data/google \
       --benchmark /data/google_benchmark \
       --config configs/new_google_mathworks_reference.json \
       --output runs/google_mathworks

    python -m phaseforge.cli calibrate \
       --development runs/model/development_scores.csv \
       --output runs/calibrated --folds 5

The suite writes new_suite_ledger.json, new_suite_summary.json, per-job
result.json, background-only checkpoints and labeled development scores.
Every job's manifest/config/data-source digest and Git SHA are tracked.
Failed jobs are written to failure.json. Repeat the same command/output
directory to resume; old revision/configuration outcomes are not reused.

## New architecture families (not present in original codebase)

- Braided CNN 1D phase adaptation: multi-branch 3/5 filters and cross-concat
- Braided CNN 2D phase adaptation: repeated bead topology, 3x3/5x5/1x1
  convolutions, pooling, crossed concatenations and preserved time-frequency
  alignment. 4-bead configuration included
- TDNN/x-vector frame context and attentive statistics
- ECAPA-style SE-Res2 dilated convolutions and multilevel aggregation
- Residual and SE-Residual 1D speaker encoders
- Depthwise-separable CNN and temporal-convolutional residual nets
- CRNN bidirectional GRU
- Phase-frame Keyword Transformer and Conformer
- Small gated selective state-space *ablation*, NOT actual full Mamba

All backbones accept phase-derived [time, phase-feature] frames and train
from random initialization on the authorized background population only.
No pretrained or external Google models enter the Kaggle research track.

## Training objectives

Cross-entropy, additive angular-margin softmax, leave-one-out GE2E,
supervised contrastive, batch-hard triplet, speaker prototypical,
Google end-to-end pairwise speaker verification *inspired* BCE,
BYOL-like EMA latent prediction and JEPA-like masked-latent prediction.
These last two are exploratory adaptations, NOT claimed exact BYOL/JEPA
reproductions. Each trained model has background-identity-separated dev
verification with cosine, quality-cosine, or learned pair scores.

## Added classical statistical and scoring backends

- UBM Baum-Welch sufficient statistics, mean MAP supervector and whitening
- Actual total-variability matrix EM with posterior i-vectors
- Speaker-label LDA, regularized two-covariance Gaussian PLDA score
- Supervector pairwise linear hinge SVM
- Stop/passphrase UBM versus all-word competing background (Google only)
- Adaptive top-k S-normalization with background-only cohorts
- Quality-aware embedding scores (enrolment dispersion)
- Development-only grouped out-of-fold logistic calibration
- Minimum normalized decision cost, Cllr, EER/AUC and paired comparisons

## MathWorks and Google published-reference tracks

A *Kaggle-eligible phase-only methodological analogue* uses 40ms windows
with 10ms hop, peak normalization, longest speech-region energy detection,
background-fitted global normalization, 32-mixture UBM, MAP r=16 on means,
variances and weights, and 3-frame median-smoothed LLR.

The **original-Google-only reference** separately uses magnitude-derived
log-mel/MFCC features to compare the MathWorks method, and log-mel/MFCC
keyword inputs for the BCNN/Transformer/DS-CNN/TC-ResNet architectures.
The MFCC reference is not an exact MATLAB replication: speech-region
detection is energy based and extraction may differ numerically from
MATLAB's built-in MFCC/filterbank implementation. This distinction is
explicit in source and results.

Additional phase-front-end experiments: phase unwrapping, temporal phase
flux, phase-only IFCC/CNPCC/APCC surrogates. These are explicitly
**surrogates**, not exact published complex-domain cepstral algorithms.

## Three distinct evaluation protocols — NEVER MIX METRICS

**Kaggle phase-only:** all trained parameters and UBM statistics from
competition background.csv/audio.zip. Dev target adaptation and trials
are allowed only in their designated roles. Neural and PLDA work here
is RESEARCH ONLY; the mandatory Kaggle GMM-UBM submission finalizer
will not accept these results.

**Original Google speaker research:** UBM/neural speaker training on the
original corpus's background speakers; compare on original dev speakers,
then evaluate the locked original test **only after selecting a model**.
All-word/stop-specific models and i-vector/PLDA belong here as separate
research experiments. No Google-trained model is imported into Kaggle.

**Original Google keyword research:** word-classification accuracy/macro-F1
using original speaker-disjoint splits. This is not speaker verification
and its accuracy cannot be compared numerically to an EER.

No output model is declared better than historical 4.61% without a
completed real-data run saved in this repository.

## Implementation boundaries and non-equivalences

- 2D braided phase architecture follows the BCNN topological idea but is
  not a byte/parameter-identical reproduction of the original mel model.
- KWT is a phase-frame Transformer adaptation, not an exact published
  patch-token KWT reconstruction.
- ECAPA uses essential SE-Res2/attention-pooling ideas but not official
  pretrained ECAPA weights or exact training schedule.
- The gated selective state-space block is explicitly \`ssm_lite\`,
  not the official Mamba implementation.
- Google paired verification is objective-inspired; GE2E is a complete
  leave-one-out centroid objective implemented from scratch.
- Original-Google source data overlaps ancestry with the competition;
  it MUST NOT be used for Kaggle speaker model fitting or data leakage.

## Background references

- Sinha, Awasthi, Ajmera (2020), "Audio classification using braided
  convolutional neural networks":
  https://doi.org/10.1049/iet-spr.2019.0381
- MathWorks: "Speaker Verification Using Gaussian Mixture Model":
  https://www.mathworks.com/help/audio/ug/speaker-verification-using-gaussian-mixture-model.html
- Google GE2E:
  https://google.github.io/speaker-id/publications/GE2E/
- Google end-to-end text-dependent speaker verification:
  https://research.google/pubs/end-to-end-text-dependent-speaker-verification/
- ECAPA-TDNN, original paper and common implementation:
  https://arxiv.org/abs/2005.07143
- Keyword Transformer:
  https://arxiv.org/abs/2104.00769
- i-vector total variability review and derivation:
  https://speechprocessingbook.aalto.fi/recognition/speaker-recognition-and-verification/

No external model code or weights were copied into this repository.
The architectures are independently authored PyTorch research implementations.


## Reliability audit and matched-seed research execution

Recent implementation fixes:

- Background caps now preserve complete 2/3-take speaker groups for GE2E,
  PLDA and SVM, and reserve original stop recordings in passphrase studies.
- Two-covariance PLDA joint regularization uses diagonal-only jitter.
- JEPA-like masked context is online-only; BYOL-like prediction is symmetric,
  EMA BatchNorm buffers are updated and latent-variance regularization added.
- Global/local background PCA uses bounded frame sampling before concatenation.
- The new suite uses executable-code/dataset hashes for resume rather than
  Git revision alone; source-model calibration dependencies are fingerprinted.
- Neural training saves an atomic per-epoch checkpoint with optimizer state,
  EMA teacher and CPU/CUDA/Python random-generator states. Interrupted jobs
  continue from the next completed epoch, unless source/config/data changes.
- The feature cache fingerprints Kaggle audio ZIP metadata or original Google
  WAV file metadata and phase extractor code. Cache saves are atomic and
  invalid/corrupted cache files are rebuilt.

### Execute the original new-method registry

    python3 -m pip install -e '.[dev,plots,search]'
    python3 -m pytest -q
    python3 -m phaseforge.cli suite-new --data /absolute/path/to/kaggle \
      --config configs/new_missing_smoke.json \
      --output runs/new_missing_smoke --track kaggle --strict
    bash scripts/run_missing_suite.sh /absolute/path/to/kaggle runs/new_missing 127

### Expand to 3 matched random seeds (381 registry jobs)

    python3 scripts/expand_missing_replicates.py \
      --source configs/missing_research_suite.json \
      --output configs/generated_missing_3seed.json \
      --seeds 2026 3107 4109

    python3 -m phaseforge.cli suite-new \
      --data /absolute/path/to/kaggle --track kaggle \
      --config configs/generated_missing_3seed.json \
      --output runs/new_missing_3seed --max-jobs 381

    python3 scripts/summarize_missing_replicates.py \
      --ledger runs/new_missing_3seed/new_suite_ledger.json \
      --output runs/new_missing_3seed/aggregate \
      --seeds 2026 3107 4109

381 is the registered total (127 x 3), not the number of Kaggle-valid jobs:
Google-only keyword and magnitude-reference jobs are skipped on Kaggle.
Resuming with the same output path reuses completed matching jobs.
The aggregation separates EER from keyword accuracy and displays missing
seed outcomes rather than assigning a misleading zero standard deviation.

The equivalent original-Google speaker-disjoint research suite uses
--track google and --benchmark with a locked benchmark directory. Do not
score its locked test population until a candidate is selected on dev.

CI uses synthetic data and numerical tests. It does NOT demonstrate
improvement over the prior 4.61% Kaggle development EER. Real-corpus
training and trial evaluation remain separate necessary steps.
