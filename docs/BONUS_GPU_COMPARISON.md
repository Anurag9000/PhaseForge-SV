# GPU GMM-UBM and end-to-end speaker verification bonuses

This bonus program is separate from mandatory Kaggle phase-only PyTorch GMM-UBM/MAP. It is meant to produce a reproducible, detailed scientific comparison and a fast, unique phase-speaker verification pipeline. No GPU/cuVS performance values are claimed until measured on CUDA hardware.

## Source code checked

- CSOgroup TorchGMM: https://github.com/CSOgroup/torchgmm — PyTorch Lightning GaussianMixture.fit, estimator.model_ core forward log-likelihood, model covariance and mixture parameters. Its high-level score_samples is NEGATIVE log-likelihood; the core model returns log p(x). Benchmark pins torchgmm 0.1.4.
- NVIDIA cuVS C++ GMM: https://github.com/NVIDIA/cuvs/blob/main/cpp/include/cuvs/cluster/gmm.hpp — verified native C++ fit/score_samples, four covariance types, GPU buffers. Our adapter is under benchmarks/cuvs and needs a complete C++ cuVS/RAFT development SDK, compiler and CUDA toolkit. The Python cuVS wheel alone is not established as a GMM Python API.
- IBM Flash-GMM: https://github.com/IBM/Flash-GMM — a relevant GPU E-step-only comparator, not equivalent to full EM fit timing, not yet wired.

## Controlled scientific experiment design

**A. Identical parameter inference.** Train ONE native GMM and pass its exact priors, means and covariances to the TorchGMM core and cuVS C++ --score-only adapter. Compare framewise log p(x), max/mean absolute error, inference wall time and throughput. Supports float32 diagonal/spherical frozen models. This removes initialization/EM differences from the kernel comparison.

**B. Independently fitted EM.** Fit all enabled backends to EXACTLY the same data, K, D, covariance type, numerical precision, regularization, training/heldout frames and seeded inputs. Measure initialization-inclusive fit time, iteration count, convergence, heldout log-likelihood, numerical stability, GPU memory and subsequent MAP speaker-verification EER on Kaggle development. Initializers, minibatch/EM semantics and stopping rules differ; record them, do not claim parameter-equivalent training. Run multiple seeds. Synthetic stress cases cover well-separated, overlapping, near-singular, imbalanced, offset, outlier and multiple covariance cases. Real data comes exclusively from Kaggle background.csv for the UBM; development recordings only adapt claimed speaker models and determine EER.

**C. End-to-end pipeline.** 35 registered Kaggle-development-only ablations cover group/temporal/CosPhase frontend, 25/40 ms windows, VAD, CMVN, mixture capacity, EM random/kmeans/fused E-step, 1-3 enrolments, MAP relevance/all moments, S-norm and adaptive S-norm, smoothing/top-k, exact batched LLR scorer and multiple batched-STFT sizes. Pipeline stage timings include feature extraction, transform, background frame pooling, GMM fitting, MAP and verification scoring. The optimized frontend shares FFTs across equal-length utterances while preserving phase-only features and reference VAD. End-to-end EER and AUC are measured, not inferred from timing.

The isolated frontend benchmark preloads the SAME PCM waveforms and separately times batch sizes 1,4,8,16,32, with GPU synchronization, peak PyTorch CUDA allocation and max feature error. Do not call a warm-cache run an end-to-end cold-start speedup. GPU memory from the PyTorch allocator is not directly comparable to cuVS native allocations; NVML process sampling approximates cuVS memory peaks and may miss short spikes.

## Kakarot preparation

    cd ~/Projects/Untapped/PhaseForge-SV
    git pull --ff-only origin main
    source .venv/bin/activate
    python3 -m pip install -e '.[dev,bonus,plots,search]'
    python3 -m pytest -q
    nvidia-smi

    export KAGGLE_DATA="/absolute/path/to/kaggle"
    python3 -m phaseforge.cli audit-kaggle --data "$KAGGLE_DATA" --deep

## Easy execution (both bonus tracks)

    bash scripts/run_bonus_campaign.sh "$KAGGLE_DATA" runs/bonus smoke

    # Full CUDA campaign: 14 benchmark scenarios, 35 end-to-end ablations,
    # exact frozen-model comparison and phase frontend benchmark
    bash scripts/run_bonus_campaign.sh "$KAGGLE_DATA" runs/bonus full

    # Optional C++ cuVS SDK installed and compiled:
    bash scripts/build_bonus_cuvs.sh
    CUVS_BINARY="$PWD/build/bonus-cuvs/bonus_cuvs_gmm" \
      bash scripts/run_bonus_campaign.sh "$KAGGLE_DATA" runs/bonus full

The full benchmark includes up to ONE MILLION synthetic training frames and 3 repeats per case. First run the smoke and inspect disk/GPU memory. Use tmux for long jobs. Missing native cuVS libraries must not be reported as a successful comparison.

## Individual invocations for reproducibility

CPU native/TorchGMM smoke:

    python3 -m phaseforge.bonus_benchmark \
      --config configs/bonus_smoke.json \
      --output runs/bonus/smoke --backends native torchgmm \
      --device cpu --strict

Same-parameter log-likelihood check:

    python3 -m phaseforge.bonus_identical_scoring \
      --config configs/bonus_smoke.json \
      --output runs/bonus/identical_cpu \
      --backends torchgmm --device cpu

Full GPU independent-fit comparison without cuVS:

    python3 -m phaseforge.bonus_benchmark \
      --config configs/bonus_gpu_benchmark.json \
      --output runs/bonus/gpu_competitors \
      --backends native torchgmm \
      --device cuda --kaggle-data "$KAGGLE_DATA" --strict

Compiled cuVS extension (requires C++ SDK, not merely a pip wheel):

    bash scripts/build_bonus_cuvs.sh

    python3 -m phaseforge.bonus_benchmark \
      --config configs/bonus_gpu_benchmark.json \
      --output runs/bonus/gpu_cuvs \
      --backends native torchgmm cuvs --device cuda \
      --kaggle-data "$KAGGLE_DATA" \
      --cuvs-binary build/bonus-cuvs/bonus_cuvs_gmm --strict

Batched phase-only frontend:

    python3 -m phaseforge.bonus_frontend_benchmark \
      --data "$KAGGLE_DATA" --output runs/bonus/frontend \
      --files 128 --mode group --batches 1 4 8 16 32 \
      --repeats 5 --device cuda

Complete end-to-end training and ablations:

    python3 -m phaseforge.bonus_end_to_end ablation \
      --data "$KAGGLE_DATA" --config configs/bonus_e2e_suite.json \
      --output runs/bonus/e2e --max-jobs 35 --strict

    python3 -m phaseforge.bonus_end_to_end compare \
      --data "$KAGGLE_DATA" \
      --checkpoint runs/bonus/e2e/jobs/group_reference/model_seed_2026.pt \
      --output runs/bonus/score_comparison --device cuda --repeats 7

Generate measured Markdown report and figures:

    python3 -m phaseforge.bonus_report \
      --benchmark runs/bonus/gpu_competitors/bonus_results.json \
      --scoring runs/bonus/score_comparison/scoring_comparison.json \
      --ablations runs/bonus/e2e/bonus_e2e_summary.json \
      --output runs/bonus/BONUS_REPORT.md

    python3 -m phaseforge.bonus_plots \
      --benchmark runs/bonus/gpu_competitors/bonus_results.json \
      --scoring runs/bonus/score_comparison/scoring_comparison.json \
      --e2e runs/bonus/e2e/bonus_e2e_summary.json \
      --output runs/bonus/figures

Replace the benchmark file path with runs/bonus/gpu_cuvs/bonus_results.json when cuVS really ran. Reports use measured artifacts and mark missing values N/A.

## Evidence already measured in GitHub Actions (CPU only)

- Run https://github.com/Anurag9000/PhaseForge-SV/actions/runs/43583549542: native 0.00282 seconds versus TorchGMM 0.05748 seconds on one tiny CPU fit, 180 training frames, D=4, K=2, 3 configured iterations. This one-case timing is NOT a GPU or general superiority claim.
- Run https://github.com/Anurag9000/PhaseForge-SV/actions/runs/43584284763: identical-parameter TorchGMM versus native maximum log-likelihood discrepancy 2.6226e-5, diagonal float32 GMM. This is evidence of numerical agreement for this case only.

## Bonus report acceptance checklist

- Exact code SHA and source dataset manifest/hash, hardware, dependencies and flags.
- Independent seeds and repetitions with means/standard deviations and failures.
- Same fit/eval features, K/D/cov/regularization/seed/precision across backends.
- Synchronize CUDA events or wall-clock boundaries; disclose warmup/init/transfers.
- Exact-parameter likelihood parity and separate independently fitted-EM outcomes.
- Full dev-speaker MAP EER, AUC, FAR/FRR, convergence and numerical stress.
- GPU memory correctly labeled PyTorch allocator versus native NVML sampled.
- End-to-end feature, transform, fit, MAP, scoring timings, with cold/warm cache disclosed.
- Comprehensive VAD/CMVN/phase/MAP/GMM/scoring optimizations and their EER costs.
- Generated comparisons/figures and logs, not unverified claims. No Google weights,
  hidden eval labels or cross-claim postprocessing in the Kaggle competition path.

No bonus mark or top-50 rank can be guaranteed solely from a registered experiment. Real repeated CUDA results, a defensible optimization contribution and a reproducible submission are required.



## Optional fourth comparator: IBM Flash-GMM Triton E-step

Source: https://github.com/IBM/Flash-GMM. It provides the spherical-GMM
flash_gmm_estep kernel returning frame log-likelihoods, occupancies,
first moments and squared scatter around old means. **This is an E-step-only
comparison, NOT a third-party GMM-UBM fit or speaker-verification EER.**
It is separately implemented as bonus_flash_estep.py with no proprietary
or third-party kernel code copied into PhaseForge.

After installing Triton and its supported NVIDIA CUDA environment:

    git clone https://github.com/IBM/Flash-GMM /path/to/Flash-GMM

    python3 -m phaseforge.bonus_flash_estep \
      --config configs/bonus_flash_estep.json \
      --flash-source /path/to/Flash-GMM/flash_gmm.py \
      --output runs/bonus/flash_estep --repeats 5

The four cases cover 10,000–1,000,000 frames, K=32–1024 and
D=24–128. Both implementations see **the same initialized spherical GMM**
and same GPU tensors. This benchmark reports speed, peak PyTorch allocator,
and logZ/Nk/first-moment/scatter discrepancies. Flash's BF16/TF32/Triton
approximations may differ; the report exposes those discrepancies instead
of demanding an unsupported exact equality. Do not put an E-step-only
speedup into the full-EM or end-to-end performance column.

cuVS current installation guidance:
https://docs.nvidia.com/cuvs/installation — current source builds require
CUDA Toolkit 12.9+ and Volta-class GPU (compute capability 7.0+). A
complete cuVS native C++ library and headers are required for our adapter.
