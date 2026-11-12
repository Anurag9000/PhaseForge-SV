# Running PhaseForge experiments

Run all commands in a local clone or a provisioned GPU server, with only
competition data for Kaggle experiments.

## Single-run commands

    pip install -e ".[dev]"
    pytest -q
    python -m phaseforge.cli audit-kaggle --data /data/kaggle
    python -m phaseforge.cli run --data /data/kaggle --config configs/kaggle_group.json --output runs/baseline
    python -m phaseforge.cli proxy --data /data/kaggle --config configs/kaggle_group.json --output runs/proxy
    python -m phaseforge.cli gmm-study --data /data/kaggle --config configs/kaggle_group.json --output runs/behavior --seeds 20
    python -m phaseforge.cli sweep --data /data/kaggle --config configs/kaggle_normalization_sweep.json --output runs/normalization
    python -m phaseforge.cli run-alt --method hmm --data /data/kaggle --config configs/kaggle_group.json --output runs/hmm
    bash scripts/run_staged_suite.sh /data/kaggle runs/staged

## GitHub Actions

Workflow **Kaggle-only experiments (manual)** downloads only the Kaggle
competition files. Set repository Actions secrets KAGGLE_USERNAME and
KAGGLE_KEY to a valid account with competition access, then run the workflow
manually from the Actions tab. Do not publish Kaggle secrets or upload raw
competition audio to Git. GitHub-hosted ubuntu-latest runners provide CPU only;
set up a trusted GPU runner before expecting GPU speed.

The workflow uploads result metadata, development-score tables and candidate
submission files as Actions artifacts. Because the repo is public, evaluate
accessibility of these artifacts before sharing private experimental outcomes.

## Research-only Google corpus

Reassemble the six original archive parts on a capable local/server machine.
Run google inventory and disjoint speaker benchmark creation, then
--track google. Never pass Google data to the Kaggle competition trainer.

## Honest results

GitHub CI passing is evidence of synthetic and end-to-end smoke tests,
not evidence of real-world EER improvement. Only a complete dataset run
with saved prediction CSV, configuration, seed and train/eval provenance
establishes a new measured benchmark.
