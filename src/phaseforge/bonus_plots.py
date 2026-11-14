"""Publication-style figures from PhaseForge measured bonus benchmark JSON.

Only draws numerical measurements already present in benchmark artifacts.
Unavailable cuVS (not compiled/no GPU) is not imputed. Requires the optional
Matplotlib installation, never used by the phase-only competition core.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .bonus_report import aggregate_bench


def render_figures(
    benchmark: dict, output: str | Path,
    e2e: dict | None = None, scoring: dict | None = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    entries = aggregate_bench(benchmark)
    backends = sorted({r["backend"] for r in entries})
    cases = sorted({r["case"] for r in entries})
    figs = []

    for measure, filename, ylabel in [
        ("fit_seconds", "fit_time_by_case.png", "EM fit wall time (seconds)"),
        ("score_seconds", "inference_time_by_case.png",
         "Held-out log-likelihood time (seconds)"),
        ("log_likelihood_mean", "heldout_log_likelihood.png",
         "Mean held-out log-likelihood per frame"),
        ("development_eer_median", "dev_eer_by_backend.png",
         "Development EER"),
    ]:
        valid = [r for r in entries if r.get(measure) is not None
                 and math.isfinite(r[measure])]
        if not valid:
            continue
        index = {case: i for i, case in enumerate(cases)}
        fig, ax = plt.subplots(figsize=(max(9, len(cases) * .65), 5))
        spread = .8 / max(1, len(backends))
        for j, backend in enumerate(backends):
            values = [r for r in valid if r["backend"] == backend]
            ax.bar(
                [index[r["case"]] - .4 + spread*(j+.5) for r in values],
                [r[measure] for r in values],
                width=spread*.9, label=backend,
            )
        ax.set_xticks(list(range(len(cases))))
        ax.set_xticklabels(cases, rotation=60, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"PhaseForge bonus — {ylabel}")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False)
        if measure in ("fit_seconds","score_seconds") and all(r[measure] > 0 for r in valid):
            ax.set_yscale("log")
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append(str(path))
    if e2e and e2e.get("development_ranked"):
        top = sorted(e2e["development_ranked"], key=lambda x:x["EER"])[:20]
        fig, ax = plt.subplots(figsize=(10, max(4, len(top)*.3)))
        ax.barh([r["id"] for r in top][::-1], [r["EER"] for r in top][::-1])
        ax.set_xlabel("Measured Kaggle development EER")
        ax.set_title("PhaseForge end-to-end development ablations")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        path = out / "e2e_development_eer.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append(str(path))
    if scoring and scoring.get("scoring"):
        modes = ["reference","batched"]
        medians = [
            scoring["scoring"][mode]["median_score_seconds"]
            for mode in modes
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(modes, medians)
        ax.set_ylabel("Median scoring wall time (s)")
        ax.set_title("Frozen UBM exact-Likelihood parity speed test")
        fig.tight_layout()
        path = out / "frozen_ubm_scoring.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append(str(path))
    return figs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--e2e")
    parser.add_argument("--scoring")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.benchmark).read_text())
    e2e = json.loads(Path(args.e2e).read_text()) if args.e2e else None
    scores = json.loads(Path(args.scoring).read_text()) if args.scoring else None
    print("\n".join(render_figures(report, args.output, e2e, scores)))


if __name__ == "__main__":
    main()
