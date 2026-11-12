"""Generate measured research reports from saved PhaseForge model runs."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import read_csv, write_csv
from .diagnostics import impostor_condition_report, paired_group_bootstrap
from .metrics import report, roc


def _load_run(location):
    directory = Path(location)
    metadata = json.loads((directory / "result.json").read_text())
    rows = read_csv(directory / "development_scores.csv")
    expected = {"trial_id", "claimed_id", "test_file", "label", "score"}
    if not rows or not expected.issubset(rows[0]):
        raise ValueError(f"Invalid development score CSV: {directory}")
    scores = torch.tensor([float(row["score"]) for row in rows], dtype=torch.float64)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.bool)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite score encountered")
    return dict(directory=directory, metadata=metadata, rows=rows,
                scores=scores, labels=labels, statistics=report(scores, labels),
                conditions=impostor_condition_report(rows, scores))


def _draw_plots(runs, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.figure(figsize=(7, 6))
    for item in runs:
        curve = roc(item["scores"], item["labels"])
        plt.plot(curve["far"].tolist(), curve["tpr"].tolist(),
                 label=f'{item["directory"].name}: EER {item["statistics"]["eer"]:.2%}')
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=.4)
    plt.xlabel("False acceptance rate")
    plt.ylabel("True acceptance rate")
    plt.title("Development ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "roc.png", dpi=160)
    plt.close()
    best = min(runs, key=lambda r: r["statistics"]["eer"])
    plt.figure(figsize=(7, 4))
    plt.hist(best["scores"][~best["labels"]].tolist(), bins=40, density=True,
             label="Impostor", alpha=.55)
    plt.hist(best["scores"][best["labels"]].tolist(), bins=30, density=True,
             label="Genuine", alpha=.55)
    plt.xlabel("Speaker log-likelihood ratio or cohort-normalized score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "score_histogram.png", dpi=160)
    plt.close()
    return ["roc.png", "score_histogram.png"]


def _study_plots(study_path, output):
    if not study_path:
        return []
    study = json.loads(Path(study_path).read_text())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    init = study.get("initialization_fits", [])
    if not init:
        return []
    plt.figure(figsize=(7, 4))
    for method in sorted({x["init"] for x in init}):
        ll = [x["training_ll"] for x in init if x["init"] == method]
        plt.hist(ll, bins=min(12, len(ll)), label=method, alpha=.5)
    plt.title("Twenty-seed Gaussian mixture initialization comparison")
    plt.xlabel("Average log-likelihood")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "gmm_initializations.png", dpi=160)
    plt.close()
    return ["gmm_initializations.png"]


def generate_report(directories, output, gmm_study=None, repetitions=250):
    if not directories or repetitions < 1:
        raise ValueError("Need at least one run and one bootstrap repetition")
    entries = [_load_run(d) for d in directories]
    ref = entries[0]
    identity = [(r["trial_id"], r["claimed_id"], r["test_file"], r["label"])
                for r in ref["rows"]]
    for item in entries[1:]:
        other = [(r["trial_id"], r["claimed_id"], r["test_file"], r["label"])
                 for r in item["rows"]]
        if other != identity:
            raise ValueError("Models have different development trials")
        for key in ("track", "source"):
            if item["metadata"].get(key) != ref["metadata"].get(key):
                raise ValueError("Comparing different data sources or evaluation tracks")
    entries.sort(key=lambda x: x["statistics"]["eer"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in entries:
        stats = item["statistics"]
        conditions = item["conditions"]
        rows.append({
            "model": item["directory"].name,
            "eer": stats["eer"], "auc": stats["auc"],
            "far": stats["far"], "frr": stats["frr"],
            "enrolled_impostor_eer": conditions.get("enrolled_impostor", {}).get("eer", ""),
            "unknown_impostor_eer": conditions.get("unknown_impostor", {}).get("eer", ""),
            "seeds": len(item["metadata"].get("config", {}).get("seeds", [])),
            "git_revision": item["metadata"].get("git_revision", ""),
        })
    write_csv(output / "model_comparison.csv", rows, list(rows[0]))
    best = entries[0]
    comparisons = [
        {
            "reference": item["directory"].name,
            "candidate": best["directory"].name,
            "recording_bootstrap": paired_group_bootstrap(
                ref["rows"], item["scores"], best["scores"], "test_file", repetitions
            ),
            "speaker_bootstrap": paired_group_bootstrap(
                ref["rows"], item["scores"], best["scores"], "claimed_id", repetitions
            ),
        }
        for item in entries[1:]
    ]
    figures = _draw_plots(entries, output) + _study_plots(gmm_study, output)
    result = dict(
        track=ref["metadata"].get("track"), source=ref["metadata"].get("source"),
        comparison=rows, paired_comparisons=comparisons, figures=figures,
        hidden_test_eer=None,
    )
    (output / "report.json").write_text(json.dumps(result, indent=2))
    markdown = [
        "# PhaseForge measured experiment report", "",
        "Results below come from saved and individually executed development trials.",
        "Private Kaggle EER cannot be inferred without private labels.", "",
        "| Model | Dev EER | AUC | Enrolled impostor EER | Unknown impostor EER |",
        "|---|---:|---:|---:|---:|",
    ]
    def percent(x):
        return "N/A" if x == "" else f"{x:.3%}"
    for item in rows:
        markdown.append(
            f'| {item["model"]} | {percent(item["eer"])} | {item["auc"]:.4f} '
            f'| {percent(item["enrolled_impostor_eer"])} '
            f'| {percent(item["unknown_impostor_eer"])} |'
        )
    markdown.extend(["", "Paired recording- and speaker-group bootstrap results are",
                     "saved in report.json. Development rankings are not private-test",
                     "guarantees.", ""])
    for filename in figures:
        markdown.append(f"![{filename}]({filename})")
    (output / "REPORT.md").write_text("\n".join(markdown) + "\n")
    return result
