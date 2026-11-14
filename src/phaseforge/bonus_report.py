"""Generate an evidence-grounded comparison report from ACTUAL PhaseForge bonus outputs.

No fabricated speedups, EERs or memory numbers. Missing GPU backends are
listed as missing, not silently omitted. A numerical-stability failure
counts against the respective backend, and a lone seed is marked as N=1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def fmt(value, digits=5):
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    if not math.isfinite(value):
        return "NONFINITE"
    return f"{value:.{digits}g}"


def aggregate_bench(results: dict):
    groups = {}
    for row in results["runs"]:
        key = (row["case"], row["backend"])
        groups.setdefault(key, []).append(row)
    out = []
    for (case, backend), rows in sorted(groups.items()):
        complete = [r for r in rows if r.get("status") == "complete"]
        failed = [r for r in rows if r.get("status") == "failed"]
        skipped = [r for r in rows if r.get("status") == "skipped"]
        metric = {}
        for value in ("fit_seconds", "score_seconds", "log_likelihood_mean"):
            measured = [float(row[value]) for row in complete
                        if row.get(value) is not None]
            metric[value] = (
                statistics.median(measured) if measured else None
            )
            metric[f"{value}_std"] = (
                statistics.stdev(measured) if len(measured) > 1 else None
            )
        dev = [
            row["dev_verification"]["eer"] for row in complete
            if row.get("dev_verification")
        ]
        metric["development_eer_median"] = (
            statistics.median(dev) if dev else None
        )
        metric["development_eer_std"] = (
            statistics.stdev(dev) if len(dev) > 1 else None
        )
        gpu = [
            row["gpu_memory"]["allocated_peak_mib"]
            for row in complete if row.get("gpu_memory")
            and row["gpu_memory"].get("allocated_peak_mib") is not None
        ]
        metric["torch_cuda_allocated_peak_mib"] = (
            max(gpu) if gpu else None
        )
        finite = [
            int(r.get("sanity", {}).get("nonfinite_heldout_logprobs", 0) or 0)
            for r in complete
        ]
        out.append({
            "case": case, "backend": backend,
            "complete": len(complete), "failed": len(failed),
            "skipped": len(skipped),
            "nonfinite_score_total": sum(finite), **metric,
        })
    return out


def render_report(
    data: dict, scoring: dict | None = None, ablations: dict | None = None,
):
    rows = aggregate_bench(data)
    machine = data.get("environment", {})
    text = [
        "# PhaseForge GMM-UBM GPU performance and robustness bonus report",
        "",
        "This report is assembled entirely from measured JSON artifacts. "
        "A missing result is shown as N/A.",
        "",
        "## Reproduction environment",
        "",
        f"- Git commit: {machine.get('git_commit')}",
        f"- PyTorch: {machine.get('pytorch')}",
        f"- CUDA runtime: {machine.get('torch_cuda')}",
        f"- GPUs: {machine.get('gpus', 'Not available')}",
        f"- Platform: {machine.get('platform')}",
        f"- Dependency versions: {machine.get('packages')}",
        "",
        "## EM fit and held-out likelihood performance",
        "",
        "| Case | Backend | Completed / failed / skipped | "
        "Median fit (s) | Fit SD (s) | Median score (s) | "
        "Held-out log p/frame | Nonfinite frames | MAP development EER | "
        "Peak PyTorch GPU alloc (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            "| " + " | ".join((
                row["case"], row["backend"],
                f"{row['complete']} / {row['failed']} / {row['skipped']}",
                fmt(row["fit_seconds"]), fmt(row["fit_seconds_std"]),
                fmt(row["score_seconds"]), fmt(row["log_likelihood_mean"]),
                str(row["nonfinite_score_total"]),
                fmt(row["development_eer_median"]),
                fmt(row["torch_cuda_allocated_peak_mib"]),
            )) + " |"
        )
    text.extend([
        "",
        "### Matched-backend speedups",
        "",
        "| Case | Native/TorchGMM fit | Native/cuVS fit | "
        "Native/TorchGMM inference | Native/cuVS inference |",
        "|---|---:|---:|---:|---:|",
    ])
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["backend"]] = row
    for case, models in sorted(by_case.items()):
        def ratio(base, candidate, field):
            x = models.get(base, {}).get(field)
            y = models.get(candidate, {}).get(field)
            return (x / y if x is not None and y is not None and y > 0
                    else None)
        text.append(
            f"| {case} | {fmt(ratio('native','torchgmm','fit_seconds'))} "
            f"| {fmt(ratio('native','cuvs','fit_seconds'))} "
            f"| {fmt(ratio('native','torchgmm','score_seconds'))} "
            f"| {fmt(ratio('native','cuvs','score_seconds'))} |"
        )
    text.extend([
        "",
        "A ratio greater than one favors the competitor; below one favors "
        "native PhaseForge. Different initialization and convergence paths "
        "mean fitted-EM time ratios are *controlled but not identical-iteration* comparisons.",
        "",
        "## Scientific interpretation and limitations",
        "",
        "- All backends receive identical phase-feature matrices, component "
        "counts, covariance type, regularization, and seeded data.",
        "- The external implementations do **not** have exactly the same "
        "k-means, EM convergence, random initialization or minibatch behavior. "
        "Any model-performance change must be attributed cautiously.",
        "- The TorchGMM fit includes Lightning orchestration. Inference uses "
        "the fitted PyTorch model directly, excluding its high-level Trainer overhead.",
        "- NVIDIA cuVS is measured through a native C++ GPU process. The "
        "C++ output includes no verified peak GPU allocation measurement; "
        "end-of-run free-memory is not a peak.",
        "- The PhaseForge GPU memory column is PyTorch's allocator high-water "
        "mark; it excludes non-PyTorch CUDA allocations. Cross-backend GPU "
        "memory is therefore not fully equivalent without a separate NVML profiler.",
        "- Speaker verification uses common PhaseForge MAP and raw LLR "
        "scoring after importing each trained UBM's weights/means/covariance. "
        "The dev trials never influence UBM fitting.",
        "- EER is computed only when both genuine and impostor dev trials "
        "are available. Synthetic density benchmarks have no speaker-verification EER.",
        "- No hidden evaluation labels, external Google model weights or "
        "cross-claim score adjustments are used in the mandatory Kaggle track.",
        "",
    ])
    if scoring:
        text.extend([
            "## Exact scoring acceleration on frozen Kaggle speaker models",
            "",
            f"- Checkpoint SHA-256: {scoring.get('checkpoint_sha256')}",
            f"- Trial count: {scoring.get('n_dev_trials')}",
            f"- Maximum absolute score difference: "
            f"{fmt(scoring.get('max_absolute_score_difference'))}",
            f"- Exact numerical parity (tolerance gate): "
            f"{scoring.get('exact_parity_within_2e_4')}",
            f"- Reference median score seconds: "
            f"{fmt(scoring.get('scoring',{}).get('reference',{}).get('median_score_seconds'))}",
            f"- Batched median score seconds: "
            f"{fmt(scoring.get('scoring',{}).get('batched',{}).get('median_score_seconds'))}",
            f"- Measured scoring speedup: {fmt(scoring.get('measured_speedup'))}",
            f"- EER absolute difference: {fmt(scoring.get('eer_difference'))}",
            "",
        ])
    if ablations:
        text.extend([
            "## Full phase-verification pipeline ablations",
            "",
            "| Experiment | Dev EER | Dev AUC | End-to-end elapsed (s) |",
            "|---|---:|---:|---:|",
        ])
        for row in ablations.get("development_ranked", []):
            text.append(
                f"| {row['id']} | {fmt(row['EER'])} | {fmt(row['AUC'])} "
                f"| {fmt(row['elapsed_seconds'])} |"
            )
        text.extend([
            "",
            "End-to-end wall time includes source extraction and may benefit "
            "from warm phase caches. Compare individual stages or alternate "
            "execution order before claiming an algorithmic speedup.",
            "",
        ])
    text.extend([
        "## Evidence checklist",
        "",
        "- [ ] At least three independent seeds per comparison",
        "- [ ] Full native/TorchGMM/cuVS CUDA benchmark on identical GPU",
        "- [ ] Consistent fit iteration or convergence stopping rule disclosed",
        "- [ ] Full feature-extraction, EM, MAP, scoring and memory/latency profiling",
        "- [ ] Phase-only, background-trained Kaggle development verification EER/AUC",
        "- [ ] Numerical robustness: near singularity, outliers, imbalance and offset",
        "- [ ] Per-speaker independent uncertainty and paired trial analysis",
        "- [ ] Self-contained source commit, environment lock and runtime hardware report",
        "- [ ] Benchmark evidence linked, no unmeasured superiority claims",
    ])
    return "\n".join(text) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--scoring")
    parser.add_argument("--ablations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    benchmark = json.loads(Path(args.benchmark).read_text())
    scoring = json.loads(Path(args.scoring).read_text()) if args.scoring else None
    ablations = json.loads(Path(args.ablations).read_text()) if args.ablations else None
    text = render_report(benchmark, scoring, ablations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(f"Report written: {output}")


if __name__ == "__main__":
    main()
