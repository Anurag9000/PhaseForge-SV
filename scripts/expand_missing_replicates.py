"""Expand the independent missing-method registry into matched-seed repetitions.

Example:
python3 scripts/expand_missing_replicates.py \
    --source configs/missing_research_suite.json \
    --output configs/generated_missing_3seed.json \
    --seeds 2026 3107 4109

Every original model/ablation is copied with identical source data and
validation protocol but a fresh random seed. Calibration source dependencies
are rewritten to the SAME repetition's training job, not another seed's
scores. This generates jobs; it does not run any experiment.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def expand_replicates(registry: dict, seeds: list[int]) -> dict:
    if registry.get("suite_family") != "new_phase_research":
        raise ValueError("Input must be the missing-method research registry")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require unique nonempty seed list")
    if any(not 0 <= s < 2**31 for s in seeds):
        raise ValueError("Seeds must be nonnegative 31-bit integers")
    jobs = registry.get("jobs", [])
    if not jobs or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Input job registry is empty or contains duplicates")
    replicated = []
    for seed in seeds:
        ids = {job["id"]: f"{job['id']}__s{seed}" for job in jobs}
        for original in jobs:
            job = deepcopy(original)
            job["id"] = ids[original["id"]]
            job["replication_seed"] = seed
            overrides = job.setdefault("overrides", {})
            kind = job["kind"]
            if kind == "gmm_phase":
                overrides["seeds"] = [seed]
                # Shared background cohort sampling controlled separately;
                # use matched cohorts across independent GMM seeds.
                overrides.setdefault("score.cohort_seed", 2026)
            elif kind in ("neural_sv", "classical_sv", "keyword_google",
                          "mathworks_reference"):
                overrides["seed"] = seed
            elif kind == "calibrate":
                if original["source_job"] not in ids:
                    raise ValueError(
                        f"Unresolvable calibration dependency: {original['source_job']}"
                    )
                job["source_job"] = ids[original["source_job"]]
            else:
                raise ValueError(f"Unsupported job type {kind}")
            replicated.append(job)
    output = deepcopy(registry)
    output["description"] = (
        registry.get("description", "") + " "
        "MATCHED-SEED REPLICATIONS: source job IDs and calibration dependencies "
        "are scoped to the exact same seed. Scores must be grouped by base "
        "method when computing across-seed means and uncertainty."
    )
    output["replication_seeds"] = seeds
    output["base_registered_jobs"] = len(jobs)
    output["jobs"] = replicated
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2026, 3107, 4109])
    args = parser.parse_args()
    registry = json.loads(Path(args.source).read_text())
    expanded = expand_replicates(registry, args.seeds)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(expanded, indent=2) + "\n")
    print(
        f"Wrote {len(expanded['jobs'])} registered runs, "
        f"{len(args.seeds)} seeds x {expanded['base_registered_jobs']} jobs"
    )


if __name__ == "__main__":
    main()
