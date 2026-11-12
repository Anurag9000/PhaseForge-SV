"""End-to-end feature -> UBM -> MAP -> trial LLR -> metrics -> artifacts.

Training eligibility is determined by explicit per-track background manifests.
Nothing in the Kaggle code path loads Google corpus files.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from .data import GoogleResearchData, KaggleData, read_csv, write_csv
from .features import FeatureConfig, PhaseExtractor
from .gmm import EMConfig, MAPConfig, TorchGMM
from .metrics import grouped_bootstrap, report
from .scoring import ScoreConfig, cohort_normalize, enrolled_models, score_trials
from .transforms import fit_background_transform


def select_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but not available")
    return torch.device(preference)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class RunConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    em: EMConfig = field(default_factory=EMConfig)
    map: MAPConfig = field(default_factory=MAPConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    transform: str = "pca"
    dimensions: int = 24
    learned: dict = field(default_factory=dict)
    seeds: tuple[int, ...] = (2026, 3107, 4109, 5101, 6101)
    enrolment_size: int = 3
    max_frames_per_background_file: int = 32
    background_sampling: str = "per_utterance"  # or speaker_balanced
    max_fit_transform_frames: int = 50000
    max_background_files: int | None = None
    evaluate_eval: bool = True
    save_model: bool = True
    device: str = "auto"
    target_far: float = .01

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        f = dict(data.get("feature", {}))
        if "multi_modes" in f:
            f["multi_modes"] = tuple(f["multi_modes"])
        if "multi_windows" in f:
            f["multi_windows"] = tuple(f["multi_windows"])
        return cls(
            feature=FeatureConfig(**f), em=EMConfig(**data.get("em", {})),
            map=MAPConfig(**data.get("map", {})), score=ScoreConfig(**data.get("score", {})),
            transform=data.get("transform", "pca"), dimensions=int(data.get("dimensions", 24)),
            learned=dict(data.get("learned", {})),
            seeds=tuple(data.get("seeds", [2026, 3107, 4109, 5101, 6101])),
            enrolment_size=int(data.get("enrolment_size", 3)),
            max_frames_per_background_file=int(data.get("max_frames_per_background_file", 32)),
            background_sampling=data.get("background_sampling", "per_utterance"),
            max_fit_transform_frames=int(data.get("max_fit_transform_frames", 50000)),
            max_background_files=data.get("max_background_files"),
            evaluate_eval=bool(data.get("evaluate_eval", True)),
            save_model=bool(data.get("save_model", True)),
            device=data.get("device", "auto"), target_far=float(data.get("target_far", .01)),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        self.feature.validate()
        self.em.validate()
        self.map.validate()
        self.score.validate()
        if not self.seeds or not 1 <= self.enrolment_size <= 3:
            raise ValueError("At least one seed and 1-3 enrolment utterances required")
        if self.max_frames_per_background_file < 1:
            raise ValueError("Frame cap must be >=1")
        if self.background_sampling not in ("per_utterance", "speaker_balanced"):
            raise ValueError("Unknown background sampling strategy")


def _speaker_from_original_id(path: str) -> str:
    return Path(path).stem.split("_nohash_", 1)[0]


def load_research_manifests(benchmark: str | Path, pool: str = "stop") -> dict:
    benchmark = Path(benchmark)
    background = read_csv(benchmark / (
        "google_ubm_stop.csv" if pool == "stop" else "google_ubm_all_words.csv"
    ))
    for row in background:
        row["speaker_id"] = row.get("speaker_id") or _speaker_from_original_id(row["file_id"])
    output = {
        "background": background,
        "dev_enrol": read_csv(benchmark / "google_dev_enrol.csv"),
        "dev_trials": read_csv(benchmark / "google_dev_trials.csv"),
        "eval_enrol": read_csv(benchmark / "google_test_enrol.csv"),
        "eval_trials": read_csv(benchmark / "google_test_trials.csv"),
    }
    a = {x["speaker_id"] for x in background}
    b = {x["speaker_id"] for x in output["dev_enrol"]}
    c = {x["speaker_id"] for x in output["eval_enrol"]}
    if a & b or a & c or b & c:
        raise ValueError("Research speaker leakage: splits intersect")
    return output


def _cache_name(source: str, root: Path, ids: list[str], feature: FeatureConfig) -> str:
    spec = json.dumps({"source": source, "root": str(root), "ids": sorted(ids),
                       "feature": asdict(feature)}, sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:20]


def extract_features(
    data, ids: list[str], cfg: FeatureConfig, cache_root: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    """Raw phase extraction only. Learned transforms are fitted later on background IDs."""
    cache_root.mkdir(parents=True, exist_ok=True)
    name = _cache_name(data.source, data.root, ids, cfg)
    cache = cache_root / f"phase_{name}.pt"
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=True)
        if set(stored) == set(ids):
            return stored
    extractor = PhaseExtractor(cfg)
    features = {}
    with torch.no_grad():
        for index, ident in enumerate(sorted(set(ids))):
            y = data.load(ident).to(device)
            features[ident] = extractor(y).cpu()
            if index and index % 500 == 0:
                print(f"Extracted {index}/{len(ids)} phase utterances", flush=True)
    torch.save(features, cache)
    return features


def _background_frames(
    feature: dict[str, torch.Tensor], background_ids: list[str],
    max_per_file: int, device: torch.device,
    speaker_by_file: dict[str, str] | None = None,
    strategy: str = "per_utterance",
) -> torch.Tensor:
    """Control UBM training-frame weights without changing its phase representation.

    per_utterance: deterministic <=max_per_file frames for each recording.
    speaker_balanced: equal total sampled frames per speaker, keeping the
    approximate total training budget fixed (rare speakers may be resampled).
    """
    if strategy not in ("per_utterance", "speaker_balanced"):
        raise ValueError("Unknown background sampling policy")
    if strategy == "speaker_balanced":
        if not speaker_by_file:
            raise ValueError("Speaker-balanced sampling requires file speaker IDs")
        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
        for ident in background_ids:
            grouped[speaker_by_file[ident]].append(feature[ident])
        budget = len(background_ids) * max_per_file
        quota = max(1, (budget + len(grouped) - 1) // len(grouped))
        parts = []
        for speaker in sorted(grouped):
            frames = torch.cat(grouped[speaker], dim=0)
            positions = torch.linspace(
                0, len(frames) - 1, quota, device=frames.device
            ).long()
            parts.append(frames[positions])
        return torch.cat(parts, dim=0).to(device)
    parts = []
    for ident in background_ids:
        frames = feature[ident]
        count = min(len(frames), max_per_file)
        positions = torch.linspace(
            0, len(frames) - 1, count, device=frames.device
        ).long()
        parts.append(frames[positions])
    return torch.cat(parts, dim=0).to(device)


def _cohort(
    background: list[dict], feature: dict[str, torch.Tensor],
    ubm: TorchGMM, cfg_map: MAPConfig, cfg_score: ScoreConfig, seed: int,
) -> tuple[dict[str, TorchGMM], list[str]]:
    if cfg_score.norm == "none":
        return {}, []
    speaker_files: dict[str, list[str]] = defaultdict(list)
    for row in background:
        speaker_files[row["speaker_id"]].append(row["file_id"])
    eligible = sorted(s for s, files in speaker_files.items() if len(files) >= 3)
    if len(eligible) < 2:
        raise ValueError("Not enough three-take background speakers to form a cohort")
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(cfg_score.cohort_speakers, len(eligible)))
    cohort_rows = [
        {"speaker_id": speaker, "file_id": f, "enrol_index": i}
        for speaker in chosen for i, f in enumerate(sorted(speaker_files[speaker])[:3])
    ]
    background_test = [
        row["file_id"] for row in background
        if row["speaker_id"] not in set(chosen)
    ]
    if not background_test:
        raise ValueError("No cohort-disjoint background test recordings")
    selected = rng.sample(background_test, min(cfg_score.cohort_test_files,
                                               len(background_test)))
    return enrolled_models(ubm, feature, cohort_rows, cfg_map), selected


def _save_scores(path: Path, trials: list[dict], scores: torch.Tensor, labeled: bool) -> None:
    cols = ["trial_id", "claimed_id", "test_file"]
    if labeled:
        cols.append("label")
    cols.append("score")
    rows = []
    for t, s in zip(trials, scores):
        row = {key: t[key] for key in cols if key != "score"}
        row["score"] = f"{float(s):.12f}"
        rows.append(row)
    write_csv(path, rows, cols)


def run(
    root: str | Path, config: RunConfig, output: str | Path, track: str = "kaggle",
    benchmark: str | Path | None = None, ubm_word_pool: str = "stop",
    manifests_override: dict | None = None,
) -> dict:
    """Train model(s), evaluate dev, optionally score eval; write deterministic artifacts."""
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    if track == "kaggle":
        data = KaggleData(root)
        manifests = {
            "background": data.background, "dev_enrol": data.dev_enrol,
            "dev_trials": data.dev_trials, "eval_enrol": data.eval_enrol,
            "eval_trials": data.eval_trials,
        }
        if any("google" in str(x).lower() for x in (root, output)):
            raise ValueError("Kaggle track may not use a Google-corpus path")
    elif track == "google":
        if benchmark is None:
            raise ValueError("--benchmark mandatory for Google research")
        data = GoogleResearchData(root, Path(benchmark) / "google_manifest.csv")
        manifests = load_research_manifests(benchmark, ubm_word_pool)
    else:
        raise ValueError("track must be kaggle or google")
    if manifests_override is not None:
        if track != "kaggle":
            raise ValueError("Background-only proxy override requires Kaggle track")
        manifests = manifests_override
    bg, dev_enrol, dev_trials = (
        manifests["background"], manifests["dev_enrol"], manifests["dev_trials"]
    )
    eval_enrol, eval_trials = manifests["eval_enrol"], manifests["eval_trials"]
    populations = [
        {r["speaker_id"] for r in pool}
        for pool in (bg, dev_enrol, eval_enrol)
    ]
    if (populations[0] & populations[1] or populations[0] & populations[2]
            or populations[1] & populations[2]):
        raise ValueError("Train/enrol speaker leakage in supplied experiment")
    bg_ids = list(dict.fromkeys(row["file_id"] for row in bg))
    if config.max_background_files is not None:
        rng = random.Random(2026)
        bg_ids = sorted(rng.sample(bg_ids, min(config.max_background_files, len(bg_ids))))
        bg = [row for row in bg if row["file_id"] in set(bg_ids)]
    if isinstance(data, KaggleData):
        data.assert_background_only(bg_ids)
    if not bg_ids:
        raise ValueError("No background recordings")
    ids = list(dict.fromkeys(
        bg_ids + [r["file_id"] for r in dev_enrol] +
        [r["test_file"] for r in dev_trials] +
        ([r["file_id"] for r in eval_enrol] +
         [r["test_file"] for r in eval_trials] if config.evaluate_eval else [])
    ))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    features = extract_features(
        data, ids, config.feature, output.parent / ".phase_cache", device
    )
    data.close() if isinstance(data, KaggleData) else None
    if config.transform in ("speaker_cnn", "contrastive_cnn"):
        from .learned import PhaseEncoderConfig, learn_phase_encoder
        learned_cfg = dict(config.learned)
        learned_cfg["task"] = (
            "background_speaker" if config.transform == "speaker_cnn" else "contrastive"
        )
        learned_cfg.setdefault("embedding", config.dimensions)
        learned_cfg.setdefault("seed", config.seeds[0])
        transformed, transform = learn_phase_encoder(
            features, bg, PhaseEncoderConfig(**learned_cfg), device
        )
    else:
        transformed, transform = fit_background_transform(
            features, bg_ids, config.transform, config.dimensions,
            config.max_fit_transform_frames, seed=config.seeds[0]
        )
    del features
    transformed = {key: value.to(device) for key, value in transformed.items()}
    speaker_by_file = {row["file_id"]: row["speaker_id"] for row in bg}
    pool = _background_frames(
        transformed, bg_ids, config.max_frames_per_background_file, device,
        speaker_by_file=speaker_by_file, strategy=config.background_sampling,
    )
    labels = torch.tensor([int(row["label"]) for row in dev_trials], device="cpu")
    dev_per_seed, eval_per_seed, diagnostics = [], [], []
    for seed in config.seeds:
        set_seed(seed)
        em = replace(config.em, seed=seed)
        trained = TorchGMM(em).fit(pool)
        adapted_dev = enrolled_models(
            trained, transformed, dev_enrol, config.map, config.enrolment_size
        )
        raw_dev = score_trials(trained, transformed, dev_trials, adapted_dev, config.score)
        if config.score.norm != "none":
            cohorts, cohort_tests = _cohort(
                bg, transformed, trained, config.map, config.score, seed
            )
            dev = cohort_normalize(trained, transformed, dev_trials, raw_dev,
                                   adapted_dev, cohorts, cohort_tests, config.score)
        else:
            dev = raw_dev
        dev_per_seed.append(dev.cpu())
        if config.evaluate_eval:
            adapted_eval = enrolled_models(
                trained, transformed, eval_enrol, config.map, config.enrolment_size
            )
            raw_eval = score_trials(trained, transformed, eval_trials, adapted_eval,
                                    config.score)
            if config.score.norm != "none":
                ev = cohort_normalize(trained, transformed, eval_trials, raw_eval,
                                      adapted_eval, cohorts, cohort_tests, config.score)
            else:
                ev = raw_eval
            eval_per_seed.append(ev.cpu())
        diag = {
            "seed": seed,
            "training_log_likelihood": trained.history[-1],
            "convergence_history": trained.history,
            "dead_components": trained.dead_components,
            "mixture_weights_min": float(trained.weights.min()),
            "covariance_min": float(trained.cov.min()) if em.covariance in
                              ("diag", "spherical") else float(trained.cov.diagonal(
                                  dim1=-2, dim2=-1).min()),
            "information_criteria": trained.information_criteria(pool),
        }
        diagnostics.append(diag)
        if config.save_model:
            torch.save({
                "source_track": track, "feature": asdict(config.feature),
                "transformation": transform, "gmm": trained.snapshot(),
                "map": asdict(config.map), "scoring": asdict(config.score),
            }, output / f"model_seed_{seed}.pt")
        print(f"seed={seed} dev_EER={report(dev.cpu(), labels)['eer']:.4%}", flush=True)
    mean_dev = torch.stack(dev_per_seed).mean(0)
    if not torch.isfinite(mean_dev).all():
        raise FloatingPointError("Nonfinite development predictions")
    _save_scores(output / "development_scores.csv", dev_trials, mean_dev, labeled=True)
    if config.evaluate_eval:
        mean_eval = torch.stack(eval_per_seed).mean(0)
        if not torch.isfinite(mean_eval).all():
            raise FloatingPointError("Nonfinite evaluation predictions")
        _save_scores(output / "evaluation_scores.csv", eval_trials, mean_eval, labeled=False)
        if track == "kaggle":
            # The required submission always contains just ID + unadjusted per-trial
            # GMM-UBM scoring (or permitted cohort normalization), no four-claim coupling.
            write_csv(
                output / "submission.csv",
                [{"trial_id": r["trial_id"], "score": f"{float(s):.12f}"}
                 for r, s in zip(eval_trials, mean_eval)],
                ["trial_id", "score"],
            )
    stats = report(mean_dev, labels, config.target_far)
    stats["bootstrap_by_claimed_speaker"] = grouped_bootstrap(
        dev_trials, mean_dev, group_key="claimed_id"
    )
    stats["bootstrap_by_test_recording"] = grouped_bootstrap(
        dev_trials, mean_dev, group_key="test_file"
    )
    # Provenance is recorded from manifests, not model-fitting on trial recordings.
    fingerprint = hashlib.sha256()
    source_files = (
        [Path(root) / x for x in ("background.csv", "dev_enrol.csv", "dev_trials.csv",
                                  "eval_enrol.csv", "eval_trials.csv", "sample_submission.csv")]
        if track == "kaggle" else
        [Path(benchmark) / x for x in ("google_manifest.csv", "google_speaker_splits.csv")]
    )
    for item in source_files:
        if item.is_file():
            fingerprint.update(item.name.encode())
            fingerprint.update(item.read_bytes())
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_revision = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    hardware = {
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device)
                            if device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                                     if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    result = {
        "track": track, "research_only": track == "google",
        "source": str(Path(root).resolve()),
        "benchmark": str(benchmark) if benchmark else None,
        "device": str(device), "torch_version": torch.__version__,
        "hardware": hardware,
        "git_revision": git_revision,
        "source_manifest_sha256": fingerprint.hexdigest(),
        "config": config.to_dict(), "metrics": stats, "seed_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - t0,
        "background_frames": len(pool), "background_utterances": len(bg_ids),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    return result
