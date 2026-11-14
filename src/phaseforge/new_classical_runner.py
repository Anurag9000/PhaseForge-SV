"""Routable, speaker-disjoint phase GMM-supervector, i-vector, PLDA and SVM experiments.

Kaggle: background.csv only for UBM, PCA, total variability, PLDA and SVM.
Google: research background speakers only. Independent research scores are
NEVER exported as a Kaggle competition submission.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import KaggleData, GoogleResearchData, write_csv
from .engine import (
    extract_features, load_research_manifests, select_device, _background_frames,
)
from .features import FeatureConfig
from .gmm import EMConfig, MAPConfig, TorchGMM
from .metrics import report
from .new_classical import (
    PairwiseLinearSVM, TotalVariability, TwoCovariancePLDA,
    map_supervector, supervised_lda,
)
from .transforms import fit_background_transform


METHODS = ("supervector_cosine", "supervector_svm",
           "ivector_cosine", "ivector_plda", "ivector_lda_plda",
           "passphrase_ubm")


@dataclass(frozen=True)
class ClassicalRun:
    method: str = "ivector_cosine"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    em: EMConfig = field(default_factory=lambda: EMConfig(mixtures=16))
    map: MAPConfig = field(default_factory=MAPConfig)
    transform: str = "pca"
    dimensions: int = 24
    tv_dimension: int = 16
    tv_iterations: int = 8
    lda_dimension: int = 12
    frame_cap: int = 24
    max_background_files: int | None = None
    seed: int = 2026
    device: str = "auto"
    evaluate_locked_test: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        feature = dict(data.get("feature", {}))
        for k in ("multi_modes", "multi_windows"):
            if k in feature:
                feature[k] = tuple(feature[k])
        return cls(feature=FeatureConfig(**feature),
                   em=EMConfig(**data.get("em", {})), map=MAPConfig(**data.get("map", {})),
                   **{k: v for k, v in data.items() if k not in ("feature", "em", "map")})

    def validate(self):
        self.feature.validate()
        self.em.validate()
        self.map.validate()
        if self.method not in METHODS:
            raise ValueError("Unknown classical speaker-verification backend")
        if self.tv_dimension < 1 or self.dimensions < 1 or self.frame_cap < 1:
            raise ValueError("Invalid classical backend dimensions")
        if self.em.covariance != "diag":
            raise ValueError("i-vector and supervector require diagonal GMM UBM")


@torch.no_grad()
def enrol_templates(features: dict[str, torch.Tensor], enrol: list[dict],
                    feature_vector, k: int):
    grouped = defaultdict(list)
    for row in enrol:
        grouped[row["speaker_id"]].append(row)
    templates = {}
    for speaker, files in grouped.items():
        if len(files) < k:
            raise ValueError(f"Speaker {speaker} has fewer than {k} enrolment takes")
        X = torch.stack([feature_vector(features[r["file_id"]])
                         for r in sorted(files, key=lambda x: int(x["enrol_index"]))[:k]])
        templates[speaker] = F.normalize(X.mean(0), dim=-1)
    return templates


@torch.no_grad()
def classify_trials(trials: list[dict], features: dict[str, torch.Tensor],
                    vector, templates: dict, backend=None):
    cache = {}
    for test_file in {row["test_file"] for row in trials}:
        cache[test_file] = F.normalize(vector(features[test_file]), dim=-1)
    scores = []
    for row in trials:
        a, b = templates[row["claimed_id"]], cache[row["test_file"]]
        if backend is None:
            score = (a * b).sum()
        else:
            score = backend.score(a, b)
        scores.append(float(score))
    return torch.tensor(scores, dtype=torch.float64)


def run_classical(
    root: str | Path, config: ClassicalRun, output: str | Path,
    track: str = "kaggle", benchmark: str | Path | None = None,
) -> dict:
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    if track == "kaggle":
        if config.method == "passphrase_ubm":
            raise ValueError("Word-specific background PBM requires Google word labels")
        data = KaggleData(root)
        manifests = dict(
            background=data.background, dev_enrol=data.dev_enrol,
            dev_trials=data.dev_trials, eval_enrol=data.eval_enrol,
            eval_trials=data.eval_trials,
        )
        k = 3
    elif track == "google":
        if benchmark is None:
            raise ValueError("Google speaker-disjoint research needs --benchmark")
        manifests = load_research_manifests(benchmark, "all")
        protocol = json.loads((Path(benchmark) / "google_benchmark.json").read_text())
        k = int(protocol["enrolment_takes_per_speaker"])
        data = GoogleResearchData(root, Path(benchmark) / "google_manifest.csv")
    else:
        raise ValueError("Unsupported track")
    background = manifests["background"]
    if config.max_background_files is not None:
        selected = set(random.Random(config.seed).sample(
            sorted({r["file_id"] for r in background}),
            min(len(background), config.max_background_files)
        ))
        background = [r for r in background if r["file_id"] in selected]
    if isinstance(data, KaggleData):
        data.assert_background_only([r["file_id"] for r in background])
    source_background = background
    if config.method == "passphrase_ubm":
        manifest = {r["file_id"]: r for r in data.records}
        background = [r for r in background if manifest[r["file_id"]]["word"] == "stop"]
        if not background:
            raise ValueError("No background stop utterances for passphrase PBM")
    ids = {r["file_id"] for r in source_background}
    ids.update(r["file_id"] for r in manifests["dev_enrol"])
    ids.update(r["test_file"] for r in manifests["dev_trials"])
    if config.evaluate_locked_test:
        ids.update(r["file_id"] for r in manifests["eval_enrol"])
        ids.update(r["test_file"] for r in manifests["eval_trials"])
    started = time.perf_counter()
    feature = extract_features(
        data, sorted(ids), config.feature, output.parent / ".classical_cache", device
    )
    if isinstance(data, KaggleData):
        data.close()
    bg_ids = [r["file_id"] for r in background]
    transformed, learned_transform = fit_background_transform(
        feature, bg_ids, config.transform, config.dimensions,
        max_frames=50000, seed=config.seed,
    )
    transformed = {name: x.to(device) for name, x in transformed.items()}
    pool = _background_frames(transformed, bg_ids, config.frame_cap, device)
    ubm = TorchGMM(config.em).fit(pool)
    method = config.method
    backend = None
    extra = {}
    if method == "passphrase_ubm":
        global_ids = [r["file_id"] for r in source_background]
        # Matched feature transform remains trained on stop-only background.
        global_pool = _background_frames(transformed, global_ids, config.frame_cap, device)
        global_ubm = TorchGMM(config.em).fit(global_pool)
        extra["global_ubm"] = global_ubm.snapshot()
        # Speaker models are MAP adapted from stop-specific UBM.
        def phrase_score(trials, enrol):
            models = {}
            grouped = defaultdict(list)
            for r in enrol:
                grouped[r["speaker_id"]].append(r)
            for speaker, files in grouped.items():
                if len(files) < k:
                    raise ValueError(f"Speaker {speaker} has too few enrolment takes")
                X = torch.cat([
                    transformed[r["file_id"]]
                    for r in sorted(files, key=lambda x: int(x["enrol_index"]))[:k]
                ])
                models[speaker] = ubm.adapt(X, config.map)
            scores = []
            for row in trials:
                x = transformed[row["test_file"]]
                # Phrase-vs-global competing background: explicitly research-only.
                numerator = models[row["claimed_id"]].log_prob(x)
                denominator = torch.logaddexp(
                    ubm.log_prob(x), global_ubm.log_prob(x)
                ) - .6931471805599453
                scores.append(float((numerator - denominator).mean()))
            return torch.tensor(scores, dtype=torch.float64)
        vector = None
        raw_bg = {}
    else:
        if method.startswith("ivector"):
            iv = TotalVariability(ubm, config.tv_dimension,
                                  config.tv_iterations, config.seed).fit({
                file: transformed[file] for file in bg_ids
            })
            vector = iv.transform
            extra["total_variability"] = iv.T.detach().cpu()
            extra["tv_history"] = iv.history
        else:
            vector = lambda x: map_supervector(ubm, x, config.map.relevance)
        # Background embeddings/labels train all discriminant backends.
        raw_bg = {r["file_id"]: vector(transformed[r["file_id"]]) for r in background}
        labels = {r["file_id"]: r["speaker_id"] for r in background}
        if method == "ivector_lda_plda":
            center, projection = supervised_lda(
                raw_bg, labels, min(config.lda_dimension, config.tv_dimension)
            )
            old_vector = vector
            vector = lambda x: F.normalize((old_vector(x) - center) @ projection, dim=-1)
            raw_bg = {
                file: F.normalize((x - center) @ projection, dim=-1)
                for file, x in raw_bg.items()
            }
            extra["lda_center"] = center.detach().cpu()
            extra["lda_projection"] = projection.detach().cpu()
        if method.endswith("plda"):
            backend = TwoCovariancePLDA(shrinkage=.2).fit(raw_bg, labels)
            extra["plda"] = {k: getattr(backend, k).detach().cpu()
                             for k in ("mean", "B", "W")}
        elif method == "supervector_svm":
            # Cap pair mining to avoid accidental O(N^2) memory in big corpora.
            speaker_ids = defaultdict(list)
            for file, speaker in labels.items():
                speaker_ids[speaker].append(file)
            rng = random.Random(config.seed)
            chosen = []
            for speaker in sorted(speaker_ids):
                chosen.extend(rng.sample(speaker_ids[speaker],
                                         min(3, len(speaker_ids[speaker]))))
            if len(chosen) > 300:
                chosen = rng.sample(chosen, 300)
            subset = {file: raw_bg[file] for file in chosen}
            backend = PairwiseLinearSVM(epochs=80).fit(subset, labels, config.seed)
            extra["svm"] = {"weight": backend.W.detach().cpu(),
                            "bias": backend.bias.detach().cpu()}
    summary = {}
    for partition in ("dev", "eval"):
        if partition == "eval" and not config.evaluate_locked_test:
            continue
        enrol = manifests[f"{partition}_enrol"]
        trials = manifests[f"{partition}_trials"]
        if method == "passphrase_ubm":
            scores = phrase_score(trials, enrol)
        else:
            templates = enrol_templates(transformed, enrol, vector, k)
            scores = classify_trials(trials, transformed, vector, templates, backend)
        labeled = partition == "dev" or track == "google"
        columns = ["trial_id", "claimed_id", "test_file"] + (
            ["label"] if labeled else []
        ) + ["score"]
        write_csv(output / ("development_scores.csv" if partition=="dev"
                             else "evaluation_scores.csv"), [
            {**{key: row[key] for key in columns if key != "score"},
             "score": f"{float(value):.12f}"}
            for row, value in zip(trials, scores)
        ], columns)
        if labeled:
            labels = torch.tensor([int(row["label"]) for row in trials])
            summary[partition] = report(scores, labels)
    torch.save({
        "research_only": True, "source_track": track,
        "config": asdict(config), "ubm": ubm.snapshot(),
        "feature_transform": learned_transform, **extra,
    }, output / "checkpoint.pt")
    result = {
        "method": method, "track": track, "research_only": True,
        "eligible_for_mandatory_kaggle_submission": False,
        "config": asdict(config),
        "metrics": summary["dev"],
        "locked_google_test_metrics": summary.get("eval") if track=="google" else None,
        "background_utterances": len(bg_ids),
        "ubm_log_likelihood": ubm.history[-1],
        "seconds": time.perf_counter() - started,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
