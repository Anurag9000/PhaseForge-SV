"""Original Google Speech Commands keyword-architecture benchmark, RESEARCH ONLY.

BCNN, KWT, DS-CNN, TC-ResNet, CRNN, and ResNet are evaluated as keyword
classifiers on phase-derived features, using original speaker-disjoint splits.
Word accuracy is NOT compared numerically to speaker-verification EER.
No Kaggle data or pretrained weights are used here.
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .data import GoogleResearchData, read_csv, write_csv
from .engine import extract_features, select_device
from .features import FeatureConfig
from .new_architectures import Architecture, PhaseSpeakerEncoder
from .new_neural_runner import frames_batch, augment_phase_features


@dataclass(frozen=True)
class KeywordRun:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    architecture: Architecture = field(default_factory=lambda: Architecture(name="bcnn"))
    words: tuple[str, ...] = (
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    )
    epochs: int = 10
    steps_per_epoch: int = 100
    batch_size: int = 32
    max_frames: int = 128
    max_files_per_split: int | None = None
    learning_rate: float = .001
    seed: int = 2026
    device: str = "auto"
    evaluate_locked_test: bool = False

    @classmethod
    def from_dict(cls, obj):
        feat = dict(obj.get("feature", {}))
        for key in ("multi_modes", "multi_windows"):
            if key in feat:
                feat[key] = tuple(feat[key])
        return cls(
            feature=FeatureConfig(**feat),
            architecture=Architecture(**obj.get("architecture", {})),
            **{key: (tuple(value) if key == "words" else value)
               for key, value in obj.items() if key not in ("feature", "architecture")},
        )

    def validate(self):
        self.feature.validate()
        self.architecture.validate()
        if self.epochs < 1 or self.steps_per_epoch < 1 or self.batch_size < 2:
            raise ValueError("Invalid keyword training settings")
        if len(set(self.words)) != len(self.words) or len(self.words) < 2:
            raise ValueError("At least two distinct target keywords required")


def run_keyword(root: str | Path, benchmark: str | Path,
                config: KeywordRun, output: str | Path):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    benchmark = Path(benchmark)
    protocol = json.loads((benchmark / "google_benchmark.json").read_text())
    if not protocol["research_only"]:
        raise PermissionError("Keyword classification only on original Google research")
    split_rows = read_csv(benchmark / "google_speaker_splits.csv")
    assignment = {row["speaker_id"]: row["split"] for row in split_rows}
    inventory = read_csv(benchmark / "google_manifest.csv")
    source = GoogleResearchData(root, benchmark / "google_manifest.csv")
    train, dev, test = [], [], []
    mapping = {name: i for i, name in enumerate(config.words)}
    for item in inventory:
        if item["word"] not in mapping or item["exact_duplicate"] != "0":
            continue
        if item["sample_rate"] != "16000" or item["channels"] != "1":
            continue
        label = assignment.get(item["speaker_id"])
        if label is None:
            raise ValueError("Original speaker not in locked benchmark population")
        row = {"file_id": item["file_id"], "label": mapping[item["word"]],
               "speaker_id": item["speaker_id"], "word": item["word"]}
        {"background": train, "dev": dev, "test": test}[label].append(row)
    if not train or not dev or (config.evaluate_locked_test and not test):
        raise ValueError("Speaker-disjoint Google keyword population is empty")
    rng = random.Random(config.seed)
    def choose(rows):
        rows = sorted(rows, key=lambda x: x["file_id"])
        if config.max_files_per_split is not None and len(rows) > config.max_files_per_split:
            rows = rng.sample(rows, config.max_files_per_split)
        return rows
    train, dev, test = map(choose, (train, dev, test))
    # This restriction is applied before feature extraction. The untouched
    # test partition is not loaded unless explicitly requested.
    used = train + dev + (test if config.evaluate_locked_test else [])
    device = select_device(config.device)
    start = time.perf_counter()
    features = extract_features(
        source, [r["file_id"] for r in used], config.feature,
        output.parent / ".keyword_cache", device,
    )
    ids = [r["file_id"] for r in train]
    targets = {r["file_id"]: r["label"] for r in used}
    torch.manual_seed(config.seed)
    encoder = PhaseSpeakerEncoder(features[ids[0]].shape[1], config.architecture).to(device)
    classifier = nn.Linear(config.architecture.embedding, len(config.words)).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()),
                                  lr=config.learning_rate)
    losses = []
    for epoch in range(config.epochs):
        encoder.train()
        order = list(ids)
        rng.shuffle(order)
        history = []
        for step in range(config.steps_per_epoch):
            chosen = rng.choices(order, k=config.batch_size)
            x, mask = frames_batch(features, chosen, config.max_frames, device, rng)
            augmented = augment_phase_features(x, mask, .02, .08)
            y = torch.tensor([targets[file] for file in chosen], device=device)
            logits = classifier(encoder(augmented, mask))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
            optimizer.step()
            history.append(float(loss.detach()))
        losses.append(sum(history) / len(history))
        print(f"Keyword {config.architecture.name} epoch={epoch+1} loss={losses[-1]:.4f}",
              flush=True)
    encoder.eval()
    classifier.eval()
    reports = {}
    with torch.no_grad():
        for name, rows in (("development", dev), ("locked_test", test)):
            if name == "locked_test" and not config.evaluate_locked_test:
                continue
            truth, predictions = [], []
            for begin in range(0, len(rows), config.batch_size):
                selected = rows[begin:begin + config.batch_size]
                x, mask = frames_batch(
                    features, [r["file_id"] for r in selected],
                    config.max_frames, device, rng,
                )
                pred = classifier(encoder(x, mask)).argmax(1).cpu().tolist()
                truth.extend([r["label"] for r in selected])
                predictions.extend(pred)
            confusion = torch.zeros(len(config.words), len(config.words), dtype=torch.int64)
            for y, yhat in zip(truth, predictions):
                confusion[y, yhat] += 1
            tp = confusion.diag().float()
            precision = tp / confusion.sum(0).clamp_min(1)
            recall = tp / confusion.sum(1).clamp_min(1)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
            reports[name] = {
                "accuracy": sum(int(a == b) for a, b in zip(truth, predictions)) /
                            max(1, len(truth)),
                "macro_f1": float(f1.mean()),
                "count": len(truth), "confusion_matrix": confusion.tolist(),
                "words": list(config.words),
            }
            write_csv(output / f"{name}_keywords.csv", [
                {"file_id": row["file_id"], "word": row["word"],
                 "true_class": y, "predicted_class": pred}
                for row, y, pred in zip(rows, truth, predictions)
            ], ["file_id", "word", "true_class", "predicted_class"])
    torch.save({
        "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "classifier": {k: v.cpu() for k, v in classifier.state_dict().items()},
        "config": asdict(config), "source_track": "original_google_keyword_research",
    }, output / "checkpoint.pt")
    result = {
        "source_track": "original_google_keyword_research",
        "research_only": True,
        "metric_is_speaker_verification_eer": False,
        "config": asdict(config), "losses": losses,
        "metrics": reports, "seconds": time.perf_counter() - start,
        "background_speakers": len({r["speaker_id"] for r in train}),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
