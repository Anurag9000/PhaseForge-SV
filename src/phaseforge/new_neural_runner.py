"""Independent background-only phase speaker-embedding training and verification.

No original Google corpus is used in a Kaggle track. These neural systems
are RESEARCH COMPARATORS, not eligible mandatory GMM-UBM Kaggle submissions.
A distinct locked Google test is only evaluated after explicit opt-in.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .data import KaggleData, GoogleResearchData, write_csv
from .engine import extract_features, load_research_manifests, select_device
from .features import FeatureConfig
from .metrics import grouped_bootstrap, report
from .new_architectures import Architecture, PhaseSpeakerEncoder
from .sampling import speaker_aware_subset
from .new_objectives import (
    AAMSoftmax, GE2ELoss, PairVerifier, batch_hard_triplet,
    latent_predictive_loss, pairwise_bce, prototypical_loss, supervised_contrastive,
)


OBJECTIVES = (
    "cross_entropy", "aam", "ge2e", "supcon", "triplet", "prototype",
    "pairwise", "byol", "jepa",
)
SCORES = ("cosine", "pairwise", "quality_cosine")


@dataclass(frozen=True)
class NeuralRun:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    architecture: Architecture = field(default_factory=Architecture)
    objective: str = "aam"
    epochs: int = 12
    steps_per_epoch: int = 100
    speakers_per_batch: int = 8
    takes_per_speaker: int = 3
    max_frames: int = 128
    learning_rate: float = 0.001
    weight_decay: float = .0001
    seed: int = 2026
    device: str = "auto"
    max_background_files: int | None = None
    score: str = "cosine"
    evaluate_locked_test: bool = False
    augment_noise: float = .02
    augment_mask_probability: float = .08
    ema_decay: float = .99
    variance_weight: float = .1
    variance_target: float = .08

    @classmethod
    def from_dict(cls, config: dict) -> "NeuralRun":
        feature = dict(config.get("feature", {}))
        for k in ("multi_modes", "multi_windows"):
            if k in feature:
                feature[k] = tuple(feature[k])
        return cls(
            feature=FeatureConfig(**feature),
            architecture=Architecture(**config.get("architecture", {})),
            **{k: v for k, v in config.items()
               if k not in ("feature", "architecture")},
        )

    def validate(self):
        self.feature.validate()
        self.architecture.validate()
        if self.objective not in OBJECTIVES or self.score not in SCORES:
            raise ValueError("Unknown objective/verification scorer")
        if self.epochs < 1 or self.steps_per_epoch < 1:
            raise ValueError("Invalid number of training steps")
        if self.speakers_per_batch < 2 or self.takes_per_speaker < 2:
            raise ValueError("Use >=2 background speakers and >=2 recordings each")
        if self.max_frames < 8 or self.learning_rate <= 0:
            raise ValueError("Invalid training feature length or learning rate")
        if not 0 <= self.augment_mask_probability < 1 or self.augment_noise < 0:
            raise ValueError("Invalid phase augmentation")
        if not 0 < self.ema_decay < 1:
            raise ValueError("EMA momentum must be in (0,1)")
        if self.variance_weight < 0 or self.variance_target <= 0:
            raise ValueError("Invalid latent variance regularization")


def speaker_batches(rows: list[dict], n: int, m: int, steps: int, seed: int):
    """Seeded, speaker-major episodic batches; replacement only across batches."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row["file_id"])
    eligible = sorted(s for s, ids in groups.items() if len(set(ids)) >= m)
    if len(eligible) < n:
        raise ValueError(f"Need {n} background speakers each with >= {m} takes; "
                         f"got {len(eligible)}")
    rng = random.Random(seed)
    for _ in range(steps):
        speakers = rng.sample(eligible, n)
        files = [file for speaker in speakers
                 for file in rng.sample(sorted(set(groups[speaker])), m)]
        yield files, torch.arange(n).repeat_interleave(m)


def frames_batch(
    store: dict[str, torch.Tensor], ids: list[str], max_frames: int,
    device: torch.device, rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    for file in ids:
        sequence = store[file]
        if len(sequence) > max_frames:
            start = rng.randrange(len(sequence) - max_frames + 1)
            sequence = sequence[start:start + max_frames]
        sequences.append(sequence)
    width = max(len(x) for x in sequences)
    tensor = torch.zeros(len(ids), width, sequences[0].shape[-1], device=device)
    valid = torch.zeros(len(ids), width, dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        tensor[i, :len(seq)] = seq.to(device)
        valid[i, :len(seq)] = True
    return tensor, valid


def augment_phase_features(
    tensor: torch.Tensor, valid: torch.Tensor,
    noise: float = .02, masked: float = .08,
) -> torch.Tensor:
    """Only phase-derived features, with additive jitter and temporal masking."""
    if noise < 0 or not 0 <= masked < 1:
        raise ValueError("Invalid feature augmentation")
    output = tensor + noise * torch.randn_like(tensor)
    if masked:
        retain = (torch.rand(tensor.shape[:2], device=tensor.device) >= masked)
        output = output * retain[:, :, None]
    return output * valid[:, :, None]


def train_phase_network(
    store: dict[str, torch.Tensor], background_rows: list[dict],
    config: NeuralRun, device: torch.device,
    checkpoint_file: str | Path | None = None,
    max_epochs_this_invocation: int | None = None,
    source_tag: str = "background_research",
) -> tuple[PhaseSpeakerEncoder, PairVerifier | None, dict]:
    """Train, with atomic at-least-once epoch checkpoints when a path is provided.

    A resumed job restores all optimizers, heads, EMA teacher, random generator
    states and history. An inconsistent source/code/config fingerprint triggers
    a fresh run rather than reusing incompatible model weights.
    """
    config.validate()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    speaker_names = sorted({r["speaker_id"] for r in background_rows})
    if len(speaker_names) < config.speakers_per_batch:
        raise ValueError("Insufficient background speaker identities")
    speaker_index = {s: i for i, s in enumerate(speaker_names)}
    file_to_speaker = {r["file_id"]: speaker_index[r["speaker_id"]] for r in background_rows}
    first = background_rows[0]["file_id"]
    encoder = PhaseSpeakerEncoder(store[first].shape[1], config.architecture).to(device)
    pair = PairVerifier(config.architecture.embedding).to(device) if (
        config.objective == "pairwise" or config.score == "pairwise"
    ) else None
    if config.score == "pairwise" and config.objective != "pairwise":
        raise ValueError("pairwise scoring requires the supervised pairwise objective")
    classifier = nn.Linear(config.architecture.embedding, len(speaker_names)).to(device)
    aam = AAMSoftmax(config.architecture.embedding, len(speaker_names)).to(device)
    ge2e = GE2ELoss().to(device)
    predictor = nn.Sequential(
        nn.Linear(config.architecture.embedding, config.architecture.embedding),
        nn.ReLU(), nn.Linear(config.architecture.embedding, config.architecture.embedding)
    ).to(device)
    teacher = None
    if config.objective in ("byol", "jepa"):
        teacher = copy.deepcopy(encoder).to(device).eval()
        teacher.requires_grad_(False)
    params = list(encoder.parameters())
    if config.objective == "cross_entropy":
        params += list(classifier.parameters())
    elif config.objective == "aam":
        params += list(aam.parameters())
    elif config.objective == "ge2e":
        params += list(ge2e.parameters())
    elif config.objective == "pairwise":
        params += list(pair.parameters())
    elif teacher is not None:
        params += list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = random.Random(config.seed)
    history = []
    source_files = (
        "new_neural_runner.py", "new_architectures.py",
        "new_objectives.py", "features.py", "sampling.py",
    )
    signature = hashlib.sha256()
    for name in source_files:
        signature.update((Path(__file__).parent / name).read_bytes())
    identity = hashlib.sha256(json.dumps({
        "source_tag": source_tag, "config": asdict(config),
        "background": sorted(
            (row["file_id"], row["speaker_id"]) for row in background_rows
        ), "source_sha256": signature.hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    checkpoint = Path(checkpoint_file) if checkpoint_file is not None else None
    start_epoch = 0
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get("identity") == identity:
            encoder.load_state_dict(saved["encoder"])
            classifier.load_state_dict(saved["classifier"])
            aam.load_state_dict(saved["aam"])
            ge2e.load_state_dict(saved["ge2e"])
            predictor.load_state_dict(saved["predictor"])
            if pair is not None:
                pair.load_state_dict(saved["pair"])
            if teacher is not None:
                teacher.load_state_dict(saved["teacher"])
            optimizer.load_state_dict(saved["optimizer"])
            history = list(saved["history"])
            start_epoch = int(saved["next_epoch"])
            if not 0 <= start_epoch <= config.epochs or len(history) != start_epoch:
                raise ValueError("Invalid resumed neural epoch history")
            rng.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if device.type == "cuda" and saved.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            print(
                f"Resumed {config.architecture.name}/{config.objective} "
                f"from completed epoch {start_epoch}", flush=True
            )
    stop_epoch = config.epochs
    if max_epochs_this_invocation is not None:
        if max_epochs_this_invocation < 1:
            raise ValueError("max_epochs_this_invocation must be positive")
        stop_epoch = min(stop_epoch, start_epoch + max_epochs_this_invocation)
    for epoch in range(start_epoch, stop_epoch):
        encoder.train()
        losses = []
        batches = speaker_batches(
            background_rows, config.speakers_per_batch,
            config.takes_per_speaker, config.steps_per_epoch, config.seed + epoch,
        )
        for files, episode_labels in batches:
            x, valid = frames_batch(store, files, config.max_frames, device, rng)
            augmented = augment_phase_features(
                x, valid, config.augment_noise, config.augment_mask_probability,
            )
            embeddings = encoder(augmented, valid)
            objective = config.objective
            episode_labels = episode_labels.to(device)
            if objective == "cross_entropy":
                global_labels = torch.tensor(
                    [file_to_speaker[file] for file in files], device=device
                )
                loss = F.cross_entropy(classifier(embeddings), global_labels)
            elif objective == "aam":
                global_labels = torch.tensor(
                    [file_to_speaker[file] for file in files], device=device
                )
                loss = aam(embeddings, global_labels)
            elif objective == "ge2e":
                loss = ge2e(embeddings, config.speakers_per_batch,
                            config.takes_per_speaker)
            elif objective == "supcon":
                loss = supervised_contrastive(embeddings, episode_labels)
            elif objective == "triplet":
                loss = batch_hard_triplet(embeddings, episode_labels)
            elif objective == "prototype":
                loss = prototypical_loss(embeddings, config.speakers_per_batch,
                                         config.takes_per_speaker)
            elif objective == "pairwise":
                loss = pairwise_bce(pair, embeddings, episode_labels)
            elif objective in ("byol", "jepa"):
                # The online network sees a corrupted context. EMA target
                # sees an INDEPENDENT, less-masked view. For JEPA-like
                # prediction, future context is removed only from ONLINE
                # input; the stop-gradient teacher retains the full context.
                target_view = augment_phase_features(
                    x, valid, config.augment_noise, config.augment_mask_probability
                )
                if objective == "jepa":
                    context_view = augmented.clone()
                    cutoff = max(1, context_view.shape[1] // 4)
                    context_view[:, -cutoff:] = 0
                    embeddings = encoder(context_view, valid)
                with torch.no_grad():
                    goal = teacher(target_view, valid)
                loss = latent_predictive_loss(predictor(embeddings), goal)
                if objective == "byol":
                    second_online = encoder(target_view, valid)
                    with torch.no_grad():
                        reversed_target = teacher(augmented, valid)
                    loss = .5 * (
                        loss + latent_predictive_loss(
                            predictor(second_online), reversed_target
                        )
                    )
                # Small VICReg-style variance floor discourages representation
                # collapse when training on tiny/short phase utterances.
                latents = embeddings.float()
                spread = latents.std(dim=0, unbiased=False)
                loss = loss + config.variance_weight * torch.relu(
                    config.variance_target - spread
                ).mean()
            else:
                raise ValueError(objective)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite neural loss: {objective}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.)
            optimizer.step()
            if teacher is not None:
                with torch.no_grad():
                    for old, new in zip(teacher.parameters(), encoder.parameters()):
                        old.lerp_(new, 1 - config.ema_decay)
                    # BatchNorm running statistics are buffers, not parameters;
                    # leaving them frozen at initialization makes target
                    # embeddings inconsistent with the EMA model weights.
                    for old, new in zip(teacher.buffers(), encoder.buffers()):
                        if old.dtype.is_floating_point:
                            old.lerp_(new, 1 - config.ema_decay)
                        else:
                            old.copy_(new)
            losses.append(float(loss.detach()))
        history.append(sum(losses) / len(losses))
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "identity": identity, "next_epoch": epoch + 1,
                "history": list(history),
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "aam": aam.state_dict(), "ge2e": ge2e.state_dict(),
                "predictor": predictor.state_dict(),
                "teacher": teacher.state_dict() if teacher is not None else None,
                "pair": pair.state_dict() if pair is not None else None,
                "optimizer": optimizer.state_dict(),
                "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda" else None,
            }
            temporary = checkpoint.with_suffix(".partial")
            torch.save(state, temporary)
            os.replace(temporary, checkpoint)
        print(f"architecture={config.architecture.name} objective={config.objective} "
              f"epoch={epoch+1}/{config.epochs} loss={history[-1]:.4f}", flush=True)
    encoder.eval()
    if pair is not None:
        pair.eval()
    artifact = {
        "model": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        "pair": ({k: v.detach().cpu() for k, v in pair.state_dict().items()}
                 if pair is not None else None),
        "loss_history": history, "completed_epochs": len(history),
        "speaker_count": len(speaker_names),
        "training_utterances": len(background_rows),
        "objective_state": {
            "classifier": {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
                if config.objective == "cross_entropy" else None,
            "aam": {k: v.detach().cpu() for k, v in aam.state_dict().items()}
                if config.objective == "aam" else None,
            "ge2e": {k: v.detach().cpu() for k, v in ge2e.state_dict().items()}
                if config.objective == "ge2e" else None,
            "predictor": {k: v.detach().cpu() for k, v in predictor.state_dict().items()}
                if teacher is not None else None,
            "ema_teacher": {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
                if teacher is not None else None,
        },
    }
    return encoder, pair, artifact


@torch.no_grad()
def compute_embeddings(
    encoder: PhaseSpeakerEncoder, features: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = {}
    for file, frames in features.items():
        x = frames[None].to(device)
        mask = torch.ones((1, len(frames)), dtype=torch.bool, device=device)
        result[file] = encoder(x, mask)[0].detach()
    return result


@torch.no_grad()
def verify_embeddings(
    embeddings: dict[str, torch.Tensor], enrolment: list[dict],
    trials: list[dict], k: int,
    scorer: str = "cosine", pair: PairVerifier | None = None,
) -> torch.Tensor:
    groups = defaultdict(list)
    for record in enrolment:
        groups[record["speaker_id"]].append(record)
    prototypes, uncertainty = {}, {}
    for speaker, rows in groups.items():
        if len(rows) < k:
            raise ValueError(f"Speaker {speaker} lacks {k} enrolment recordings")
        utterances = torch.stack([
            embeddings[row["file_id"]]
            for row in sorted(rows, key=lambda z: int(z["enrol_index"]))[:k]
        ])
        prototype = F.normalize(utterances.mean(0), dim=0)
        prototypes[speaker] = prototype
        uncertainty[speaker] = float(1 - (utterances @ prototype).mean())
    results = []
    for row in trials:
        claimed = row["claimed_id"]
        test = embeddings[row["test_file"]]
        if scorer == "pairwise":
            if pair is None:
                raise ValueError("Missing trained pairwise verifier")
            score = pair(prototypes[claimed][None], test[None]).squeeze()
        else:
            score = (prototypes[claimed] * test).sum()
            if scorer == "quality_cosine":
                score = score / (1 + uncertainty[claimed])
            elif scorer != "cosine":
                raise ValueError(scorer)
        results.append(float(score))
    return torch.tensor(results, dtype=torch.float64)


def run_neural(
    root: str | Path, config: NeuralRun, output: str | Path,
    track: str = "kaggle", benchmark: str | Path | None = None,
    pool: str = "stop",
) -> dict:
    """Train and score strictly background-only neural research models.

    A Kaggle track creates NO submission.csv; these models are outside
    the challenge's required primary GMM-UBM formulation.
    """
    config.validate()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    if track == "kaggle":
        data = KaggleData(root)
        rows = {
            "background": data.background, "dev_enrol": data.dev_enrol,
            "dev_trials": data.dev_trials, "eval_enrol": data.eval_enrol,
            "eval_trials": data.eval_trials,
        }
    elif track == "google":
        if benchmark is None:
            raise ValueError("Google research requires locked --benchmark directory")
        rows = load_research_manifests(benchmark, pool)
        protocol = json.loads((Path(benchmark) / "google_benchmark.json").read_text())
        data = GoogleResearchData(root, Path(benchmark) / "google_manifest.csv")
    else:
        raise ValueError(track)
    background = speaker_aware_subset(
        list(rows["background"]), config.max_background_files,
        min_takes=config.takes_per_speaker,
        min_speakers=config.speakers_per_batch,
        seed=config.seed,
    )
    if isinstance(data, KaggleData):
        data.assert_background_only([r["file_id"] for r in background])
    if config.score == "pairwise" and config.objective != "pairwise":
        raise ValueError("Only pairwise-trained verifiers may use pairwise scores")
    start = time.perf_counter()
    ids = {r["file_id"] for r in background}
    ids.update(r["file_id"] for r in rows["dev_enrol"])
    ids.update(r["test_file"] for r in rows["dev_trials"])
    if config.evaluate_locked_test:
        ids.update(r["file_id"] for r in rows["eval_enrol"])
        ids.update(r["test_file"] for r in rows["eval_trials"])
    features = extract_features(
        data, sorted(ids), config.feature, out.parent / ".new_phase_cache", device
    )
    if isinstance(data, KaggleData):
        data.close()
    encoder, pair, trained = train_phase_network(
        features, background, config, device,
        checkpoint_file=out / "training_epoch_checkpoint.pt",
        source_tag=f"{track}:{Path(root).resolve()}",
    )
    embeddings = compute_embeddings(encoder, features, device)
    k = (3 if track == "kaggle" else int(protocol["enrolment_takes_per_speaker"]))
    # The benchmark specifies enrolment size. Never tune this on hidden trials.
    dev = verify_embeddings(
        embeddings, rows["dev_enrol"], rows["dev_trials"], k,
        config.score, pair,
    )
    dev_labels = torch.tensor([int(r["label"]) for r in rows["dev_trials"]])
    dev_metrics = report(dev, dev_labels)
    write_csv(out / "development_scores.csv", [
        {**{k: r[k] for k in ("trial_id", "claimed_id", "test_file", "label")},
         "score": f"{float(z):.12f}"}
        for r, z in zip(rows["dev_trials"], dev)
    ], ["trial_id", "claimed_id", "test_file", "label", "score"])
    test_metrics = None
    if config.evaluate_locked_test:
        ev = verify_embeddings(
            embeddings, rows["eval_enrol"], rows["eval_trials"], k,
            config.score, pair,
        )
        labeled = track == "google"
        columns = ["trial_id", "claimed_id", "test_file"] + (
            ["label"] if labeled else []
        ) + ["score"]
        write_csv(out / "evaluation_scores.csv", [
            {**{k: r[k] for k in columns if k != "score"},
             "score": f"{float(z):.12f}"}
            for r, z in zip(rows["eval_trials"], ev)
        ], columns)
        if labeled:
            test_labels = torch.tensor([int(r["label"]) for r in rows["eval_trials"]])
            test_metrics = report(ev, test_labels)
    torch.save({
        "source_track": track, "research_only": True,
        "config": asdict(config), **trained,
    }, out / "checkpoint.pt")
    result = {
        "source_track": track, "research_only": True,
        "eligible_for_mandatory_kaggle_submission": False,
        "architecture": config.architecture.name, "objective": config.objective,
        "config": asdict(config), "train_losses": trained["loss_history"],
        "metrics": dev_metrics, "locked_google_test_metrics": test_metrics,
        "background_utterances": len(background),
        "seconds": time.perf_counter() - start,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result
