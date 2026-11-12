"""Input boundaries, source isolation and speaker-disjoint benchmark construction.

Only this module opens audio files. Competition code *never* opens the Google corpus.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torchaudio

def decode_pcm16_wav(content: bytes) -> tuple[torch.Tensor, int]:
    """Read mono PCM16 WAV with stdlib wave (I/O) and torch (numeric conversion).

    This avoids torchaudio>=2.9's optional TorchCodec runtime dependency;
    phase feature extraction, GMM EM and scoring remain torch-only.
    """
    with wave.open(io.BytesIO(content), "rb") as recording:
        channels = recording.getnchannels()
        sample_width = recording.getsampwidth()
        rate = recording.getframerate()
        if channels != 1 or sample_width != 2:
            raise ValueError("Expected uncompressed mono 16-bit PCM WAV")
        samples = recording.readframes(recording.getnframes())
    waveform = torch.frombuffer(bytearray(samples), dtype=torch.int16).float() / 32768.0
    return waveform, rate


_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_REQUIRED = (
    "background.csv", "dev_enrol.csv", "dev_trials.csv",
    "eval_enrol.csv", "eval_trials.csv", "sample_submission.csv",
)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Iterable[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


class KaggleData:
    """Strict file-ID allowlist; training and feature-fitting IDs must be background-only."""

    source = "kaggle"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for name in _REQUIRED:
            if not (self.root / name).is_file():
                raise FileNotFoundError(f"Required competition file missing: {self.root / name}")
        self.audio_zip = self.root / "audio.zip"
        self.audio_dir = self.root / "audio"
        if not self.audio_zip.is_file() and not self.audio_dir.is_dir():
            raise FileNotFoundError("Expected competition audio.zip or audio/ subdirectory")
        self.background = read_csv(self.root / "background.csv")
        self.dev_enrol = read_csv(self.root / "dev_enrol.csv")
        self.eval_enrol = read_csv(self.root / "eval_enrol.csv")
        self.dev_trials = read_csv(self.root / "dev_trials.csv")
        self.eval_trials = read_csv(self.root / "eval_trials.csv")
        self.sample_submission = read_csv(self.root / "sample_submission.csv")
        groups = [
            {r["speaker_id"] for r in x}
            for x in [self.background, self.dev_enrol, self.eval_enrol]
        ]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Speaker split leakage across background/dev/eval")
        for name, rows in (("dev", self.dev_enrol), ("eval", self.eval_enrol)):
            counts = Counter(r["speaker_id"] for r in rows)
            if any(count != 3 for count in counts.values()):
                raise ValueError(f"{name} target does not have exactly three enrolment utterances")
        if [r["trial_id"] for r in self.eval_trials] != [
            r["trial_id"] for r in self.sample_submission
        ]:
            raise ValueError("Evaluation trial IDs differ from sample submission")
        self.background_ids = frozenset(r["file_id"] for r in self.background)
        self.dev_enrol_ids = frozenset(r["file_id"] for r in self.dev_enrol)
        self.eval_enrol_ids = frozenset(r["file_id"] for r in self.eval_enrol)
        self.dev_test_ids = frozenset(r["test_file"] for r in self.dev_trials)
        self.eval_test_ids = frozenset(r["test_file"] for r in self.eval_trials)
        self.allowed_ids = (
            self.background_ids | self.dev_enrol_ids | self.eval_enrol_ids
            | self.dev_test_ids | self.eval_test_ids
        )
        for file_id in self.allowed_ids:
            if not _ID.fullmatch(file_id):
                raise ValueError(f"Unsafe audio file ID: {file_id!r}")
        self._zip: zipfile.ZipFile | None = None

    def assert_background_only(self, ids: Iterable[str]) -> None:
        ids = set(ids)
        extraneous = ids - self.background_ids
        if extraneous:
            raise ValueError(f"Non-background training audio forbidden: {sorted(extraneous)[:4]}")

    def load(self, file_id: str) -> torch.Tensor:
        if file_id not in self.allowed_ids:
            raise PermissionError("Audio ID is not in the Kaggle competition manifests")
        if self.audio_zip.is_file():
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.audio_zip)
            content = self._zip.read(f"audio/{file_id}.wav")
        else:
            content = (self.audio_dir / f"{file_id}.wav").read_bytes()
        waveform, rate = decode_pcm16_wav(content)
        if rate != 16000:
            raise ValueError(f"Expected mono 16kHz: {file_id}, rate={rate}")
        return waveform

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> "KaggleData":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def summary(self) -> dict:
        return {
            "source": self.source,
            "background_utterances": len(self.background),
            "background_speakers": len({r["speaker_id"] for r in self.background}),
            "dev_enrol": len(self.dev_enrol),
            "eval_enrol": len(self.eval_enrol),
            "dev_trials": len(self.dev_trials),
            "eval_trials": len(self.eval_trials),
            "dev_genuine": sum(int(x["label"]) for x in self.dev_trials),
        }


def wav_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_google(root: str | Path, out_dir: str | Path) -> dict:
    """Build an independent Google-v0.02 manifest; no Kaggle IDs or models are loaded.

    Exact SHA duplicates are detected. Lightly perturbed/renamed Kaggle audio cannot
    be proven independent by hash alone; do not interpret this as a contamination guarantee.
    """
    root, out_dir = Path(root).resolve(), Path(out_dir)
    records, seen = [], {}
    for path in sorted(root.rglob("*.wav")):
        if "_nohash_" not in path.stem:
            continue
        speaker = path.stem.split("_nohash_", 1)[0]
        word = path.parent.name
        digest = wav_sha256(path)
        with wave.open(str(path), "rb") as audio:
            rate, channels, nframes = (
                audio.getframerate(), audio.getnchannels(), audio.getnframes()
            )
        records.append({
            "file_id": path.relative_to(root).as_posix(),
            "speaker_id": speaker,
            "word": word,
            "sha256": digest,
            "exact_duplicate": int(digest in seen),
            "sample_rate": rate,
            "channels": channels,
            "frames": nframes,
        })
        seen.setdefault(digest, str(path))
    if not records:
        raise RuntimeError("No Google Speech Commands utterances found")
    fields = list(records[0])
    write_csv(out_dir / "google_manifest.csv", records, fields)
    summary = {
        "recordings": len(records),
        "exact_duplicates": sum(x["exact_duplicate"] for x in records),
        "unique_speakers": len({x["speaker_id"] for x in records}),
        "unique_words": dict(Counter(x["word"] for x in records if not x["exact_duplicate"])),
        "sample_rates": dict(Counter(x["sample_rate"] for x in records)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "google_inventory.json").write_text(json.dumps(summary, indent=2))
    return summary


def google_benchmark(
    manifest_file: str | Path, output: str | Path, seed: int = 2026,
    target_enrol: int = 3, target_test: int = 2, negatives_per_test: int = 8,
) -> dict:
    """Deterministic stop-only trials with disjoint speaker splits and locked test.

    Requires at least five unique stop takes per target. Unused Google speakers join
    the UBM population. No Kaggle data are used in this research construction.
    """
    if negatives_per_test < 1:
        raise ValueError("Require at least one impostor claim per target test")
    manifest = [
        r for r in read_csv(manifest_file)
        if r["exact_duplicate"] == "0" and r["sample_rate"] == "16000"
        and r["channels"] == "1"
    ]
    by_stop: dict[str, list[str]] = defaultdict(list)
    by_all: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        by_all[row["speaker_id"]].append(row["file_id"])
        if row["word"] == "stop":
            by_stop[row["speaker_id"]].append(row["file_id"])
    eligible = sorted(s for s, files in by_stop.items() if len(files) >= target_enrol + target_test)
    if len(eligible) < 30:
        raise ValueError("Not enough eligible stop speakers for a speaker-disjoint evaluation")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    a, b = int(len(eligible) * 0.7), int(len(eligible) * 0.85)
    eligible_set = set(eligible)
    split = {speaker: "background" for speaker in by_all}
    for speaker in eligible[a:b]:
        split[speaker] = "dev"
    for speaker in eligible[b:]:
        split[speaker] = "test"
    output = Path(output)
    write_csv(output / "google_speaker_splits.csv",
              [{"speaker_id": k, "split": v} for k, v in sorted(split.items())],
              ["speaker_id", "split"])
    bg_stop, bg_all = [], []
    for speaker, files in by_all.items():
        if split[speaker] == "background":
            bg_all.extend(files)
            bg_stop.extend(by_stop.get(speaker, []))
    write_csv(output / "google_ubm_stop.csv", [{"file_id": f} for f in bg_stop], ["file_id"])
    write_csv(output / "google_ubm_all_words.csv", [{"file_id": f} for f in bg_all], ["file_id"])
    summary = {"seed": seed, "eligible_stop_speakers": len(eligible), "ubm_stop": len(bg_stop),
               "ubm_all_words": len(bg_all), "research_only": True}
    for partition in ("dev", "test"):
        targets = sorted(s for s in eligible if split[s] == partition)
        enrol, trials = [], []
        for speaker in targets:
            files = sorted(by_stop[speaker])
            enrol.extend({"speaker_id": speaker, "file_id": f, "enrol_index": i}
                         for i, f in enumerate(files[:target_enrol]))
            for audio in files[target_enrol:target_enrol + target_test]:
                trials.append({"claimed_id": speaker, "test_file": audio, "label": 1})
                for claimed in rng.sample([s for s in targets if s != speaker],
                                          min(negatives_per_test, len(targets) - 1)):
                    trials.append({"claimed_id": claimed, "test_file": audio, "label": 0})
        # Unenrolled stop speakers from this partition are possible only when fewer than
        # the required number of takes exist. They are deliberately not used here.
        rng.shuffle(trials)
        for i, row in enumerate(trials):
            row["trial_id"] = f"{partition}_{i:08d}"
        write_csv(output / f"google_{partition}_enrol.csv", enrol,
                  ["speaker_id", "file_id", "enrol_index"])
        write_csv(output / f"google_{partition}_trials.csv", trials,
                  ["trial_id", "claimed_id", "test_file", "label"])
        summary[partition] = {"target_speakers": len(targets), "trials": len(trials),
                              "genuine": sum(x["label"] for x in trials)}
    (output / "google_benchmark.json").write_text(json.dumps(summary, indent=2))
    return summary


class GoogleResearchData:
    """Separate WAV reader for the original corpus, never passed to Kaggle pipeline."""

    source = "google_research"

    def __init__(self, root: str | Path, manifest: str | Path):
        self.root = Path(root).resolve()
        self.records = read_csv(manifest)
        self.allowed_ids = frozenset(x["file_id"] for x in self.records)

    def load(self, file_id: str) -> torch.Tensor:
        if file_id not in self.allowed_ids:
            raise PermissionError(f"Not in research manifest: {file_id}")
        path = (self.root / file_id).resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError("Path escape")
        waveform, sr = decode_pcm16_wav(path.read_bytes())
        if sr != 16000:
            raise ValueError(f"Invalid research WAV: {file_id}")
        return waveform
