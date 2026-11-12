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
        self._zip_index: dict[str, str] | None = None

    def assert_background_only(self, ids: Iterable[str]) -> None:
        ids = set(ids)
        extraneous = ids - self.background_ids
        if extraneous:
            raise ValueError(f"Non-background training audio forbidden: {sorted(extraneous)[:4]}")

    def _open_zip_index(self) -> dict[str, str]:
        """Discover allowed WAVs by stable file ID, independent of ZIP subfolder layout."""
        if self._zip_index is not None:
            return self._zip_index
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.audio_zip)
        found = {}
        for member in self._zip.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".wav"):
                continue
            ident = Path(member.filename).stem
            if ident in self.allowed_ids:
                if ident in found:
                    raise ValueError(f"Duplicate archive WAV identifier: {ident}")
                found[ident] = member.filename
        missing = self.allowed_ids - found.keys()
        if missing:
            raise FileNotFoundError(
                f"Competition archive missing {len(missing)} manifest WAVs: "
                f"{sorted(missing)[:5]}"
            )
        self._zip_index = found
        return found

    def audit_audio(self, check_crc: bool = False) -> dict:
        """Deep manifest/archive audit; checks all WAV headers and exact PCM hashes.

        This audits structure and integrity, *not* independence from Google
        Speech Commands, which may contain lightly perturbed source recordings.
        """
        entries = self._open_zip_index() if self.audio_zip.is_file() else None
        hashes: dict[str, str] = {}
        duplicates = []
        formats = Counter()
        min_frames = None
        max_frames = 0
        for ident in sorted(self.allowed_ids):
            if entries is not None:
                raw = self._zip.read(entries[ident])
            else:
                path = self.audio_dir / f"{ident}.wav"
                if not path.is_file():
                    raise FileNotFoundError(path)
                raw = path.read_bytes()
            with wave.open(io.BytesIO(raw), "rb") as wav:
                shape = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
                frames = wav.getnframes()
                if shape != (16000, 1, 2):
                    raise ValueError(f"Invalid WAV format {ident}: {shape}")
                pcm = wav.readframes(frames)
            formats[str(shape)] += 1
            min_frames = frames if min_frames is None else min(min_frames, frames)
            max_frames = max(max_frames, frames)
            digest = hashlib.sha256(pcm).hexdigest()
            if digest in hashes:
                duplicates.append({"file_id": ident, "other_file_id": hashes[digest]})
            hashes.setdefault(digest, ident)
        if check_crc and self._zip is not None:
            failure = self._zip.testzip()
            if failure:
                raise ValueError(f"ZIP CRC check failed: {failure}")
        return {
            **self.summary(),
            "audited_files": len(self.allowed_ids),
            "exact_pcm_duplicate_pairs": duplicates,
            "format_counts": dict(formats),
            "min_frames": min_frames,
            "max_frames": max_frames,
            "zip_crc_verified": check_crc and self.audio_zip.is_file(),
            "extra_archive_wavs": (
                sum(member.filename.lower().endswith(".wav") for member in self._zip.infolist())
                - len(entries)
                if entries is not None else None
            ),
        }

    def load(self, file_id: str) -> torch.Tensor:
        if file_id not in self.allowed_ids:
            raise PermissionError("Audio ID is not in the Kaggle competition manifests")
        if self.audio_zip.is_file():
            member = self._open_zip_index()[file_id]
            content = self._zip.read(member)
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
            self._zip_index = None

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
    distinct_stops = Counter(
        x["speaker_id"] for x in records
        if x["word"] == "stop" and not x["exact_duplicate"]
    )
    summary = {
        "recordings": len(records),
        "exact_duplicates": sum(x["exact_duplicate"] for x in records),
        "unique_speakers": len({x["speaker_id"] for x in records}),
        "unique_words": dict(Counter(x["word"] for x in records if not x["exact_duplicate"])),
        "sample_rates": dict(Counter(x["sample_rate"] for x in records)),
        "stop_takes_per_speaker": dict(sorted(
            Counter(distinct_stops.values()).items()
        )),
        "eligible_stop_speakers_by_take_protocol": {
            f"{enrol}_enrol_{test}_test":
                sum(count >= enrol + test for count in distinct_stops.values())
            for enrol, test in ((1, 1), (2, 1), (3, 1), (3, 2))
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "google_inventory.json").write_text(json.dumps(summary, indent=2))
    return summary


def google_benchmark(
    manifest_file: str | Path, output: str | Path, seed: int = 2026,
    target_enrol: int = 1, target_test: int = 1, negatives_per_test: int = 8,
    minimum_targets: int = 30,
) -> dict:
    """Speaker-disjoint stop-only research verification with explicit take counts.

    Google recordings commonly have fewer repeated stop takes than the Kaggle
    task. Default 1+1 permits broader speaker coverage; 2+1, 3+1 or 3+2 may
    be specified explicitly when enough recordings exist. All evaluations
    must report the chosen protocol. No Google data train Kaggle UBMs.
    """
    if not 1 <= target_enrol <= 3 or target_test < 1:
        raise ValueError("Require 1-3 enrolment takes and at least one heldout take")
    if negatives_per_test < 1 or minimum_targets < 5:
        raise ValueError("Require impostor count >0 and at least five target speakers")
    manifest_sha256 = hashlib.sha256(Path(manifest_file).read_bytes()).hexdigest()
    locked = Path(output) / "google_benchmark.json"
    if locked.exists():
        previous = json.loads(locked.read_text())
        expected = {
            "seed": seed,
            "enrolment_takes_per_speaker": target_enrol,
            "genuine_test_takes_per_speaker": target_test,
            "wrong_enrolled_claims_per_test": negatives_per_test,
            "minimum_target_speakers": minimum_targets,
        "source_manifest_sha256": manifest_sha256,
            "source_manifest_sha256": manifest_sha256,
        }
        if any(previous.get(k) != v for k, v in expected.items()):
            raise PermissionError(
                "Locked Google benchmark already exists with different speaker "
                "partition, trial protocol or source manifest; use a new output "
                "directory instead of overwriting the held-out population."
            )
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
    if len(eligible) < minimum_targets:
        raise ValueError(
            f"Only {len(eligible)} speakers have {target_enrol}+{target_test} "
            f"distinct stop takes, below the required {minimum_targets}; "
            "reduce enrol/test counts explicitly or inspect the inventory."
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    # Split eligible enrolled speakers and low-take/unknown speakers separately.
    # Every word from a given original speaker remains in exactly one partition.
    leftovers = sorted(set(by_all) - set(eligible))
    rng.shuffle(leftovers)
    split = {}
    for population in (eligible, leftovers):
        end_bg = int(len(population) * .70)
        end_dev = int(len(population) * .85)
        for speaker in population[:end_bg]:
            split[speaker] = "background"
        for speaker in population[end_bg:end_dev]:
            split[speaker] = "dev"
        for speaker in population[end_dev:]:
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
    summary = {
        "seed": seed, "eligible_stop_speakers": len(eligible),
        "enrolment_takes_per_speaker": target_enrol,
        "genuine_test_takes_per_speaker": target_test,
        "wrong_enrolled_claims_per_test": negatives_per_test,
        "minimum_target_speakers": minimum_targets,
        "background_speakers": sum(v == "background" for v in split.values()),
        "ubm_stop": len(bg_stop), "ubm_all_words": len(bg_all),
        "research_only": True,
        "independent_from_kaggle": "not established: shared source corpus",
    }
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
        # Include actual unseen impostors whose speakers have no enrolment model.
        unknown_audio = [
            file for speaker in sorted(by_all)
            if split[speaker] == partition and speaker not in targets
            for file in sorted(by_stop.get(speaker, []))[:2]
        ]
        rng.shuffle(unknown_audio)
        genuine_count = len(targets) * target_test
        needed_files = (5 * genuine_count + 3) // 4
        for test_file in unknown_audio[:needed_files]:
            for claimed in rng.sample(targets, min(4, len(targets))):
                trials.append({"claimed_id": claimed, "test_file": test_file, "label": 0})
        rng.shuffle(trials)
        for i, row in enumerate(trials):
            row["trial_id"] = f"{partition}_{i:08d}"
        write_csv(output / f"google_{partition}_enrol.csv", enrol,
                  ["speaker_id", "file_id", "enrol_index"])
        write_csv(output / f"google_{partition}_trials.csv", trials,
                  ["trial_id", "claimed_id", "test_file", "label"])
        summary[partition] = {
            "target_speakers": len(targets),
            "trials": len(trials),
            "genuine": sum(x["label"] for x in trials),
            "impostor": sum(x["label"] == 0 for x in trials),
            "unknown_source_impostor_recordings": min(len(unknown_audio), needed_files),
        }
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
