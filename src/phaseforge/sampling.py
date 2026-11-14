"""Deterministic, repeat-preserving background subsampling for speaker research.

Naive random recording caps destroy GE2E/PLDA episodes when each background
identity has several recordings; select identities first, then takes.
Never draws from development enrolment/trials or external Google data.
"""
from __future__ import annotations

import random
from collections import defaultdict


def speaker_aware_subset(
    rows: list[dict], max_files: int | None,
    min_takes: int = 2, min_speakers: int = 2, seed: int = 2026,
) -> list[dict]:
    """Select at most max_files with at least min_takes per selected speaker.

    If no cap is imposed, return original rows without filtering. With a cap,
    discard ineligible speakers, sample whole speaker groups and then fill any
    remaining capacity from the selected speakers. Preserves input row order.
    """
    if min_takes < 1 or min_speakers < 1:
        raise ValueError("min_takes and min_speakers must be positive")
    if max_files is None:
        return list(rows)
    if max_files < min_takes * min_speakers:
        raise ValueError(
            f"Background cap {max_files} cannot contain {min_speakers} "
            f"speakers with {min_takes} recordings each"
        )
    groups = defaultdict(list)
    seen_files = set()
    for row in rows:
        file_id, speaker = row["file_id"], row["speaker_id"]
        if file_id in seen_files:
            raise ValueError(f"Duplicate background file ID: {file_id}")
        seen_files.add(file_id)
        groups[speaker].append(file_id)
    eligible = sorted(s for s, files in groups.items()
                      if len(files) >= min_takes)
    if len(eligible) < min_speakers:
        raise ValueError(
            f"Only {len(eligible)} speakers have >= {min_takes} recordings; "
            f"need at least {min_speakers}"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    count = min(len(eligible), max_files // min_takes)
    chosen = eligible[:count]
    selected = set()
    for speaker in chosen:
        selected.update(rng.sample(sorted(groups[speaker]), min_takes))
    # Fill leftover capacity using DIFFERENT recordings from already chosen
    # speakers. Do not add a new speaker with fewer than min_takes.
    remainder = [file for speaker in chosen
                 for file in sorted(groups[speaker]) if file not in selected]
    rng.shuffle(remainder)
    selected.update(remainder[:max(0, max_files - len(selected))])
    return [row for row in rows if row["file_id"] in selected]
