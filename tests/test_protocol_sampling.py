"""Protocol-correct finite-background training under realistic speaker distributions."""
from collections import Counter

import pytest

from phaseforge.sampling import speaker_aware_subset, passphrase_aware_subset


def _rows(speakers=50, takes=5):
    return [
        {"speaker_id":f"s{s:03d}", "file_id":f"s{s:03d}_t{t}"}
        for s in range(speakers) for t in range(takes)
    ]


def test_bounded_ge2e_preserves_three_repetitions_and_is_deterministic():
    background = _rows(80, 5)
    a = speaker_aware_subset(background, 60, min_takes=3,
                             min_speakers=8, seed=2026)
    b = speaker_aware_subset(background, 60, min_takes=3,
                             min_speakers=8, seed=2026)
    counts = Counter(x["speaker_id"] for x in a)
    assert len(a) == 60
    assert len(counts) >= 8
    assert all(n >= 3 for n in counts.values())
    assert a == b
    assert {x["file_id"] for x in a} <= {x["file_id"] for x in background}


def test_plda_preserves_two_repetitions():
    rows = _rows(35, 4)
    out = speaker_aware_subset(rows, 25, min_takes=2,
                               min_speakers=2, seed=3)
    counts = Counter(x["speaker_id"] for x in out)
    assert len(out) == 25
    assert min(counts.values()) >= 2


def test_never_silent_when_budget_cannot_hold_speaker_episode():
    with pytest.raises(ValueError, match="cannot contain"):
        speaker_aware_subset(_rows(), 5, min_takes=3, min_speakers=2)
    with pytest.raises(ValueError, match="Only"):
        speaker_aware_subset(_rows(2, 2), 20, min_takes=3, min_speakers=2)


def test_google_passphrase_budget_reserves_stop_word():
    rows = _rows(40, 5)
    words = {r["file_id"]: "stop" if r["file_id"].endswith(("t0", "t1")) else "go"
             for r in rows}
    subset = passphrase_aware_subset(rows, words, 24, seed=5)
    selected_speakers = {r["speaker_id"] for r in subset}
    assert len(subset) <= 24
    assert len(selected_speakers) >= 2
    assert sum(words[r["file_id"]] == "stop" for r in subset) >= 12
    assert all(r["speaker_id"] in selected_speakers for r in subset)


def test_no_cap_preserves_original_input_unchanged():
    rows = _rows(5, 1)
    assert speaker_aware_subset(rows, None, min_takes=3,
                                min_speakers=8) == rows
    assert passphrase_aware_subset(rows, {}, None) == rows
