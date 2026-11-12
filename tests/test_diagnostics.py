import torch

from phaseforge.diagnostics import (
    impostor_condition_report, paired_group_bootstrap,
    per_speaker_diagnostics,
)


def test_speaker_errors_and_impostor_split():
    trials = []
    scores = []
    for i in range(3):
        file = f"target{i}"
        trials.extend([
            {"claimed_id": f"s{i}", "test_file": file, "label": 1},
            {"claimed_id": f"s{(i+1)%3}", "test_file": file, "label": 0},
        ])
        scores.extend([3.0, -1.0])
    for i in range(4):
        trials.append({"claimed_id": f"s{i%3}", "test_file": f"unknown{i}", "label": 0})
        scores.append(-2.0)
    z = torch.tensor(scores)
    per, summary = per_speaker_diagnostics(trials, z, 0)
    assert summary["speakers"] == 3
    assert summary["speakers_with_any_error"] == 0
    parts = impostor_condition_report(trials, z)
    assert parts["enrolled_impostor"]["impostor"] == 3
    assert parts["unknown_impostor"]["impostor"] == 4
    assert parts["enrolled_impostor"]["eer"] == 0
    res = paired_group_bootstrap(trials, z, z, repetitions=20)
    assert res["ci95"] == [0.0, 0.0]
