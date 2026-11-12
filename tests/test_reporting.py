import csv
import json
from pathlib import Path

import pytest

from phaseforge.reporting import generate_report


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, name, source="/demo/source"):
    run = root / name
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "track": "kaggle", "source": source, "config": {"seeds": [2026]}
    }))
    scores = [
        {"trial_id": f"t{i}", "claimed_id": f"s{i//2}",
         "test_file": f"f{i//2}", "label": str(i%2),
         "score": str(1 if i%2 else -1)}
        for i in range(8)
    ]
    save_csv(run / "development_scores.csv", scores)
    return run


def test_completed_report_and_no_hidden_results(tmp_path):
    a = make_run(tmp_path, "A")
    b = make_run(tmp_path, "B")
    outcome = generate_report([a, b], tmp_path / "report", repetitions=15)
    assert len(outcome["comparison"]) == 2
    assert outcome["hidden_test_eer"] is None
    assert (tmp_path / "report" / "REPORT.md").exists()


def test_reject_different_data_sources(tmp_path):
    a = make_run(tmp_path, "A")
    b = make_run(tmp_path, "B", source="/another/source")
    with pytest.raises(ValueError):
        generate_report([a, b], tmp_path / "report", repetitions=3)
