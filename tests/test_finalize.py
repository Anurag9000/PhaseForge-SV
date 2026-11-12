import csv
import json
from pathlib import Path

import pytest

from phaseforge.finalize import finalize_kaggle


def _csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def fake_completed_run(tmp_path):
    data = tmp_path / "competition"
    data.mkdir()
    bg = [{"speaker_id": "bg", "file_id": "b0"}]
    dev = [{"speaker_id": "dev", "file_id": f"e{i}", "enrol_index": str(i)}
           for i in range(3)]
    ev = [{"speaker_id": "eval", "file_id": f"v{i}", "enrol_index": str(i)}
          for i in range(3)]
    trials = [
        {"trial_id": "d1", "claimed_id": "dev", "test_file": "dt1", "label": "1"},
        {"trial_id": "d2", "claimed_id": "dev", "test_file": "dt2", "label": "0"},
    ]
    etrials = [
        {"trial_id": "t1", "claimed_id": "eval", "test_file": "tt1"},
        {"trial_id": "t2", "claimed_id": "eval", "test_file": "tt2"},
    ]
    _csv(data / "background.csv", ["speaker_id","file_id"], bg)
    _csv(data / "dev_enrol.csv", ["speaker_id","file_id","enrol_index"], dev)
    _csv(data / "eval_enrol.csv", ["speaker_id","file_id","enrol_index"], ev)
    _csv(data / "dev_trials.csv", ["trial_id","claimed_id","test_file","label"], trials)
    _csv(data / "eval_trials.csv", ["trial_id","claimed_id","test_file"], etrials)
    _csv(data / "sample_submission.csv", ["trial_id","score"],
         [{"trial_id": t["trial_id"],"score":"0.0"} for t in etrials])
    (data / "audio.zip").write_bytes(b"fake zip: never decoded by finalizer")
    location = tmp_path / "run"
    location.mkdir()
    (location / "result.json").write_text(json.dumps({
        "track": "kaggle", "research_only": False,
        "source": str(data.resolve()), "seed_diagnostics": [{"seed": 2026}],
        "config": {"feature": {"mode": "group"}, "em": {"mixtures": 2},
                   "score": {"norm": "none"}, "seeds": [2026]},
    }))
    _csv(location / "development_scores.csv",
         ["trial_id","claimed_id","test_file","label","score"],
         [{**t, "score": str(2 if t["label"] == "1" else -2)} for t in trials])
    _csv(location / "evaluation_scores.csv",
         ["trial_id","claimed_id","test_file","score"],
         [{**t, "score": str(1 if i == 0 else -1)}
          for i,t in enumerate(etrials)])
    return data, location


def test_finalize_single_run(fake_completed_run, tmp_path):
    data, run = fake_completed_run
    result = finalize_kaggle(data, [run], tmp_path / "final")
    assert result["dev_metrics"]["eer"] == 0.0
    assert result["evaluation_trials"] == 2
    assert (tmp_path / "final" / "submission.csv").read_text().startswith("trial_id,score")


def test_reject_google_research_run(fake_completed_run, tmp_path):
    data, run = fake_completed_run
    r = json.loads((run / "result.json").read_text())
    r["track"] = "google"
    r["research_only"] = True
    (run / "result.json").write_text(json.dumps(r))
    with pytest.raises(PermissionError):
        finalize_kaggle(data, [run], tmp_path / "final")


def test_reject_tampered_trials(fake_completed_run, tmp_path):
    data, run = fake_completed_run
    source = (run / "evaluation_scores.csv").read_text()
    (run / "evaluation_scores.csv").write_text(source.replace("t1,eval,tt1", "WRONG,eval,tt1"))
    with pytest.raises(ValueError):
        finalize_kaggle(data, [run], tmp_path / "final")
