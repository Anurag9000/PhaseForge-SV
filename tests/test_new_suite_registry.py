"""Validate EVERY registered missing-method job statically before expensive GPU work."""
import json
from pathlib import Path

from phaseforge.engine import RunConfig
from phaseforge.new_architectures import MODEL_NAMES
from phaseforge.new_classical_runner import ClassicalRun
from phaseforge.new_keyword import KeywordRun
from phaseforge.new_neural_runner import NeuralRun
from phaseforge.new_suite import ALLOWED, _deep_set
from phaseforge.google_mathworks_reference import MathWorksResearch


REPO = Path(__file__).resolve().parents[1]


def test_all_new_research_jobs_have_valid_runnable_configs():
    manifest = json.loads((REPO / "configs/missing_research_suite.json").read_text())
    assert manifest["suite_family"] == "new_phase_research"
    jobs = manifest["jobs"]
    assert len(jobs) == 127
    assert len({job["id"] for job in jobs}) == len(jobs)
    counts = {}
    previous = set()
    for job in jobs:
        kind = job["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        assert kind in ALLOWED
        filename = REPO / job["config"]
        assert filename.is_file(), job
        config = json.loads(filename.read_text())
        for name, value in job.get("overrides", {}).items():
            _deep_set(config, name, value)
        if kind == "gmm_phase":
            RunConfig.from_dict(config).validate()
        elif kind == "neural_sv":
            NeuralRun.from_dict(config).validate()
        elif kind == "classical_sv":
            ClassicalRun.from_dict(config).validate()
        elif kind == "keyword_google":
            assert "google" in job.get("tracks", [])
            KeywordRun.from_dict(config).validate()
        elif kind == "mathworks_reference":
            assert "google" in job.get("tracks", [])
            MathWorksResearch.from_dict(config).reference.validate()
        else:
            assert kind == "calibrate"
            assert job["source_job"] in previous
            assert config["folds"] >= 2
        previous.add(job["id"])
    assert counts["neural_sv"] >= 65
    assert counts["classical_sv"] >= 11
    assert counts["gmm_phase"] >= 21
    assert counts["keyword_google"] >= 14
    assert counts["mathworks_reference"] >= 7


def test_published_family_labels_are_not_aliases_of_only_one_network():
    from phaseforge.new_architectures import PhaseSpeakerEncoder, Architecture
    architectures = ["bcnn", "bcnn_2d", "xvector", "ecapa", "kwt",
                     "conformer", "ds_cnn", "tc_resnet", "crnn"]
    all_types = set()
    for architecture in architectures:
        model = PhaseSpeakerEncoder(
            8, Architecture(name=architecture, channels=16,
                            embedding=8, depth=1, heads=2)
        )
        all_types.add(type(model.blocks[0]).__name__
                      if len(model.blocks) else type(model.sequence).__name__)
    assert len(all_types) >= 6



def test_matched_seed_expansion_is_complete_and_calibrations_do_not_cross_seeds():
    import importlib.util

    source = REPO / "scripts/expand_missing_replicates.py"
    spec = importlib.util.spec_from_file_location("phaseforge_replicates", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = json.loads((REPO / "configs/missing_research_suite.json").read_text())
    expanded = module.expand_replicates(registry, [2026, 3107, 4109])
    base = len(registry["jobs"])
    assert len(expanded["jobs"]) == base * 3
    assert len({j["id"] for j in expanded["jobs"]}) == len(expanded["jobs"])

    ids = set()
    for job in expanded["jobs"]:
        seed = job["replication_seed"]
        assert job["id"].endswith(f"__s{seed}")
        if job["kind"] == "gmm_phase":
            assert job["overrides"]["seeds"] == [seed]
        elif job["kind"] == "calibrate":
            assert job["source_job"].endswith(f"__s{seed}")
            assert job["source_job"] in ids
        else:
            assert job["overrides"]["seed"] == seed
        ids.add(job["id"])
