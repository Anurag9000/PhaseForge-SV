from types import SimpleNamespace
from phaseforge.studies import proxy_manifests
from phaseforge.experiments import expand_grid


def test_speaker_disjoint_proxy_and_unenrolled_impostors():
    rows = [{"speaker_id": f"s{i}", "file_id": f"f{i}_{j}"}
            for i in range(12) for j in range(4)]
    result = proxy_manifests(
        SimpleNamespace(background=rows), seed=52,
        maximum_targets=4, maximum_impostors=3
    )
    background = {x["speaker_id"] for x in result["background"]}
    enrolled = {x["speaker_id"] for x in result["dev_enrol"]}
    assert background.isdisjoint(enrolled)
    assert len(enrolled) == 4
    assert len(result["dev_trials"]) == 28
    assert {x["test_file"] for x in result["dev_trials"]}.isdisjoint(
        {x["file_id"] for x in result["dev_enrol"]}
    )
    assert len({x["trial_id"] for x in result["dev_trials"]}) == 28


def test_huge_grid_sampling_bounded_and_deterministic():
    grid = {"a": list(range(200)), "b": list(range(200)), "c": list(range(200))}
    a = expand_grid(grid, max_runs=19, strategy="random", seed=47)
    b = expand_grid(grid, max_runs=19, strategy="random", seed=47)
    assert a == b
    assert len(a) == 19
    assert len({tuple(row.values()) for row in a}) == 19


def test_small_grid_enumeration():
    rows = expand_grid({"a": [0, 1], "b": ["x", "y"]},
                       max_runs=100, strategy="grid")
    assert rows == [
        {"a": 0, "b": "x"}, {"a": 0, "b": "y"},
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"},
    ]
