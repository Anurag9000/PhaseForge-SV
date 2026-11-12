"""Optional Optuna/TPE Bayesian orchestration over native PyTorch SV experiments.

Only experiment scheduling uses Optuna; phase extraction, GMM EM, MAP and trial
scoring remain implemented in torch. The default validation objective is
speaker-excluded Kaggle background proxy EER, not an in-sample evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .engine import RunConfig, run
from .experiments import _deep_set
from .studies import run_proxy


def suggest_value(trial, name: str, definition):
    """Sample one search-space variable with conditional-valid dimensions upstream."""
    if isinstance(definition, list):
        if not definition:
            raise ValueError(f"Empty choice list: {name}")
        return trial.suggest_categorical(name, definition)
    if not isinstance(definition, dict):
        raise ValueError(f"Unsupported hyperparameter definition: {name}")
    kind = definition.get("type")
    if kind == "float":
        return trial.suggest_float(
            name, float(definition["low"]), float(definition["high"]),
            step=float(definition["step"]) if "step" in definition else None,
            log=bool(definition.get("log", False))
        )
    if kind == "int":
        return trial.suggest_int(
            name, int(definition["low"]), int(definition["high"]),
            step=int(definition.get("step", 1)),
            log=bool(definition.get("log", False))
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, definition["choices"])
    raise ValueError(f"Invalid hyperparameter type: {name}")


def bayesian_search(data_root: str, spec: dict, output: str) -> dict:
    """Resume a persistent TPE study and save all completed/failed experiment results."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional search package: pip install -e '.[search]'"
        ) from exc
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if spec.get("track", "kaggle") != "kaggle":
        raise ValueError("Bayesian competition search is Kaggle-background-only")
    validation = spec.get("validation", "proxy")
    if validation not in ("proxy", "development"):
        raise ValueError("Choose proxy or development validation")
    space = dict(spec["space"])
    base = copy.deepcopy(spec["base"])
    n_trials = int(spec.get("trials", 40))
    if n_trials < 1:
        raise ValueError("trials must be >=1")
    seed = int(spec.get("seed", 2026))
    rotations = int(spec.get("proxy_rotations", 1))
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=False)
    storage = "sqlite:///" + str((out / "optuna.db").resolve())
    study = optuna.create_study(
        direction="minimize", study_name="phaseforge-kaggle-background",
        sampler=sampler, storage=storage, load_if_exists=True,
    )

    def objective(trial):
        config = copy.deepcopy(base)
        for name in sorted(space):
            _deep_set(config, name, suggest_value(trial, name, space[name]))
        config["evaluate_eval"] = False
        config["save_model"] = False
        serial = json.dumps(config, sort_keys=True)
        digest = hashlib.sha256(serial.encode()).hexdigest()[:14]
        trial_dir = out / "runs" / digest
        trial_dir.mkdir(parents=True, exist_ok=True)
        result_path = trial_dir / "objective.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            value = float(result["validation_eer"])
        else:
            try:
                cfg = RunConfig.from_dict(config)
                if validation == "proxy":
                    outcome = run_proxy(
                        data_root, cfg, trial_dir, seed=seed, rotations=rotations
                    )
                    value = float(outcome["mean_eer"])
                else:
                    outcome = run(data_root, cfg, trial_dir, track="kaggle")
                    value = float(outcome["metrics"]["eer"])
                result = {
                    "trial_number": trial.number, "config": config,
                    "validation": validation, "validation_eer": value,
                    "source_track": "kaggle_background_only"
                }
                result_path.write_text(json.dumps(result, indent=2))
            except Exception as error:
                (trial_dir / "failure.json").write_text(json.dumps({
                    "config": config, "trial_number": trial.number,
                    "error": repr(error)
                }, indent=2))
                raise optuna.TrialPruned(f"Invalid or failed run: {error}") from error
        trial.set_user_attr("run_directory", str(trial_dir))
        return value

    study.optimize(objective, n_trials=n_trials)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        result = {"complete": 0, "failed_or_pruned": len(study.trials)}
    else:
        trial = study.best_trial
        result = {
            "complete": len(complete), "failed_or_pruned": len(study.trials) - len(complete),
            "best_validation_eer": trial.value, "best_params": trial.params,
            "best_run": trial.user_attrs.get("run_directory"),
            "study_name": study.study_name,
            "validation": validation,
        }
        best_config = copy.deepcopy(base)
        for k, v in trial.params.items():
            _deep_set(best_config, k, v)
        best_config["evaluate_eval"] = False
        (out / "best_config.json").write_text(json.dumps(best_config, indent=2))
    (out / "bayesian_summary.json").write_text(json.dumps(result, indent=2))
    return result
