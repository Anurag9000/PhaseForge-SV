"""PhaseForge command line: audit, train, sweep, fuse, research split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import KaggleData, google_benchmark, inventory_google
from .engine import RunConfig, run
from .experiments import fuse_saved_runs, sweep


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="phaseforge")
    commands = parser.add_subparsers(dest="action", required=True)
    audit = commands.add_parser("audit-kaggle", help="Check competition manifests and source boundaries")
    audit.add_argument("--data", required=True)
    inv = commands.add_parser("audit-google", help="Inventory original Google WAVs (research only)")
    inv.add_argument("--data", required=True)
    inv.add_argument("--output", required=True)
    split = commands.add_parser("split-google", help="Lock speaker-disjoint Google benchmark")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=2026)
    for name in ("run", "sweep"):
        p = commands.add_parser(name, help=f"Execute or resume {name}")
        p.add_argument("--data", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--track", choices=["kaggle", "google"], default="kaggle")
        p.add_argument("--benchmark", default=None)
        p.add_argument("--ubm-words", choices=["stop", "all"], default="stop")
    fusion = commands.add_parser("fuse", help="Dev-fitted scale fusion across trained runs")
    fusion.add_argument("--runs", nargs="+", required=True)
    fusion.add_argument("--output", required=True)
    fusion.add_argument("--weights", nargs="+", type=float)
    fusion.add_argument("--no-eval", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "audit-kaggle":
        with KaggleData(args.data) as dataset:
            print(json.dumps(dataset.summary(), indent=2))
    elif args.action == "audit-google":
        print(json.dumps(inventory_google(args.data, args.output), indent=2))
    elif args.action == "split-google":
        print(json.dumps(google_benchmark(args.manifest, args.output, args.seed), indent=2))
    elif args.action == "run":
        config = RunConfig.from_dict(json.loads(Path(args.config).read_text()))
        result = run(args.data, config, args.output, args.track, args.benchmark,
                     args.ubm_words)
        print(json.dumps(result["metrics"], indent=2))
        print("Files:", Path(args.output).resolve())
    elif args.action == "sweep":
        config = json.loads(Path(args.config).read_text())
        print(json.dumps(sweep(args.data, config, args.output,
                               args.track, args.benchmark), indent=2))
    elif args.action == "fuse":
        print(json.dumps(fuse_saved_runs(args.runs, args.output, args.weights,
                                        not args.no_eval), indent=2))


if __name__ == "__main__":
    main()
