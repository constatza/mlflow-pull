"""Download the artifacts of an MLflow run, its nested runs, or a whole experiment.

The tracking server comes from ``MLFLOW_TRACKING_URI``, like the mlflow CLI.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

PARENT_RUN_TAG = "mlflow.parentRunId"
DEFAULT_OUT = Path("mlflow-downloads")
SEARCH_PAGE_SIZE = 50_000  # MLflow's server-side maximum per search call
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def _dir_name(run: Run) -> str:
    name = _UNSAFE_PATH_CHARS.sub("_", run.info.run_name or "run")
    return f"{name}-{run.info.run_id[:8]}"


def _children_by_parent(client: MlflowClient, experiment_id: str) -> dict[str | None, list[Run]]:
    """One search for the whole experiment; run -> children map built in memory."""
    runs = client.search_runs([experiment_id], max_results=SEARCH_PAGE_SIZE)
    children: dict[str | None, list[Run]] = defaultdict(list)
    for run in runs:
        children[run.data.tags.get(PARENT_RUN_TAG)].append(run)
    return children


def _download(
    client: MlflowClient, run: Run, children: dict[str | None, list[Run]], parent_dir: Path
) -> int:
    """Download ``run`` and its descendants under ``parent_dir``; return the run count."""
    run_dir = parent_dir / _dir_name(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(run.info.run_id, "", str(run_dir))
    return 1 + sum(
        _download(client, child, children, run_dir) for child in children[run.info.run_id]
    )


def download_run_tree(client: MlflowClient, run_id: str, out: Path) -> int:
    """Download a run and all its nested runs; return how many runs were downloaded."""
    run = client.get_run(run_id)
    children = _children_by_parent(client, run.info.experiment_id)
    return _download(client, run, children, out)


def download_experiment(client: MlflowClient, name_or_id: str, out: Path) -> int:
    """Download every top-level run of an experiment with its nested runs."""
    experiment = client.get_experiment_by_name(name_or_id) or client.get_experiment(name_or_id)
    children = _children_by_parent(client, experiment.experiment_id)
    experiment_dir = out / _UNSAFE_PATH_CHARS.sub("_", experiment.name)
    return sum(_download(client, run, children, experiment_dir) for run in children[None])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlpull", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="destination directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", parents=[common], help="a run and all its nested runs").add_argument(
        "run_id"
    )
    sub.add_parser("experiment", parents=[common], help="every run of an experiment").add_argument(
        "experiment"
    )
    args = parser.parse_args(argv)

    client = MlflowClient()
    if args.command == "run":
        count = download_run_tree(client, args.run_id, args.out)
    else:
        count = download_experiment(client, args.experiment, args.out)
    print(f"Downloaded {count} run(s) to {args.out.resolve()}")
