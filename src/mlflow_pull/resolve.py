"""Decide what to download. Reads through the gateway; writes nothing."""

from __future__ import annotations

from pathlib import Path

from mlflow.entities import Experiment, Run

from mlflow_pull.paths import safe_component
from mlflow_pull.ports import RunGateway
from mlflow_pull.selection import FINISHED, RunStatusError, quote
from mlflow_pull.tree import (
    Children,
    Resolution,
    flatten,
    children_by_parent,
    matched_roots,
)

ALL_EXPERIMENTS = ""  # an empty MLflow filter lists every active experiment



class NoMatchError(ValueError):
    """A selection that is well formed but names nothing that exists, such as a glob over experiment names."""


Root = tuple[Run, Children, Path]  # a root run, its experiment's run tree, its parent directory


def _assemble(out: Path, allowed: frozenset[str], roots: list[Root]) -> Resolution:
    flattened = [flatten(run, children, parent_dir, allowed) for run, children, parent_dir in roots]
    return Resolution(
        root=out,
        trees=tuple(tuple(targets) for targets, _ in flattened),
        skipped=tuple(skip for _, skipped in flattened for skip in skipped),
    )


def resolve_runs(
    gateway: RunGateway, run_ids: list[str], out: Path, allowed: frozenset[str] = FINISHED
) -> Resolution:
    """Named runs with their nested runs; a named run outside ``allowed`` raises ``RunStatusError``."""
    runs = [gateway.get_run(run_id) for run_id in run_ids]
    refused = [f"{r.info.run_id} is {r.info.status}" for r in runs if r.info.status not in allowed]
    if refused:
        raise RunStatusError("; ".join(refused))
    trees = {
        eid: children_by_parent(gateway.search_runs(eid)) for eid in {r.info.experiment_id for r in runs}
    }
    return _assemble(out, allowed, [(r, trees[r.info.experiment_id], out) for r in runs])


def _find_experiments(gateway: RunGateway, names_or_ids: list[str]) -> list[Experiment]:
    """Names, ids or ``*`` globs over names; each experiment once, in argument order."""
    found: dict[str, Experiment] = {}
    for name_or_id in names_or_ids:
        if "*" in name_or_id:
            # shortcut: "_" and "%" in the pattern act as LIKE wildcards too; harmless for globs.
            matches = gateway.search_experiments(f"name LIKE {quote(name_or_id.replace('*', '%'))}")
            if not matches:
                raise NoMatchError(f"no experiment matches {name_or_id}")
        else:
            matches = [gateway.find_experiment(name_or_id)]
        found |= {e.experiment_id: e for e in matches}
    return list(found.values())


def resolve_experiments(
    gateway: RunGateway,
    names_or_ids: list[str],
    out: Path,
    allowed: frozenset[str] = FINISHED,
    filter_string: str | None = None,
) -> Resolution:
    """Runs of each experiment with their nested runs.

    Without ``filter_string`` the top-level runs are the roots; with it, every
    matching run at any depth is, minus matches nested under another match.
    """
    roots: list[Root] = []
    for experiment in _find_experiments(gateway, names_or_ids):
        children = children_by_parent(gateway.search_runs(experiment.experiment_id))
        if filter_string is None:
            top_level = children.get(None, [])
        else:
            matches = gateway.search_runs(experiment.experiment_id, filter_string)
            top_level = matched_roots(children, matches)
        experiment_dir = out / safe_component(experiment.name)
        roots += [(run, children, experiment_dir) for run in top_level]
    return _assemble(out, allowed, roots)


def resolve_run_list(
    gateway: RunGateway, names_or_ids: list[str], filter_string: str | None = None
) -> list[Run]:
    """Every run matching ``filter_string``, at any depth, in the named experiments (all when none)."""
    experiments = (
        _find_experiments(gateway, names_or_ids)
        if names_or_ids
        else gateway.search_experiments(ALL_EXPERIMENTS)
    )
    return [run for e in experiments for run in gateway.search_runs(e.experiment_id, filter_string)]
