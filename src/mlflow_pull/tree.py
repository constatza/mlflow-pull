"""Run-tree data and the pure functions over it: grouping, root selection, flattening."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mlflow.entities import Run

from mlflow_pull.paths import safe_component

PARENT_RUN_TAG = "mlflow.parentRunId"
RUN_NAME_FALLBACK = "run"
SHORT_ID_LENGTH = 8

Children = dict[str | None, list[Run]]


@dataclass(frozen=True, slots=True)
class RunTarget:
    """A run and the directory its artifacts go to."""

    run: Run
    dir: Path


@dataclass(frozen=True, slots=True)
class Skipped:
    """A run left out because its status is not allowed; its nested runs are left out too.

    ``target`` holds where the run would have gone, so a report can still name it.
    """

    target: RunTarget
    status: str

    @property
    def label(self) -> str:
        return self.target.dir.name


@dataclass(frozen=True, slots=True)
class Resolution:
    """What a command will download: one tree per root run (parents before children)."""

    root: Path
    trees: tuple[tuple[RunTarget, ...], ...]
    skipped: tuple[Skipped, ...]

    @property
    def run_count(self) -> int:
        return sum(len(tree) for tree in self.trees)


def dir_name(run: Run) -> str:
    return f"{safe_component(run.info.run_name or RUN_NAME_FALLBACK)}-{run.info.run_id[:SHORT_ID_LENGTH]}"


def nested_run_names(targets: Iterable[RunTarget]) -> dict[Path, frozenset[str]]:
    """Per directory, the names of the run directories directly inside it; absent: none."""
    names: defaultdict[Path, set[str]] = defaultdict(set)
    for target in targets:
        names[target.dir.parent].add(target.dir.name)
    return {directory: frozenset(found) for directory, found in names.items()}


def children_by_parent(runs: Iterable[Run]) -> Children:
    """Run -> children map; ``None`` keys the top-level runs."""
    children: defaultdict[str | None, list[Run]] = defaultdict(list)
    for run in runs:
        children[run.data.tags.get(PARENT_RUN_TAG)].append(run)
    return dict(children)


def matched_roots(children: Children, matches: list[Run]) -> list[Run]:
    """Matched runs that have no matched ancestor, so no subtree is downloaded twice."""
    by_id = {run.info.run_id: run for runs in children.values() for run in runs}
    matched = {run.info.run_id for run in matches}

    def has_matched_ancestor(run: Run) -> bool:
        parent = run.data.tags.get(PARENT_RUN_TAG)
        seen: set[str] = set()  # the parent tag is user-settable, so a cycle is possible
        while parent is not None and parent not in seen:
            if parent in matched:
                return True
            seen.add(parent)
            parent = by_id[parent].data.tags.get(PARENT_RUN_TAG) if parent in by_id else None
        return False

    return [by_id.get(r.info.run_id, r) for r in matches if not has_matched_ancestor(r)]


def flatten(
    run: Run,
    children: Children,
    parent_dir: Path,
    allowed: frozenset[str],
    ancestors: frozenset[str] = frozenset(),
) -> tuple[list[RunTarget], list[Skipped]]:
    """``run`` and its allowed descendants, parents first; a refused run hides its subtree.

    ``ancestors`` cuts a ``mlflow.parentRunId`` cycle (the tag is user-settable).
    """
    target = RunTarget(run, parent_dir / dir_name(run))
    if run.info.status not in allowed:
        return [], [Skipped(target, run.info.status)]
    targets, skipped = [target], []
    below = ancestors | {run.info.run_id}
    for child in children.get(run.info.run_id, []):
        if child.info.run_id in below:
            continue
        child_targets, child_skipped = flatten(child, children, target.dir, allowed, below)
        targets += child_targets
        skipped += child_skipped
    return targets, skipped
