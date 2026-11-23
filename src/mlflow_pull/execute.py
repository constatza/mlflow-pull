"""Plan and carry out a ``Resolution``, or gather run summaries. Effects: listings, directories, downloads, history reads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlflow.entities import Metric, Run
from mlflow.exceptions import MlflowException

from mlflow_pull.localfs import (
    prepare_destination,
    ensure_directory,
    read_small_text,
    stat_file,
    verified_download,
    write_atomically,
)
from mlflow_pull.paths import RelPath, UnsafePathError, require_within
from mlflow_pull.pathfilter import PathFilter
from mlflow_pull.plan import (
    MANIFEST_NAME,
    SUMMARY_NAME,
    UNRECORDED,
    Content,
    OnDiskFile,
    PullOptions,
    PullPlan,
    Recorded,
    RunOutcome,
    RunPlan,
    Unrecorded,
    outcome_of,
    parse_manifest,
    plan_run,
)
from mlflow_pull.ports import ArtifactDir, ArtifactFile, RunGateway
from mlflow_pull.render import record
from mlflow_pull.resolve import resolve_experiments, resolve_runs
from mlflow_pull.selection import FINISHED
from mlflow_pull.summary import Point, RunSummary, in_display_order, matching_keys, summarise_run, to_points
from mlflow_pull.tree import Resolution, RunTarget, nested_run_names

DEFAULT_JOBS = 4
MAX_DEPTH = 64  # directory levels below the run root; real runs rarely pass a dozen
MAX_ENTRIES = 200_000  # listed files and directories per run: bounds memory and requests against a hostile server


class ListingLimitError(ValueError):
    """A run's artifact listing is deeper or larger than any real run's: a broken or hostile server."""


def _list_files(gateway: RunGateway, run_id: str, path_filter: PathFilter) -> list[ArtifactFile]:
    """Walk the run's artifact listing, one call per directory, skipping directories no glob can reach.

    A directory is listed once however often it is named, and the walk stops at ``MAX_DEPTH`` levels
    and ``MAX_ENTRIES`` entries, so a server cannot keep it running or make it exponential.
    """
    files: list[ArtifactFile] = []
    pending: list[RelPath | None] = [None]
    walked: set[RelPath] = set()
    entries = 0
    while pending:
        directory = pending.pop()
        for entry in gateway.list_artifacts(run_id, directory):
            entries += 1
            if entries > MAX_ENTRIES:
                raise ListingLimitError(f"run {run_id}: the artifact listing has more than {MAX_ENTRIES} entries")
            if len(entry.path.parts) > MAX_DEPTH:
                raise ListingLimitError(f"run {run_id}: the artifact listing is deeper than {MAX_DEPTH} levels")
            if not entry.path.is_child_of(directory):  # also stops a server from looping the walk
                raise UnsafePathError(f"run {run_id}: {entry.path} is not directly inside {directory or 'the run root'}")
            match entry:
                case ArtifactFile():
                    files.append(entry)
                case ArtifactDir() if entry.path not in walked and path_filter.descends_into(entry.path):
                    walked.add(entry.path)
                    pending.append(entry.path)
                case ArtifactDir():
                    pass
    return files


def _read_manifest(directory: Path) -> Recorded:
    """The manifest of ``directory`` restricted to files still on disk with their recorded size and mtime."""
    text = read_small_text(directory / MANIFEST_NAME)
    recorded = UNRECORDED if text is None else parse_manifest(text)
    if isinstance(recorded, Unrecorded):
        return recorded
    on_disk: dict[RelPath, OnDiskFile] = {}
    for path in recorded.files:
        if (found := stat_file(directory, path)) is not None:
            on_disk[path] = found
    return recorded.restricted_to(on_disk)


def _plan_target(
    gateway: RunGateway, target: RunTarget, nested: frozenset[str], options: PullOptions
) -> RunPlan:
    content = options.content
    if content.artifacts:
        listing = _list_files(gateway, target.run.info.run_id, options.path_filter)
        recorded = _read_manifest(target.dir)
    else:
        listing, recorded = [], UNRECORDED
    return plan_run(target, listing, recorded, options.path_filter, content.reserved, nested)


def plan_pull(
    gateway: RunGateway, resolution: Resolution, options: PullOptions = PullOptions(), jobs: int = DEFAULT_JOBS
) -> PullPlan:
    """List every run's artifacts and diff them against what is on disk. Writes nothing.

    Every destination is checked before anything is read from it.
    """
    targets = [target for tree in resolution.trees for target in tree]
    for target in targets:
        require_within(resolution.root, target.dir)
    nested = nested_run_names(targets)
    with ThreadPoolExecutor(jobs) as pool:
        runs = tuple(pool.map(lambda target: _plan_target(gateway, target, nested.get(target.dir, frozenset()), options), targets))
    return PullPlan(options.content, resolution.root, runs)


def _fetch(gateway: RunGateway, root: Path, plan: RunPlan) -> Mapping[RelPath, OnDiskFile]:
    """Download each planned file into its run directory and return what landed, as found on disk."""
    run_id, directory = plan.target.run.info.run_id, plan.target.dir
    fetched: dict[RelPath, OnDiskFile] = {}
    for file in plan.files:
        prepare_destination(root, file.path.under(directory))
        gateway.download_artifact(run_id, file.path, directory)
        fetched[file.path] = verified_download(directory, file)
    return fetched


def _execute_run(gateway: RunGateway, root: Path, plan: RunPlan, content: Content) -> RunOutcome:
    run = plan.target.run
    directory = plan.target.dir
    if plan.files or content.metadata:
        ensure_directory(root, directory)
    fetched = _fetch(gateway, root, plan)
    if plan.files:  # after the files, so an interrupted pull never vouches for a missing one
        write_atomically(directory / MANIFEST_NAME, plan.record(fetched).to_json())
    if content.metadata:
        write_atomically(directory / SUMMARY_NAME, json.dumps(record(summarise_run(run, {})), indent=2))
    return outcome_of(plan, content)


def execute_plan(gateway: RunGateway, plan: PullPlan, jobs: int = DEFAULT_JOBS) -> tuple[RunOutcome, ...]:
    """Fetch exactly the planned files, one worker per run."""
    with ThreadPoolExecutor(jobs) as pool:
        return tuple(pool.map(lambda run: _execute_run(gateway, plan.root, run, plan.content), plan.runs))


def pull(
    gateway: RunGateway, resolution: Resolution, options: PullOptions = PullOptions(), jobs: int = DEFAULT_JOBS
) -> tuple[RunOutcome, ...]:
    """Plan, refuse when over ``options.max_bytes``, then execute; one outcome per run."""
    plan = plan_pull(gateway, resolution, options, jobs)
    plan.check_size(options.max_bytes)
    return execute_plan(gateway, plan, jobs)


def download(gateway: RunGateway, resolution: Resolution, jobs: int = DEFAULT_JOBS) -> int:
    """Download every artifact of every target; return the run count."""
    return len(pull(gateway, resolution, PullOptions(), jobs))


def download_runs(
    gateway: RunGateway,
    run_ids: list[str],
    out: Path,
    allowed: frozenset[str] = FINISHED,
    jobs: int = DEFAULT_JOBS,
) -> int:
    """Download runs with their nested runs; return how many runs were downloaded.

    Every named run is checked before anything is written, so a refused run never
    leaves a partial download behind.
    """
    return download(gateway, resolve_runs(gateway, run_ids, out, allowed), jobs)


def download_experiments(
    gateway: RunGateway,
    names_or_ids: list[str],
    out: Path,
    allowed: frozenset[str] = FINISHED,
    jobs: int = DEFAULT_JOBS,
    filter_string: str | None = None,
) -> int:
    """Download runs of each experiment with their nested runs."""
    return download(gateway, resolve_experiments(gateway, names_or_ids, out, allowed, filter_string), jobs)


def _read_history(gateway: RunGateway, run_id: str, key: str) -> list[Metric]:
    try:
        return gateway.metric_history(run_id, key)
    except MlflowException as err:
        raise MlflowException(f"reading metric {key!r} of run {run_id}: {err}") from err


def summarise_runs(
    gateway: RunGateway,
    runs: Sequence[Run],
    metric_globs: Sequence[str] = (),
    history_limit: int | None = None,
    jobs: int = DEFAULT_JOBS,
) -> list[RunSummary]:
    """Summaries in start-time order; one history read per run per matched metric key, none without globs.

    Lives here, not in ``summary``, because the history reads are the only effect and share the pool.
    """
    wanted = [(run, key) for run in runs for key in matching_keys(metric_globs, run.data.metrics)]
    with ThreadPoolExecutor(jobs) as pool:
        try:
            fetched = list(pool.map(lambda want: _read_history(gateway, want[0].info.run_id, want[1]), wanted))
        except BaseException:
            pool.shutdown(cancel_futures=True)  # do not drain thousands of queued reads after a failure
            raise
    histories: dict[str, dict[str, tuple[Point, ...]]] = {run.info.run_id: {} for run in runs}
    for (run, key), history in zip(wanted, fetched):
        if history:  # a key can lose its points between the search and the read
            histories[run.info.run_id][key] = to_points(history)
    return in_display_order(summarise_run(run, histories[run.info.run_id], history_limit) for run in runs)
