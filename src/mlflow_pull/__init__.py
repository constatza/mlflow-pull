"""Download the artifacts of MLflow runs with their nested runs, or of whole experiments.

The tracking server comes from ``MLFLOW_TRACKING_URI``, like the mlflow CLI.
With ``--ssh HOST`` (or ``MLPULL_SSH``) an SSH tunnel to a server that listens
only on HOST is opened for the duration of the download.
Only FINISHED runs are downloaded unless a status flag widens the selection.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

import typer
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

PARENT_RUN_TAG = "mlflow.parentRunId"
DEFAULT_OUT = Path("mlflow-downloads")
DEFAULT_JOBS = 4
SEARCH_PAGE_SIZE = 50_000  # MLflow's server-side maximum per search call
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')
DEFAULT_REMOTE = "localhost:5000"  # server address as seen from the SSH host
TUNNEL_TIMEOUT_S = 15.0
_POLL_S = 0.2
_LOOPBACK = "127.0.0.1"

FINISHED = frozenset({"FINISHED"})
UNFINISHED = frozenset({"RUNNING", "SCHEDULED"})
FAILED = frozenset({"FAILED", "KILLED"})

_RELATIVE_WHEN = re.compile(r"(\d+)([mhdw])")
_WHEN_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}

Children = dict[Optional[str], list[Run]]
Task = tuple[Run, Children, Path]  # a root run, its experiment's run tree, destination
Download = Callable[[MlflowClient, frozenset[str], int], int]


class RunStatusError(ValueError):
    """A run named explicitly has a status outside the allowed set."""


def allowed_statuses(unfinished: bool = False, failed: bool = False) -> frozenset[str]:
    return FINISHED | (UNFINISHED if unfinished else frozenset()) | (FAILED if failed else frozenset())


def parse_when(text: str, now: datetime) -> datetime:
    """``30m``/``6h``/``2d``/``1w`` before ``now``, or an ISO date/datetime (naive = local time)."""
    if match := _RELATIVE_WHEN.fullmatch(text):
        return now - timedelta(**{_WHEN_UNITS[match[2]]: int(match[1])})
    return datetime.fromisoformat(text).astimezone()


def _quote(value: str) -> str:
    # MLflow's filter grammar has no escapes: pick the quote the value does not contain.
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    raise ValueError(f"value contains both quote characters: {value}")


def build_filter(
    since: datetime | None = None,
    until: datetime | None = None,
    tags: list[tuple[str, str]] | None = None,
    params: list[tuple[str, str]] | None = None,
    raw: str | None = None,
) -> str | None:
    """AND the selections into one MLflow run filter string; ``None`` when empty."""
    clauses = []
    if since is not None:
        clauses.append(f"attributes.start_time >= {int(since.timestamp() * 1000)}")
    if until is not None:
        clauses.append(f"attributes.start_time < {int(until.timestamp() * 1000)}")
    for prefix, pairs in (("tags", tags or []), ("params", params or [])):
        clauses += [f"{prefix}.`{key}` = {_quote(value)}" for key, value in pairs]
    if raw:
        clauses.append(raw)
    return " and ".join(clauses) or None


def _dir_name(run: Run) -> str:
    name = _UNSAFE_PATH_CHARS.sub("_", run.info.run_name or "run")
    return f"{name}-{run.info.run_id[:8]}"


def _children_by_parent(client: MlflowClient, experiment_id: str) -> Children:
    """One search for the whole experiment; run -> children map built in memory."""
    runs = client.search_runs([experiment_id], max_results=SEARCH_PAGE_SIZE)
    children: Children = defaultdict(list)
    for run in runs:
        children[run.data.tags.get(PARENT_RUN_TAG)].append(run)
    return children


def _download(
    client: MlflowClient, run: Run, children: Children, parent_dir: Path, allowed: frozenset[str]
) -> int:
    """Download ``run`` and its allowed descendants under ``parent_dir``; return the run count."""
    if run.info.status not in allowed:
        typer.echo(f"skipped {_dir_name(run)}: {run.info.status}", err=True)
        return 0
    run_dir = parent_dir / _dir_name(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(run.info.run_id, "", str(run_dir))
    return 1 + sum(
        _download(client, child, children, run_dir, allowed) for child in children[run.info.run_id]
    )


def _run_all(client: MlflowClient, tasks: list[Task], allowed: frozenset[str], jobs: int) -> int:
    with ThreadPoolExecutor(jobs) as pool:
        return sum(pool.map(lambda task: _download(client, *task, allowed), tasks))


def download_runs(
    client: MlflowClient,
    run_ids: list[str],
    out: Path,
    allowed: frozenset[str] = FINISHED,
    jobs: int = DEFAULT_JOBS,
) -> int:
    """Download runs with their nested runs; return how many runs were downloaded.

    Every named run is checked before anything is written, so a refused run never
    leaves a partial download behind.
    """
    runs = [client.get_run(run_id) for run_id in run_ids]
    refused = [f"{r.info.run_id} is {r.info.status}" for r in runs if r.info.status not in allowed]
    if refused:
        raise RunStatusError("; ".join(refused))
    trees = {eid: _children_by_parent(client, eid) for eid in {r.info.experiment_id for r in runs}}
    return _run_all(client, [(r, trees[r.info.experiment_id], out) for r in runs], allowed, jobs)


def _resolve_experiments(client: MlflowClient, names_or_ids: list[str]) -> list[Experiment]:
    """Names, ids or ``*`` globs over names; each experiment once, in argument order."""
    found: dict[str, Experiment] = {}
    for name_or_id in names_or_ids:
        if "*" in name_or_id:
            # shortcut: "_" and "%" in the pattern act as LIKE wildcards too; harmless for globs.
            pattern = _quote(name_or_id.replace("*", "%"))
            matches = client.search_experiments(filter_string=f"name LIKE {pattern}")
            if not matches:
                raise ValueError(f"no experiment matches {name_or_id}")
        else:
            matches = [client.get_experiment_by_name(name_or_id) or client.get_experiment(name_or_id)]
        found |= {e.experiment_id: e for e in matches}
    return list(found.values())


def _matched_roots(children: Children, matches: list[Run]) -> list[Run]:
    """Matched runs that have no matched ancestor, so no subtree is downloaded twice."""
    by_id = {run.info.run_id: run for runs in children.values() for run in runs}
    matched = {run.info.run_id for run in matches}

    def has_matched_ancestor(run: Run) -> bool:
        parent = run.data.tags.get(PARENT_RUN_TAG)
        while parent is not None:
            if parent in matched:
                return True
            parent = by_id[parent].data.tags.get(PARENT_RUN_TAG) if parent in by_id else None
        return False

    return [by_id.get(r.info.run_id, r) for r in matches if not has_matched_ancestor(r)]


def download_experiments(
    client: MlflowClient,
    names_or_ids: list[str],
    out: Path,
    allowed: frozenset[str] = FINISHED,
    jobs: int = DEFAULT_JOBS,
    filter_string: str | None = None,
) -> int:
    """Download runs of each experiment with their nested runs.

    Without ``filter_string`` the top-level runs are the roots; with it, every
    matching run at any depth is, minus matches nested under another match.
    """
    tasks: list[Task] = []
    for experiment in _resolve_experiments(client, names_or_ids):
        children = _children_by_parent(client, experiment.experiment_id)
        if filter_string is None:
            roots = children[None]
        else:
            matches = client.search_runs(
                [experiment.experiment_id], filter_string, max_results=SEARCH_PAGE_SIZE
            )
            roots = _matched_roots(children, matches)
        experiment_dir = out / _UNSAFE_PATH_CHARS.sub("_", experiment.name)
        tasks += [(run, children, experiment_dir) for run in roots]
    return _run_all(client, tasks, allowed, jobs)


def _free_port() -> int:
    # shortcut: the port may be taken between close and ssh's bind; ssh then fails
    # fast (ExitOnForwardFailure) and the error is reported. Retry loop if it bites.
    with socket.socket() as sock:
        sock.bind((_LOOPBACK, 0))
        return sock.getsockname()[1]


def ssh_command(host: str, remote: str, local_port: int, compress: bool = False) -> list[str]:
    """``ssh`` argv forwarding ``local_port`` to ``remote`` as reached from ``host``."""
    return [
        "ssh", "-N",
        *(["-C"] if compress else []),
        "-o", "ExitOnForwardFailure=yes",
        # Fail instead of waiting on a password prompt nobody sees.
        "-o", "BatchMode=yes",
        "-L", f"{_LOOPBACK}:{local_port}:{remote}",
        host,
    ]


@contextmanager
def ssh_tunnel(
    host: str,
    remote: str = DEFAULT_REMOTE,
    compress: bool = False,
    timeout: float = TUNNEL_TIMEOUT_S,
) -> Iterator[str]:
    """Open an SSH tunnel to ``remote`` via ``host``; yield its local HTTP URI."""
    port = _free_port()
    proc = subprocess.Popen(
        ssh_command(host, remote, port, compress),
        stdin=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"ssh to {host} failed: {proc.stderr.read().strip()}")
            try:
                socket.create_connection((_LOOPBACK, port), _POLL_S).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"ssh tunnel to {host} not ready after {timeout:g} s")
                time.sleep(_POLL_S)
        yield f"http://{_LOOPBACK}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# --- CLI ---------------------------------------------------------------------

app = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)

Out = Annotated[Path, typer.Option("--out", "-o", envvar="MLPULL_OUT", help="Destination directory.")]
Ssh = Annotated[
    Optional[str],
    typer.Option("--ssh", envvar="MLPULL_SSH", metavar="HOST",
                 help="Tunnel to the server through this SSH host."),
]
Remote = Annotated[
    str,
    typer.Option("--remote", envvar="MLPULL_REMOTE", metavar="HOST:PORT",
                 help="Server address as seen from the SSH host."),
]
Compress = Annotated[
    bool, typer.Option("--compress", "-z", help="Compress data in transit (ssh -C); needs --ssh.")
]
Unfinished = Annotated[
    bool, typer.Option("--unfinished", "-u", help="Also download RUNNING and SCHEDULED runs.")
]
Failed = Annotated[bool, typer.Option("--failed", "-f", help="Also download FAILED and KILLED runs.")]
AnyStatus = Annotated[bool, typer.Option("--all", "-a", help="Download runs of any status (-u -f).")]
Jobs = Annotated[int, typer.Option("--jobs", "-j", min=1, help="Parallel downloads.")]


def _key_values(items: list[str] | None, flag: str) -> list[tuple[str, str]]:
    pairs = []
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or "`" in key:
            raise typer.BadParameter(f"expected KEY=VALUE, got {item!r}", param_hint=flag)
        pairs.append((key, value))
    return pairs


def _when(text: str | None, flag: str) -> datetime | None:
    if text is None:
        return None
    try:
        return parse_when(text, datetime.now().astimezone())
    except ValueError:
        raise typer.BadParameter(
            f"expected an ISO date/datetime or 30m/6h/2d/1w, got {text!r}", param_hint=flag
        ) from None


def _pull(
    download: Download,
    out: Path,
    ssh: str | None,
    remote: str,
    compress: bool,
    unfinished: bool,
    failed: bool,
    any_status: bool,
    jobs: int,
) -> None:
    """Shared command body: tunnel, status selection, error reporting."""
    if compress and not ssh:
        raise typer.BadParameter("only applies with --ssh", param_hint="--compress")
    allowed = allowed_statuses(unfinished or any_status, failed or any_status)
    with ExitStack() as stack:
        if ssh:
            # Env, not just the client argument: proxied ``mlflow-artifacts:`` stores
            # resolve the server from MLFLOW_TRACKING_URI.
            os.environ["MLFLOW_TRACKING_URI"] = stack.enter_context(ssh_tunnel(ssh, remote, compress))
        try:
            count = download(MlflowClient(), allowed, jobs)
        except RunStatusError as err:
            typer.echo(f"refused (override with -u, -f or -a): {err}", err=True)
            raise typer.Exit(1) from None
        except ValueError as err:
            typer.echo(str(err), err=True)
            raise typer.Exit(1) from None
    typer.echo(f"Downloaded {count} run(s) to {out.resolve()}")


@app.command()
def run(
    run_ids: Annotated[list[str], typer.Argument(metavar="RUN_ID...")],
    out: Out = DEFAULT_OUT,
    ssh: Ssh = None,
    remote: Remote = DEFAULT_REMOTE,
    compress: Compress = False,
    unfinished: Unfinished = False,
    failed: Failed = False,
    any_status: AnyStatus = False,
    jobs: Jobs = DEFAULT_JOBS,
) -> None:
    """Runs and all their nested runs."""
    _pull(
        lambda client, allowed, j: download_runs(client, run_ids, out, allowed, j),
        out, ssh, remote, compress, unfinished, failed, any_status, jobs,
    )


@app.command()
def experiment(
    experiments: Annotated[list[str], typer.Argument(metavar="EXPERIMENT...",
                                                     help="Names, ids or * globs over names.")],
    out: Out = DEFAULT_OUT,
    ssh: Ssh = None,
    remote: Remote = DEFAULT_REMOTE,
    compress: Compress = False,
    unfinished: Unfinished = False,
    failed: Failed = False,
    any_status: AnyStatus = False,
    jobs: Jobs = DEFAULT_JOBS,
    since: Annotated[Optional[str], typer.Option(
        metavar="WHEN", help="Runs started at or after WHEN (ISO date/datetime or 30m, 6h, 2d, 1w ago).")] = None,
    until: Annotated[Optional[str], typer.Option(
        metavar="WHEN", help="Runs started before WHEN.")] = None,
    tag: Annotated[Optional[list[str]], typer.Option(
        metavar="KEY=VALUE", help="Runs with this tag value (repeatable).")] = None,
    param: Annotated[Optional[list[str]], typer.Option(
        metavar="KEY=VALUE", help="Runs with this parameter value (repeatable).")] = None,
    filter_: Annotated[Optional[str], typer.Option(
        "--filter", metavar="EXPR", help="Raw MLflow run filter, ANDed with the others.")] = None,
) -> None:
    """Runs of experiments, nested runs included.

    Without selection options: every top-level run. With them: every matching run
    at any depth, each with its nested runs.
    """
    try:
        filter_string = build_filter(
            _when(since, "--since"), _when(until, "--until"),
            _key_values(tag, "--tag"), _key_values(param, "--param"), filter_,
        )
    except ValueError as err:  # a value containing both quote characters
        raise typer.BadParameter(str(err)) from None
    _pull(
        lambda client, allowed, j: download_experiments(
            client, experiments, out, allowed, j, filter_string
        ),
        out, ssh, remote, compress, unfinished, failed, any_status, jobs,
    )
