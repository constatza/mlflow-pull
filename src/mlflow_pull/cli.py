"""Download the artifacts of MLflow runs with their nested runs, or of whole experiments.

The tracking server comes from ``MLFLOW_TRACKING_URI``, like the mlflow CLI.
With ``--ssh HOST`` (or ``MLPULL_SSH``) an SSH tunnel to a server that listens
only on HOST is opened for the duration of the download.
Only FINISHED runs are downloaded unless a status flag widens the selection.
``--path``, ``--max-size`` and ``--dry-run`` make downloads selective; a manifest in each
run directory lets a repeated pull fetch only what is missing.
``--format jsonl`` makes ``run`` and ``experiment`` print one JSON record per run on stdout and
every human note on stderr. Exit codes: 0 done, 1 refused or failed, 2 usage, 3 nothing matched.
``runs`` only reads: it summarises metrics, params and tags of runs of any status.

This module is the composition root: it reads arguments and environment, opens the
tunnel, builds the gateway, and reports. Decisions live in the pure modules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, TypeVar

import typer
from mlflow.exceptions import MlflowException

from mlflow_pull.execute import DEFAULT_JOBS, execute_plan, plan_pull, summarise_runs
from mlflow_pull.gateway import MlflowGateway
from mlflow_pull.exitcodes import ExitCode
from mlflow_pull.pathfilter import PathFilter
from mlflow_pull.plan import ARTIFACTS_ONLY, Content, PullOptions, PullPlan, SizeLimitError, UnknownSizeError, parse_size
from mlflow_pull.ports import RunGateway
from mlflow_pull.render import Format, default_columns, parse_columns, render
from mlflow_pull.report import OutputFormat, Reported, planned, render_jsonl, render_outcomes, render_plan
from mlflow_pull.resolve import NoMatchError, resolve_experiments, resolve_run_list, resolve_runs
from mlflow_pull.summary import MIN_HISTORY_POINTS, RunSummary
from mlflow_pull.selection import RunStatusError, allowed_statuses, build_filter, parse_when
from mlflow_pull.tree import Resolution
from mlflow_pull.tunnel import DEFAULT_REMOTE, RemoteAddr, SshHost, ssh_tunnel

DEFAULT_OUT = Path("mlflow-downloads")
DEFAULT_KEY_TAG = "run_key"

T = TypeVar("T")
R = TypeVar("R")

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
Jobs = Annotated[int, typer.Option("--jobs", "-j", min=1, help="Parallel downloads or history reads.")]

Since = Annotated[Optional[str], typer.Option(
    metavar="WHEN", help="Runs started at or after WHEN (ISO date/datetime or 30m, 6h, 2d, 1w ago).")]
Until = Annotated[Optional[str], typer.Option(metavar="WHEN", help="Runs started before WHEN.")]
Tag = Annotated[Optional[list[str]], typer.Option(
    metavar="KEY=VALUE", help="Runs with this tag value (repeatable).")]
Param = Annotated[Optional[list[str]], typer.Option(
    metavar="KEY=VALUE", help="Runs with this parameter value (repeatable).")]
RawFilter = Annotated[Optional[str], typer.Option(
    "--filter", metavar="EXPR", help="Raw MLflow run filter, ANDed with the others.")]
Key = Annotated[Optional[str], typer.Option(
    metavar="VALUE", help="Runs whose --tag-name tag equals VALUE.")]
TagName = Annotated[str, typer.Option(
    envvar="MLPULL_KEY_TAG", metavar="NAME", help="Tag that --key looks at.")]

PathGlobs = Annotated[Optional[list[str]], typer.Option(
    "--path", metavar="GLOB",
    help="Only artifacts matching GLOB, relative to the run root (repeatable). "
         "* stays inside a directory, ** crosses directories, !GLOB excludes; [ is a literal character. "
         "A directory needs dir/**, not dir/.")]
MaxSize = Annotated[Optional[str], typer.Option(
    "--max-size", metavar="SIZE",
    help="Refuse, before writing anything, to fetch more than SIZE (500M, 1.5G or bytes; 1024-based).")]
DryRun = Annotated[bool, typer.Option(
    "--dry-run", help="List the files that would be fetched with sizes and write nothing.")]
WhatOption = Annotated[Optional[list[str]], typer.Option(
    "--what", metavar="WHAT,...",
    help="metrics, params, tags and/or artifacts (comma list or repeated; default artifacts). "
         "Anything but artifacts writes run.json in each run directory.")]

OutFormat = Annotated[OutputFormat, typer.Option(
    "--format", help="text for people; jsonl: one JSON record per run on stdout, notes on stderr.")]

Resolve = Callable[[RunGateway, frozenset[str], Path], Resolution]


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
    except (ValueError, OverflowError):
        raise typer.BadParameter(
            f"expected an ISO date/datetime or 30m/6h/2d/1w, got {text!r}", param_hint=flag
        ) from None


def _ssh_host(text: str) -> SshHost:
    try:
        return SshHost(text)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--ssh") from None


def _remote(text: str) -> RemoteAddr:
    try:
        return RemoteAddr.parse(text)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--remote") from None


@contextmanager
def _connected(ssh: str | None, remote: str, compress: bool) -> Iterator[MlflowGateway]:
    """A gateway to the server, behind an SSH tunnel when ``ssh`` is given."""
    if compress and not ssh:
        raise typer.BadParameter("only applies with --ssh", param_hint="--compress")
    host = _ssh_host(ssh) if ssh else None
    address = _remote(remote)
    with ExitStack() as stack:
        # The tunnel URI goes to the client directly; no process-global environment write.
        uri = stack.enter_context(ssh_tunnel(host, address, compress)) if host else None
        yield MlflowGateway.connect(uri)


def _artifact_store_errors() -> tuple[type[Exception], ...]:
    """boto3's errors when it is installed: S3 artifact reads raise them unwrapped, not as MlflowException.

    botocore is not a dependency of this package, so it is looked up only when a failure needs mapping.
    """
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return ()
    return (BotoCoreError, ClientError)


@contextmanager
def _failing_cleanly() -> Iterator[None]:
    try:
        yield
    except NoMatchError as err:  # a valid selection that picks nothing
        typer.echo(str(err), err=True)
        raise typer.Exit(ExitCode.NOTHING_MATCHED) from None
    except (ValueError, OverflowError, OSError, MlflowException, *_artifact_store_errors()) as err:
        # bad input, a server time no calendar holds, unwritable path or missing ssh, unknown id
        typer.echo(str(err), err=True)
        raise typer.Exit(ExitCode.REFUSED) from None


def _selection_filter(
    since: str | None,
    until: str | None,
    tag: list[str] | None,
    param: list[str] | None,
    raw: str | None,
    key: str | None,
    tag_name: str,
) -> str | None:
    """The run filter for the shared selection options; ``--key`` is a tag filter on ``--tag-name``."""
    tags = _key_values(tag, "--tag") + ([(tag_name, key)] if key is not None else [])
    try:
        return build_filter(
            _when(since, "--since"), _when(until, "--until"), tags, _key_values(param, "--param"), raw
        )
    except ValueError as err:  # a value containing both quote characters
        raise typer.BadParameter(str(err)) from None


class PullMode(Enum):
    DRY_RUN = "dry-run"
    WRITE = "write"


def _parsed(parse: Callable[[T], R], value: T, flag: str) -> R:
    try:
        return parse(value)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint=flag) from None


def _pull_options(paths: list[str] | None, max_size: str | None, what: list[str] | None) -> PullOptions:
    return PullOptions(
        _parsed(Content.parse, what, "--what") if what else ARTIFACTS_ONLY,
        _parsed(PathFilter.parse, paths or [], "--path"),
        None if max_size is None else _parsed(parse_size, max_size, "--max-size"),
    )


def _echo_records(fmt: OutputFormat, records: Sequence[Reported]) -> None:
    """The run records, on stdout, for ``jsonl`` only; nothing at all when there are none."""
    if fmt is OutputFormat.JSONL and records:
        typer.echo(render_jsonl(records))


def _echo_report(fmt: OutputFormat, text: str, records: Sequence[Reported]) -> None:
    """``text`` is the human report: stdout for ``text``, stderr beside the records for ``jsonl``."""
    _echo_records(fmt, records)
    typer.echo(text, err=fmt is OutputFormat.JSONL)


def _echo_unsupported(plan: PullPlan, out: Path) -> None:
    """One stderr line per artifact MLflow cannot download by name: they are skipped, not fatal."""
    for run in plan.runs:
        for file in run.unsupported:
            typer.echo(f"unsupported name skipped: {file.path.under(run.target.dir).relative_to(out)}", err=True)


def _pull(
    resolve: Resolve,
    out: Path,
    ssh: str | None,
    remote: str,
    compress: bool,
    unfinished: bool,
    failed: bool,
    any_status: bool,
    jobs: int,
    options: PullOptions,
    mode: PullMode,
    fmt: OutputFormat,
) -> None:
    """Shared command body: tunnel, gateway, status selection, plan, size check, download, reporting.

    With ``jsonl`` stdout carries only the run records; every note goes to stderr.
    """
    allowed = options.content.statuses(unfinished or any_status, failed or any_status)

    # The handler is outermost so a failure to open the tunnel is one line, not a traceback.
    with _failing_cleanly(), _connected(ssh, remote, compress) as gateway:
        out = out.resolve()  # records and the summary name absolute directories
        try:
            resolution = resolve(gateway, allowed, out)
            for skip in resolution.skipped:
                typer.echo(f"skipped {skip.label}: {skip.status}", err=True)
            if resolution.run_count == 0:
                _echo_records(fmt, resolution.skipped)
                typer.echo("nothing matched: no run to download", err=True)
                raise typer.Exit(ExitCode.NOTHING_MATCHED)
            plan = plan_pull(gateway, resolution, options, jobs)
            _echo_unsupported(plan, out)
            if mode is PullMode.DRY_RUN:
                _echo_report(fmt, render_plan(plan, out), (*planned(plan), *resolution.skipped))
            plan.check_size(options.max_bytes)
            if mode is PullMode.DRY_RUN:
                return
            outcomes = execute_plan(gateway, plan, jobs)
        except RunStatusError as err:
            typer.echo(f"refused (override with -u, -f or -a): {err}", err=True)
            raise typer.Exit(ExitCode.REFUSED) from None
        except (SizeLimitError, UnknownSizeError) as err:
            typer.echo(f"refused (--max-size): {err}", err=True)
            raise typer.Exit(ExitCode.REFUSED) from None
    _echo_report(fmt, render_outcomes(outcomes, out), (*outcomes, *resolution.skipped))


@app.command()
def run(
    run_ids: Annotated[list[str], typer.Argument(metavar="RUN_ID...")],
    out: Out = DEFAULT_OUT,
    ssh: Ssh = None,
    remote: Remote = str(DEFAULT_REMOTE),
    compress: Compress = False,
    unfinished: Unfinished = False,
    failed: Failed = False,
    any_status: AnyStatus = False,
    jobs: Jobs = DEFAULT_JOBS,
    path: PathGlobs = None,
    max_size: MaxSize = None,
    dry_run: DryRun = False,
    what: WhatOption = None,
    format_: OutFormat = OutputFormat.TEXT,
) -> None:
    """Runs and all their nested runs."""
    options = _pull_options(path, max_size, what)
    _pull(
        lambda gateway, allowed, root: resolve_runs(gateway, run_ids, root, allowed),
        out, ssh, remote, compress, unfinished, failed, any_status, jobs,
        options, PullMode.DRY_RUN if dry_run else PullMode.WRITE, format_,
    )


@app.command()
def experiment(
    experiments: Annotated[list[str], typer.Argument(metavar="EXPERIMENT...",
                                                     help="Names, ids or * globs over names.")],
    out: Out = DEFAULT_OUT,
    ssh: Ssh = None,
    remote: Remote = str(DEFAULT_REMOTE),
    compress: Compress = False,
    unfinished: Unfinished = False,
    failed: Failed = False,
    any_status: AnyStatus = False,
    jobs: Jobs = DEFAULT_JOBS,
    since: Since = None,
    until: Until = None,
    tag: Tag = None,
    param: Param = None,
    filter_: RawFilter = None,
    key: Key = None,
    tag_name: TagName = DEFAULT_KEY_TAG,
    path: PathGlobs = None,
    max_size: MaxSize = None,
    dry_run: DryRun = False,
    what: WhatOption = None,
    format_: OutFormat = OutputFormat.TEXT,
) -> None:
    """Runs of experiments, nested runs included.

    Without selection options: every top-level run. With them: every matching run
    at any depth, each with its nested runs.
    """
    filter_string = _selection_filter(since, until, tag, param, filter_, key, tag_name)
    options = _pull_options(path, max_size, what)
    _pull(
        lambda gateway, allowed, root: resolve_experiments(gateway, experiments, root, allowed, filter_string),
        out, ssh, remote, compress, unfinished, failed, any_status, jobs,
        options, PullMode.DRY_RUN if dry_run else PullMode.WRITE, format_,
    )


@app.command()
def runs(
    experiments: Annotated[Optional[list[str]], typer.Option(
        "--experiment", "-e", metavar="NAME|ID|GLOB",
        help="Experiment to read (repeatable); default: all experiments.")] = None,
    ssh: Ssh = None,
    remote: Remote = str(DEFAULT_REMOTE),
    compress: Compress = False,
    jobs: Jobs = DEFAULT_JOBS,
    since: Since = None,
    until: Until = None,
    tag: Tag = None,
    param: Param = None,
    filter_: RawFilter = None,
    key: Key = None,
    tag_name: TagName = DEFAULT_KEY_TAG,
    metric: Annotated[Optional[list[str]], typer.Option(
        metavar="GLOB", help="Metric keys to summarise from their full history (repeatable).")] = None,
    history: Annotated[Optional[int], typer.Option(
        min=MIN_HISTORY_POINTS, metavar="N", help="Also give at most N history points per --metric key.")] = None,
    format_: Annotated[Format, typer.Option("--format", help="Output format.")] = Format.TEXT,
    columns: Annotated[Optional[str], typer.Option(
        metavar="COL,...",
        help="Table columns: run_id,name,status,start,end and KEY:min|last|min_step|count.")] = None,
) -> None:
    """Read-only summary of runs, nested runs included, of any status.

    Without --metric only the latest metric values are shown and no history is read.
    """
    globs = metric or []
    if history is not None and not globs:
        raise typer.BadParameter("needs --metric", param_hint="--history")
    if history is not None and format_ not in (Format.JSON, Format.JSONL):
        raise typer.BadParameter("only json and jsonl output carry history", param_hint="--history")
    filter_string = _selection_filter(since, until, tag, param, filter_, key, tag_name)
    try:
        chosen = None if columns is None else parse_columns(columns, globs)
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--columns") from None
    rows: Sequence[RunSummary] = ()
    with _failing_cleanly(), _connected(ssh, remote, compress) as gateway:
        try:
            found = resolve_run_list(gateway, experiments or [], filter_string)
        except NoMatchError as err:  # still rendered, so a script reads [] before it sees exit 3
            typer.echo(str(err), err=True)
        else:
            rows = summarise_runs(gateway, found, globs, history, jobs)
    text = render(format_, rows, chosen or default_columns(rows, globs))
    if text:  # an empty jsonl result is no output, not a blank line
        typer.echo(text)
    if not rows:
        raise typer.Exit(ExitCode.NOTHING_MATCHED)
