"""Text for a pull plan (``--dry-run``) and for the final line of a pull, and one JSON record per run. Pure; returns str."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from mlflow_pull.plan import (
    SUMMARY_NAME,
    Downloaded,
    PullPlan,
    RunOutcome,
    RunPlan,
    UpToDate,
    WouldDownload,
    format_bytes,
    planned_outcome,
)
from mlflow_pull.ports import FileSize, KnownSize
from mlflow_pull.render import dumps_strict
from mlflow_pull.tree import Skipped

SIZE_WIDTH = 10
UNKNOWN_SIZE_TEXT = "unknown"


class OutputFormat(Enum):
    """How ``run`` and ``experiment`` report: for people, or one JSON record per run for scripts."""

    TEXT = "text"
    JSONL = "jsonl"


class Outcome(Enum):
    DOWNLOADED = "downloaded"
    UP_TO_DATE = "up_to_date"
    SKIPPED_STATUS = "skipped_status"
    WOULD_DOWNLOAD = "would_download"


Reported = Downloaded | UpToDate | WouldDownload | Skipped


def _size_text(size: FileSize) -> str:
    return format_bytes(size.bytes) if isinstance(size, KnownSize) else UNKNOWN_SIZE_TEXT


def unsupported_note(count: int) -> str:
    """``, 3 unsupported name(s) skipped``, or nothing when there are none."""
    return f", {count} unsupported name(s) skipped" if count else ""


def _run_lines(plan: RunPlan, root: Path, writes_summary: bool) -> list[str]:
    target = plan.target
    head = (
        f"{target.dir.relative_to(root)}  {target.run.info.status}: "
        f"{len(plan.files)} file(s), {format_bytes(plan.fetch_bytes)} to fetch"
        f", {len(plan.up_to_date)} up to date{unsupported_note(len(plan.unsupported))}"
    )
    lines = [head, *(f"  {_size_text(f.size):>{SIZE_WIDTH}}  {f.path}" for f in plan.files)]
    return lines + ([f"  {'':>{SIZE_WIDTH}}  {SUMMARY_NAME}"] if writes_summary else [])


def render_plan(plan: PullPlan, root: Path) -> str:
    """Per run the files that would be fetched with sizes, then the totals."""
    lines = [line for run in plan.runs for line in _run_lines(run, root, plan.content.metadata)]
    up_to_date = sum(len(run.up_to_date) for run in plan.runs)
    lines.append(
        f"Would fetch {plan.fetch_files} file(s), {format_bytes(plan.fetch_bytes)} "
        f"from {len(plan.runs)} run(s) ({up_to_date} file(s) already up to date{unsupported_note(plan.unsupported)})"
    )
    return "\n".join(lines)


def render_outcomes(outcomes: Sequence[RunOutcome], out: Path) -> str:
    downloaded = [o for o in outcomes if isinstance(o, Downloaded)]
    current = [o for o in outcomes if isinstance(o, UpToDate)]
    files = sum(o.files for o in downloaded)
    size = format_bytes(sum(o.bytes for o in downloaded))
    line = f"Downloaded {len(downloaded)} run(s), {files} file(s), {size} to {out.resolve()}"
    line += unsupported_note(sum(o.unsupported for o in outcomes))
    return line + (f" ({len(current)} run(s) already up to date)" if current else "")


def planned(plan: PullPlan) -> tuple[WouldDownload | UpToDate, ...]:
    """The ``--dry-run`` counterpart of the outcomes of ``execute_plan``."""
    return tuple(planned_outcome(run, plan.content) for run in plan.runs)


def outcome_record(item: Reported) -> dict[str, object]:
    """The JSON shape of one run in a pull report; a skipped run has no files, bytes or unsupported names."""
    match item:
        case Downloaded(files=files, bytes=size, unsupported=unsupported):
            outcome = Outcome.DOWNLOADED
        case UpToDate(files=files, bytes=size, unsupported=unsupported):
            outcome = Outcome.UP_TO_DATE
        case WouldDownload(files=files, bytes=size, unsupported=unsupported):
            outcome = Outcome.WOULD_DOWNLOAD
        case Skipped():
            outcome, files, size, unsupported = Outcome.SKIPPED_STATUS, 0, 0, 0
    run = item.target.run.info
    return {
        "run_id": run.run_id,
        "name": run.run_name,
        "status": run.status,
        "dir": str(item.target.dir),
        "outcome": outcome.value,
        "files": files,
        "bytes": size,
        "unsupported": unsupported,
    }


def render_jsonl(items: Sequence[Reported]) -> str:
    """One strict JSON object per line, in the order given; no items give an empty string."""
    return "\n".join(dumps_strict(outcome_record(item)) for item in items)
