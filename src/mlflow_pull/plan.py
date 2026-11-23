"""Decide which files of each run to fetch: the content to pull, sizes, manifests, plans. Pure.

A run is planned from its artifact listing and what an earlier pull recorded, so
``--dry-run``, ``--max-size`` and the skip of unchanged files all read the same plan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path

from mlflow_pull.paths import RelPath, UnsafePathError, fold
from mlflow_pull.pathfilter import EVERYTHING, PathFilter
from mlflow_pull.ports import ArtifactFile, KnownSize, UnknownSize, can_fetch
from mlflow_pull.selection import RunStatus, allowed_statuses
from mlflow_pull.tree import RunTarget

MANIFEST_NAME = ".mlpull.json"
SUMMARY_NAME = "run.json"
MANIFEST_VERSION = 2  # 2 records the mtime of each file beside its size
EVERY_STATUS = allowed_statuses(unfinished=True, failed=True)

# --- what to pull -------------------------------------------------------------------------


class What(Enum):
    METRICS = "metrics"
    PARAMS = "params"
    TAGS = "tags"
    ARTIFACTS = "artifacts"


@dataclass(frozen=True, slots=True)
class Content:
    """A non-empty choice of what to pull per run. Anything but artifacts becomes ``run.json``."""

    what: frozenset[What]

    def __post_init__(self) -> None:
        if not self.what:
            raise ValueError("--what needs at least one of " + ",".join(w.value for w in What))

    @classmethod
    def parse(cls, texts: Sequence[str]) -> Content:
        """Comma lists and repeats: ``["metrics,tags", "artifacts"]``."""
        names = [name.strip() for text in texts for name in text.split(",")]
        try:
            return cls(frozenset(What(name) for name in names))
        except ValueError:
            known = ",".join(w.value for w in What)
            raise ValueError(f"--what takes a comma list of {known}, got {','.join(names)!r}") from None

    @property
    def artifacts(self) -> bool:
        return What.ARTIFACTS in self.what

    @property
    def metadata(self) -> bool:
        return bool(self.what - {What.ARTIFACTS})

    @property
    def reserved(self) -> frozenset[RelPath]:
        """Run-root names mlpull writes itself, which an artifact must not take."""
        names = [MANIFEST_NAME, *([SUMMARY_NAME] if self.metadata else [])]
        return frozenset(RelPath(name) for name in names)

    def statuses(self, unfinished: bool, failed: bool) -> frozenset[str]:
        """The run status gate: it protects artifacts only; metadata is safe to read at any status."""
        return allowed_statuses(unfinished, failed) if self.artifacts else EVERY_STATUS


ARTIFACTS_ONLY = Content(frozenset({What.ARTIFACTS}))


@dataclass(frozen=True, slots=True)
class PullOptions:
    content: Content = ARTIFACTS_ONLY
    path_filter: PathFilter = EVERYTHING
    max_bytes: int | None = None


# --- sizes --------------------------------------------------------------------------------

UNIT_BASE = 1024
SIZE_UNITS = "KMGT"
_SIZE = re.compile(rf"(\d+(?:\.\d+)?)([{SIZE_UNITS}]?)(?:I?B)?", re.IGNORECASE)


def parse_size(text: str) -> int:
    """``500M``, ``1.5G``, ``2GiB`` or bare bytes, in units of 1024; rounds down, must be positive."""
    match = _SIZE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"expected a size like 500M, 1.5G or a byte count, got {text!r}")
    number, unit = match[1], match[2].upper()
    size = int(Decimal(number) * UNIT_BASE ** (SIZE_UNITS.index(unit) + 1 if unit else 0))
    if size <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return size


def format_bytes(count: int) -> str:
    if count < UNIT_BASE:
        return f"{count} B"
    value = float(count)
    for unit in SIZE_UNITS:
        value /= UNIT_BASE
        if value < UNIT_BASE:
            break
    return f"{value:.1f} {unit}iB"


class SizeLimitError(ValueError):
    """The files to fetch add up to more than ``--max-size``."""

    def __init__(self, files: int, total: int, limit: int) -> None:
        super().__init__(
            f"{files} file(s), {format_bytes(total)} to fetch is over the limit of {format_bytes(limit)}"
        )


class UnknownSizeError(ValueError):
    """``--max-size`` was given but the server did not say how big some files to fetch are."""

    def __init__(self, files: Sequence[ArtifactFile]) -> None:
        super().__init__(
            f"{len(files)} file(s) to fetch have no size on the server (first: {files[0].path}), "
            "so --max-size cannot be checked; drop --max-size or leave them out with --path '!GLOB'"
        )


# --- the manifest: what an earlier pull left in a run directory --------------------------------


@dataclass(frozen=True, slots=True)
class OnDiskFile:
    """What the file system says about a downloaded file; both numbers must still match to trust the file."""

    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class Manifest:
    run_id: str
    status: RunStatus  # the run's status at that pull
    files: Mapping[RelPath, OnDiskFile] = field(default_factory=dict)

    def to_json(self) -> str:
        files = {
            path.value: {"size": found.size, "mtime_ns": found.mtime_ns} for path, found in sorted(self.files.items())
        }
        return json.dumps(
            {"version": MANIFEST_VERSION, "run_id": self.run_id, "status": self.status.value, "files": files},
            indent=2,
        )

    def restricted_to(self, on_disk: Mapping[RelPath, OnDiskFile]) -> Manifest:
        """Only the files whose recorded size and mtime are what ``on_disk`` (the shell's stat) shows now."""
        kept = {path: found for path, found in self.files.items() if on_disk.get(path) == found}
        return Manifest(self.run_id, self.status, kept)


@dataclass(frozen=True, slots=True)
class Unrecorded:
    """No usable manifest: missing, unreadable, corrupt or from an unknown version."""


Recorded = Manifest | Unrecorded
UNRECORDED = Unrecorded()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _on_disk_file(entry: object) -> OnDiskFile | None:
    match entry:
        case {"size": int() as size, "mtime_ns": int() as mtime_ns} if (
            _is_int(size) and _is_int(mtime_ns) and size >= 0
        ):
            return OnDiskFile(size, mtime_ns)
        case _:
            return None


def parse_manifest(text: str) -> Recorded:
    """Validate a manifest file's text; never raises. The file is boundary data and untrusted."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return UNRECORDED
    match data:
        case {"version": int() as version, "run_id": str() as run_id, "status": str() as status,
              "files": dict() as files} if _is_int(version) and version == MANIFEST_VERSION:
            found = {name: _on_disk_file(entry) for name, entry in files.items()}
            if any(entry is None for entry in found.values()):
                return UNRECORDED
            try:
                return Manifest(
                    run_id, RunStatus.parse(status), {RelPath(name): entry for name, entry in found.items() if entry}
                )
            except UnsafePathError:
                return UNRECORDED
        case _:
            return UNRECORDED


# --- planning -------------------------------------------------------------------------------


class PlanConflictError(ValueError):
    """Two artifacts, or an artifact and a file mlpull writes, cannot both land on disk."""


class ReservedNameError(PlanConflictError):
    """An artifact would land on a name mlpull writes itself, or in a nested run's directory."""


class DuplicatePathError(PlanConflictError):
    """Two artifacts are the same file on a case-insensitive or Unicode-normalising file system."""


class PrefixConflictError(PlanConflictError):
    """An artifact is a file and the directory another artifact needs."""


@dataclass(frozen=True, slots=True)
class RunPlan:
    target: RunTarget
    files: tuple[ArtifactFile, ...]  # to fetch
    up_to_date: tuple[ArtifactFile, ...]  # recorded, intact, same size: skipped
    unsupported: tuple[ArtifactFile, ...]  # wanted, but MLflow cannot download that name: skipped
    kept: Mapping[RelPath, OnDiskFile]  # recorded files that stay: the up to date ones and those out of scope

    @property
    def fetch_bytes(self) -> int:
        return sum(f.known_bytes for f in self.files)

    @property
    def up_to_date_bytes(self) -> int:
        return sum(f.known_bytes for f in self.up_to_date)

    @property
    def unknown_size_files(self) -> tuple[ArtifactFile, ...]:
        return tuple(f for f in self.files if isinstance(f.size, UnknownSize))

    def record(self, fetched: Mapping[RelPath, OnDiskFile]) -> Manifest:
        """The manifest once ``files`` have landed, with each fetched file as found on disk."""
        info = self.target.run.info
        return Manifest(info.run_id, RunStatus.parse(info.status), {**self.kept, **fetched})


def _reusable(recorded: Recorded, run_id: str) -> Mapping[RelPath, OnDiskFile]:
    """Files an earlier pull of this run can vouch for; a pull of a still-changing run vouches for none."""
    if isinstance(recorded, Unrecorded) or recorded.run_id != run_id or not recorded.status.terminal:
        return {}
    return recorded.files


def _check_names(run_id: str, paths: Iterable[RelPath], reserved: frozenset[RelPath], nested: frozenset[str]) -> None:
    """Refuse artifacts that would land on a name mlpull owns; names compare case- and normalisation-folded."""
    owned = {fold(name.value) for name in reserved}
    runs = {fold(name) for name in nested}
    for path in sorted(paths):
        top = fold(path.parts[0])
        if top in owned:
            raise ReservedNameError(
                f"run {run_id}: artifact {path} collides with a file mlpull writes; skip it with --path '!{path}'"
            )
        if top in runs:
            raise ReservedNameError(
                f"run {run_id}: artifact {path} would land in the directory of a nested run; "
                f"skip it with --path '!{path}'"
            )


def _check_layout(run_id: str, paths: Iterable[RelPath]) -> None:
    """Refuse two artifacts that are one file, or a file that is also another artifact's directory."""
    folded: dict[str, RelPath] = {}
    for path in sorted(paths):
        key = fold(path.value)
        if key in folded:
            raise DuplicatePathError(
                f"run {run_id}: artifacts {folded[key]} and {path} are the same file after case/Unicode folding"
            )
        folded[key] = path
    for key, path in folded.items():
        parts = key.split("/")
        for length in range(1, len(parts)):
            if (ancestor := folded.get("/".join(parts[:length]))) is not None:
                raise PrefixConflictError(f"run {run_id}: artifact {ancestor} is a file but {path} needs it as a directory")


def plan_run(
    target: RunTarget,
    listing: Iterable[ArtifactFile],
    recorded: Recorded,
    path_filter: PathFilter,
    reserved: frozenset[RelPath],
    nested: frozenset[str] = frozenset(),
) -> RunPlan:
    """Diff the run's listing against the manifest, file by file.

    ``recorded`` holds only files verified on disk. Recorded files outside ``path_filter``
    stay in the new manifest, so a later pull with another ``--path`` set fetches only what is missing.
    ``nested`` holds the directory names of the run's direct nested runs: artifacts must not land there.
    Names are checked on the files the filter accepts, so ``--path '!NAME'`` is the way out of a clash.
    """
    run_id = target.run.info.run_id
    accepted = {f.path: f for f in listing if path_filter.accepts(f.path)}
    wanted = {path: f for path, f in accepted.items() if can_fetch(path)}
    _check_names(run_id, wanted, reserved, nested)
    _check_layout(run_id, wanted)
    known = _reusable(recorded, run_id)
    ordered = [wanted[path] for path in sorted(wanted)]
    up_to_date = tuple(f for f in ordered if _unchanged(known.get(f.path), f))
    fetched = tuple(f for f in ordered if not _unchanged(known.get(f.path), f))
    unsupported = tuple(accepted[path] for path in sorted(accepted.keys() - wanted.keys()))
    carried = {path: found for path, found in known.items() if not path_filter.accepts(path)}
    kept = carried | {f.path: known[f.path] for f in up_to_date}
    return RunPlan(target, fetched, up_to_date, unsupported, kept)


def _unchanged(on_disk: OnDiskFile | None, listed: ArtifactFile) -> bool:
    """The file on disk is the recorded one (``on_disk`` is ``None`` when not) and the server still lists its size."""
    return on_disk is not None and isinstance(listed.size, KnownSize) and listed.size.bytes == on_disk.size


@dataclass(frozen=True, slots=True)
class PullPlan:
    content: Content
    root: Path
    runs: tuple[RunPlan, ...]

    @property
    def fetch_files(self) -> int:
        return sum(len(run.files) for run in self.runs)

    @property
    def fetch_bytes(self) -> int:
        return sum(run.fetch_bytes for run in self.runs)

    @property
    def unsupported(self) -> int:
        return sum(len(run.unsupported) for run in self.runs)

    def check_size(self, limit: int | None) -> None:
        """Raise when the files to fetch exceed ``limit`` or any has no known size (``None``: no limit)."""
        if limit is None:
            return
        if unknown := [f for run in self.runs for f in run.unknown_size_files]:
            raise UnknownSizeError(unknown)
        if self.fetch_bytes > limit:
            raise SizeLimitError(self.fetch_files, self.fetch_bytes, limit)


# --- outcomes ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Downloaded:
    """Something was written for the run: files fetched, or its ``run.json``."""

    target: RunTarget
    files: int
    bytes: int
    unsupported: int


@dataclass(frozen=True, slots=True)
class UpToDate:
    """Nothing to write; ``files`` and ``bytes`` are what an earlier pull left in place."""

    target: RunTarget
    files: int
    bytes: int
    unsupported: int


@dataclass(frozen=True, slots=True)
class WouldDownload:
    """``--dry-run``: ``files`` and ``bytes`` are what a pull would fetch."""

    target: RunTarget
    files: int
    bytes: int
    unsupported: int


RunOutcome = Downloaded | UpToDate


def _writes(plan: RunPlan, content: Content) -> bool:
    return bool(plan.files) or content.metadata


def _up_to_date(plan: RunPlan) -> UpToDate:
    return UpToDate(plan.target, len(plan.up_to_date), plan.up_to_date_bytes, len(plan.unsupported))


def outcome_of(plan: RunPlan, content: Content) -> RunOutcome:
    if _writes(plan, content):
        return Downloaded(plan.target, len(plan.files), plan.fetch_bytes, len(plan.unsupported))
    return _up_to_date(plan)


def planned_outcome(plan: RunPlan, content: Content) -> WouldDownload | UpToDate:
    """What ``outcome_of`` would report for ``plan``, before anything is written."""
    if _writes(plan, content):
        return WouldDownload(plan.target, len(plan.files), plan.fetch_bytes, len(plan.unsupported))
    return _up_to_date(plan)
