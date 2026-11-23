"""What the core needs from an MLflow server. The only MLflow-facing contract."""

from __future__ import annotations

import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mlflow.entities import Experiment, Metric, Run

from mlflow_pull.paths import RelPath


class InvalidSizeError(ValueError):
    """The server reported a size no file can have."""


@dataclass(frozen=True, slots=True)
class KnownSize:
    """A size the server reported; validated once here so no downstream sum can be driven negative."""

    bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise InvalidSizeError(f"invalid file size {self.bytes!r}: expected a non-negative integer")


@dataclass(frozen=True, slots=True)
class UnknownSize:
    """The server did not say: ``--max-size`` cannot vouch for such a file."""


UNKNOWN_SIZE = UnknownSize()
FileSize = KnownSize | UnknownSize


def size_from_listing(raw: int | None, path: RelPath) -> FileSize:
    """The boundary where a server's ``file_size`` becomes a size: ``None`` is unknown, a negative number an error."""
    if raw is None:
        return UNKNOWN_SIZE
    try:
        return KnownSize(raw)
    except InvalidSizeError as err:
        raise InvalidSizeError(f"artifact {path}: {err}") from None


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: RelPath
    size: FileSize

    @property
    def known_bytes(self) -> int:
        """The bytes the server vouched for; an unknown size adds nothing here (``PullPlan.check_size`` refuses those)."""
        return self.size.bytes if isinstance(self.size, KnownSize) else 0


@dataclass(frozen=True, slots=True)
class ArtifactDir:
    path: RelPath


ArtifactEntry = ArtifactFile | ArtifactDir


@dataclass(frozen=True, slots=True)
class UnsupportedName:
    """A name the server holds that this machine cannot write, kept as text for reports only.

    It is not a ``RelPath``: nothing can be written or read through it. The listing boundary has
    already refused hostile names, so ``under`` is only ever used to print where the name would go.
    """

    text: str

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.text.split("/"))

    def is_child_of(self, directory: RelPath | None) -> bool:
        return self.parts[:-1] == (() if directory is None else directory.parts)

    def under(self, root: Path) -> Path:
        return root.joinpath(*self.parts)


@dataclass(frozen=True, slots=True)
class UnsupportedEntry:
    """A listed file or directory the plan skips and counts: a directory's subtree is never listed."""

    path: UnsupportedName

    @classmethod
    def named(cls, text: str) -> UnsupportedEntry:
        return cls(UnsupportedName(text))


ListedEntry = ArtifactFile | ArtifactDir | UnsupportedEntry

# HTTP-served stores (the ``mlflow-artifacts:`` proxy, ``HttpArtifactRepository``) join the raw artifact
# path into the endpoint URL, so ``#`` and ``?`` cut it short and ``%41`` is decoded to ``A`` by the server:
# such a download fetches nothing or another file (checked against MLflow 3.17; over HTTP its own listing
# already refuses ``#``). Other stores would carry the names, but one rule keeps every store's result the same.
_UNFETCHABLE = re.compile(r"[#?]|%[0-9A-Fa-f]{2}")


def can_fetch(path: RelPath) -> bool:
    """Whether the MLflow client can download ``path``; ``;``, a lone ``%``, spaces and non-ASCII are fine."""
    return _UNFETCHABLE.search(path.value) is None


class RunGateway(Protocol):
    def get_run(self, run_id: str) -> Run: ...

    def search_runs(self, experiment_id: str, filter_string: str | None = None) -> list[Run]:
        """Every run of the experiment matching ``filter_string``."""

    def find_experiment(self, name_or_id: str) -> Experiment:
        """By name, else by id."""

    def search_experiments(self, filter_string: str) -> list[Experiment]: ...

    def metric_history(self, run_id: str, key: str) -> list[Metric]:
        """Every logged point of one metric of the run."""

    def list_artifacts(self, run_id: str, path: RelPath | None) -> list[ListedEntry]:
        """The direct children of the artifact directory ``path`` (``None``: the run root).

        Paths are relative to the run root and validated, so a hostile listing raises
        ``UnsafePathError`` here; a harmless name this machine cannot create is an ``UnsupportedEntry``.
        """

    def artifact_session(self, run_id: str) -> AbstractContextManager[None]:
        """Scope of one run's downloads on the calling thread: what they share (a store client, a thread pool) is released on exit.

        Another thread may hold a session of the same run at the same time.
        """

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        """The file ``path`` to ``dest_dir / path``; the caller has made every directory between.

        Only inside ``artifact_session(run_id)`` on the same thread.
        """
