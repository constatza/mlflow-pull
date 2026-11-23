"""Turn server-controlled names into safe local paths. No network, no writes.

Experiment and run names come from the tracking server, so they are untrusted:
``..`` is a valid MLflow experiment name and would otherwise write outside ``--out``.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = os.name == "nt"  # read at call time, so tests can stand in for the other platform
MAX_COMPONENT_BYTES = 255  # the longest file name ext4, APFS and NTFS (in UTF-16 units) accept
MAX_PATH_BYTES = 1024  # a whole artifact path; far below PATH_MAX, so a run dir prefix still fits


def names_fold_on(platform: str) -> bool:
    """Whether the platform's usual file systems (NTFS, APFS) treat names differing in case or Unicode form as one file."""
    return platform in ("win32", "darwin")


FOLD_NAMES = names_fold_on(sys.platform)

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
_FALLBACK = "_"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE_PREFIX = re.compile(r"[A-Za-z]:")
_DOT_COMPONENTS = frozenset({".", ".."})


class UnsafePathError(ValueError):
    """A name or destination that is hostile or resolves outside the output directory: the pull stops."""


class UnsupportedNameError(UnsafePathError):
    """A harmless name this platform cannot create (too long, or a Windows-only rule): skip it, do not stop."""


def safe_component(name: str) -> str:
    """One plain path component: never empty, ``.``/``..``, a separator or a Windows device name."""
    cleaned = _UNSAFE_CHARS.sub("_", name).rstrip(" .")  # Windows drops trailing dots and spaces
    if not cleaned:
        return _FALLBACK
    if cleaned.split(".")[0].lower() in _WINDOWS_RESERVED:
        return f"{_FALLBACK}{cleaned}"
    return cleaned


def fold(text: str) -> str:
    """The key under which two names are the same file on a case-insensitive or normalising file system.

    Canonical decomposition both before and after case folding is the standard recipe for caseless
    matching: folding can produce characters that only decompose afterwards.
    """
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", text).casefold())


def _hard_violation(text: str) -> str | None:
    """Why ``text`` is hostile on every platform: an error, never a name to skip."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: no file system stores it
        return "not valid UTF-8"
    if not unicodedata.normalize("NFC", text):
        return "empty"
    if _CONTROL_CHARS.search(text):
        return "control character"
    if text.startswith("/"):
        return "absolute"
    parts = text.split("/")
    if "" in parts:
        return "empty component"
    if _DOT_COMPONENTS & set(parts):
        return "'.' or '..' component"
    return None


def _size_violation(text: str) -> str | None:
    if len(text.encode("utf-8")) > MAX_PATH_BYTES:
        return f"longer than {MAX_PATH_BYTES} bytes"
    if any(len(part.encode("utf-8")) > MAX_COMPONENT_BYTES for part in text.split("/")):
        return f"a component is longer than {MAX_COMPONENT_BYTES} bytes"
    return None


def _windows_violation(parts: list[str]) -> str | None:
    """Names only Windows refuses: a separator, a drive letter, characters, device names, trailing dots."""
    if any("\\" in part for part in parts):
        return "backslash"
    if _DRIVE_PREFIX.match(parts[0]):
        return "drive letter"
    if any(safe_component(part) != part for part in parts):
        return "not a valid Windows file name"
    return None


def _unsupported_reason(text: str) -> str | None:
    """Why this platform cannot create ``text`` though a server may hold it, or ``None`` when it can."""
    return _size_violation(text) or (_windows_violation(text.split("/")) if IS_WINDOWS else None)


@dataclass(frozen=True, slots=True, order=True)
class RelPath:
    """A path inside a run's artifacts, as the server reports it: validated on construction.

    Artifact paths come from the tracking server, so they are untrusted: one like
    ``../../x`` would otherwise write outside the run directory.
    """

    value: str

    def __post_init__(self) -> None:
        if (reason := _hard_violation(self.value)) is not None:
            raise UnsafePathError(f"unsafe artifact path {self.value!r}: {reason}")
        if (reason := _unsupported_reason(self.value)) is not None:
            raise UnsupportedNameError(f"unsupported artifact path {self.value!r}: {reason}")

    def __str__(self) -> str:
        return self.value

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.value.split("/"))

    def is_child_of(self, directory: RelPath | None) -> bool:
        """Directly below ``directory`` (``None`` is the run root)."""
        if directory is None:
            return len(self.parts) == 1
        return self.parts[:-1] == directory.parts

    def under(self, root: Path) -> Path:
        return root.joinpath(*self.parts)


def require_within(root: Path, path: Path) -> None:
    """Raise unless ``path`` is ``root`` or below it (lexically; symlinks are the user's choice)."""
    base, target = Path(os.path.abspath(root)), Path(os.path.abspath(path))
    if not target.is_relative_to(base):
        raise UnsafePathError(f"{path} is outside {root}")
