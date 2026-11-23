"""Glob filters over artifact paths. Pure.

``*`` and ``?`` stay inside one path component and ``**`` as a whole component crosses
directories, so ``plots/*.png`` does not match ``plots/a/b.png`` while ``plots/**`` does.
``fnmatch`` is not used because its ``*`` crosses ``/``. A leading ``!`` excludes.
``[`` is an ordinary character: there are no character classes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from mlflow_pull.paths import RelPath

SEPARATOR = "/"
CROSS_DIRECTORIES = "**"
EXCLUDE_PREFIX = "!"
DOT_SEGMENTS = (".", "..")
_STAR_RUN = re.compile(r"\*+")


@lru_cache(maxsize=None)
def _segment_regex(segment: str) -> re.Pattern[str]:
    """One component: ``*`` is any run of characters but ``/``, ``?`` is one such character."""
    pieces = (
        "[^/]*" if char == "*" else "[^/]" if char == "?" else re.escape(char)
        for char in _STAR_RUN.sub("*", segment)
    )
    return re.compile("".join(pieces))


@lru_cache(maxsize=None)
def _path_regex(segments: tuple[str, ...]) -> re.Pattern[str]:
    pieces = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == CROSS_DIRECTORIES:
            pieces.append(".+" if last else "(?:[^/]+/)*")
        else:
            pieces.append(_segment_regex(segment).pattern + ("" if last else SEPARATOR))
    return re.compile("".join(pieces))


@dataclass(frozen=True, slots=True)
class Glob:
    segments: tuple[str, ...]

    @classmethod
    def parse(cls, text: str) -> Glob:
        if not text:
            raise ValueError("empty path glob")
        if "\\" in text:
            raise ValueError(f"path glob {text!r}: use / as the separator")
        segments = tuple(text.split(SEPARATOR))
        if segments[-1] == "" and "" not in segments[:-1] and segments[0] != "":
            raise ValueError(
                f"path glob {text!r}: a trailing / matches nothing; "
                f"write {text + CROSS_DIRECTORIES!r} for everything below it"
            )
        if "" in segments:
            raise ValueError(f"path glob {text!r}: relative paths only, no empty components")
        for dots in DOT_SEGMENTS:
            if dots in segments:
                raise ValueError(f"path glob {text!r}: '{dots}' never matches an artifact")
        return cls(segments)

    def matches(self, path: RelPath) -> bool:
        return _path_regex(self.segments).fullmatch(path.value) is not None

    def may_match_below(self, directory: RelPath) -> bool:
        """Whether some path strictly inside ``directory`` can match; ``False`` only when provably none can."""
        parts = directory.parts
        for index, segment in enumerate(self.segments):
            if segment == CROSS_DIRECTORIES or index == len(parts):
                return True
            if not _segment_regex(segment).fullmatch(parts[index]):
                return False
        return False  # the glob ends at or above the directory

    def covers_below(self, directory: RelPath) -> bool:
        """Whether every path inside ``directory`` matches (``dir/**`` with ``dir`` matching)."""
        if len(self.segments) < 2 or self.segments[-1] != CROSS_DIRECTORIES:
            return False
        return _path_regex(self.segments[:-1]).fullmatch(directory.value) is not None


@dataclass(frozen=True, slots=True)
class PathFilter:
    """Include globs (none: everything) minus exclude globs; excludes win."""

    includes: tuple[Glob, ...] = ()
    excludes: tuple[Glob, ...] = ()

    @classmethod
    def parse(cls, specs: Sequence[str]) -> PathFilter:
        excluded = [spec.removeprefix(EXCLUDE_PREFIX) for spec in specs if spec.startswith(EXCLUDE_PREFIX)]
        included = [spec for spec in specs if not spec.startswith(EXCLUDE_PREFIX)]
        return cls(tuple(map(Glob.parse, included)), tuple(map(Glob.parse, excluded)))

    def accepts(self, path: RelPath) -> bool:
        included = not self.includes or any(glob.matches(path) for glob in self.includes)
        return included and not any(glob.matches(path) for glob in self.excludes)

    def descends_into(self, directory: RelPath) -> bool:
        """Whether listing ``directory`` can find an accepted file: the only test that prunes the walk."""
        if any(glob.covers_below(directory) for glob in self.excludes):
            return False
        return not self.includes or any(glob.may_match_below(directory) for glob in self.includes)


EVERYTHING = PathFilter()
