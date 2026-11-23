"""Which runs to take: status gate, time parsing and the MLflow filter string. No I/O."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum

FINISHED = frozenset({"FINISHED"})
UNFINISHED = frozenset({"RUNNING", "SCHEDULED"})
FAILED = frozenset({"FAILED", "KILLED"})


class RunStatus(Enum):
    """A run status MLflow names, for data that came back from disk; anything else is ``UNKNOWN``."""

    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, text: str) -> RunStatus:
        try:
            return cls(text)
        except ValueError:
            return cls.UNKNOWN

    @property
    def terminal(self) -> bool:
        """No longer changes its artifacts; an unknown status is not trusted to be."""
        return self.value in FINISHED | FAILED


_RELATIVE_WHEN = re.compile(r"(\d+)([mhdw])")
_WHEN_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


class RunStatusError(ValueError):
    """A run named explicitly has a status outside the allowed set."""


def allowed_statuses(unfinished: bool = False, failed: bool = False) -> frozenset[str]:
    return FINISHED | (UNFINISHED if unfinished else frozenset()) | (FAILED if failed else frozenset())


def parse_when(text: str, now: datetime) -> datetime:
    """``30m``/``6h``/``2d``/``1w`` before ``now``, or an ISO date/datetime (naive = local time)."""
    if match := _RELATIVE_WHEN.fullmatch(text):
        return now - timedelta(**{_WHEN_UNITS[match[2]]: int(match[1])})
    return datetime.fromisoformat(text).astimezone()


def quote(value: str) -> str:
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
        for key, _ in pairs:
            if "`" in key:  # keys sit inside backticks and the grammar has no escapes
                raise ValueError(f"{prefix[:-1]} key contains a backtick: {key}")
        clauses += [f"{prefix}.`{key}` = {quote(value)}" for key, value in pairs]
    if raw:
        clauses.append(raw)
    return " and ".join(clauses) or None
