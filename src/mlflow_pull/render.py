"""Turn run summaries into text: json, jsonl, a Markdown table, aligned text. Pure; returns str."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from glob import has_magic
from math import isfinite

from mlflow_pull.summary import MetricStats, RunSummary, matching_keys

STAT_SEPARATOR = ":"
TEXT_GAP = "  "


class Format(Enum):
    JSON = "json"
    JSONL = "jsonl"
    MD_TABLE = "md-table"
    TEXT = "text"


class RunField(Enum):
    RUN_ID = "run_id"
    NAME = "name"
    STATUS = "status"
    START = "start"
    END = "end"


class Stat(Enum):
    MIN = "min"
    LAST = "last"
    MIN_STEP = "min_step"
    COUNT = "count"


@dataclass(frozen=True, slots=True)
class FieldColumn:
    field: RunField

    @property
    def label(self) -> str:
        return self.field.value


@dataclass(frozen=True, slots=True)
class MetricColumn:
    key: str
    stat: Stat

    @property
    def label(self) -> str:
        return f"{self.key}{STAT_SEPARATOR}{self.stat.value}"


Column = FieldColumn | MetricColumn

BASE_COLUMNS: tuple[Column, ...] = tuple(FieldColumn(f) for f in (RunField.RUN_ID, RunField.NAME, RunField.STATUS))


def parse_columns(text: str, metric_globs: Sequence[str]) -> tuple[Column, ...]:
    """``name,status,val/loss:min``: a metric column is ``KEY:STAT``, split on the last colon.

    The key is literal (no glob characters). Every statistic except ``last`` needs a key matched by
    a ``--metric`` glob; ``last`` also works without one and then shows the latest value.
    """
    columns: list[Column] = []
    for item in (part.strip() for part in text.split(",")):
        key, separator, stat = item.rpartition(STAT_SEPARATOR)
        if not separator:
            try:
                columns.append(FieldColumn(RunField(item)))
            except ValueError:
                raise ValueError(f"unknown column {item!r}; fields: {_values(RunField)}") from None
            continue
        try:
            column = MetricColumn(key, Stat(stat))
        except ValueError:
            raise ValueError(f"column {item!r}: unknown statistic {stat!r}; one of {_values(Stat)}") from None
        if not key or has_magic(key):
            raise ValueError(f"column {item!r}: the metric key must be literal, got {key!r}")
        if column.stat is not Stat.LAST and not matching_keys(metric_globs, [key]):
            raise ValueError(f"column {item!r}: {key!r} matches no --metric glob")
        columns.append(column)
    return tuple(columns)


def _values(enum: type[Enum]) -> str:
    return ",".join(member.value for member in enum)


def default_columns(rows: Sequence[RunSummary], metric_globs: Sequence[str] = ()) -> tuple[Column, ...]:
    """Run id, name, status, then min and last of every metric the ``--metric`` globs matched.

    Without globs the latest value of every metric stands in for ``last``. Globs that match
    nothing give no metric columns: they never fall back to the latest values.
    """
    with_stats = bool(metric_globs)
    keys = sorted({key for row in rows for key in (row.metrics if with_stats else row.latest)})
    stats = (Stat.MIN, Stat.LAST) if with_stats else (Stat.LAST,)
    return BASE_COLUMNS + tuple(MetricColumn(key, stat) for key in keys for stat in stats)


def _number(value: float | int | None) -> str:
    """Floats use ``repr``, the shortest text that parses back to the same float, so a table read by
    a machine loses nothing (a whole float shows as ``1.0``, apart from the integer steps and counts)."""
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def _cell(row: RunSummary, column: Column) -> str:
    if isinstance(column, FieldColumn):
        match column.field:
            case RunField.RUN_ID:
                return row.run_id
            case RunField.NAME:
                return row.name
            case RunField.STATUS:
                return row.status
            case RunField.START:
                return row.start.isoformat(timespec="seconds")
            case RunField.END:
                return "" if row.end is None else row.end.isoformat(timespec="seconds")
    stats = row.metrics.get(column.key)
    if stats is None:  # no history fetched: only the latest value is known
        return _number(row.latest.get(column.key)) if column.stat is Stat.LAST else ""
    match column.stat:
        case Stat.MIN:
            return _number(stats.min)
        case Stat.LAST:
            return _number(stats.last)
        case Stat.MIN_STEP:
            return _number(stats.min_step)
        case Stat.COUNT:
            return _number(stats.count)


def _finite(value: float) -> float | None:
    """Strict JSON has no NaN or infinity: those become null."""
    return value if isfinite(value) else None


def _stats_record(stats: MetricStats) -> dict[str, object]:
    return {
        "min": _finite(stats.min), "min_step": stats.min_step,
        "last": _finite(stats.last), "last_step": stats.last_step, "count": stats.count,
    }


def record(row: RunSummary) -> dict[str, object]:
    """The JSON shape of one run; non-finite metric values are null."""
    return {
        "run_id": row.run_id,
        "name": row.name,
        "status": row.status,
        "start": row.start.isoformat(timespec="seconds"),
        "end": None if row.end is None else row.end.isoformat(timespec="seconds"),
        "params": dict(row.params),
        "tags": dict(row.tags),
        "latest": {key: _finite(value) for key, value in row.latest.items()},
        "metrics": {key: _stats_record(stats) for key, stats in row.metrics.items()},
        "history": {key: [[p.step, _finite(p.value)] for p in points] for key, points in row.history.items()},
    }


def printable(cell: str) -> str:
    """Server-controlled text must not drive the terminal: non-printable characters become escapes.

    Control characters of every range (C0, DEL, C1 such as CSI) and format characters (bidi overrides)
    are not printable, so ``isprintable`` is the whole rule; every human-readable line holding names
    from a server goes through this one function.
    """
    return "".join(ch if ch.isprintable() else _escape_char(ch) for ch in cell)


def _escape_char(ch: str) -> str:
    code = ord(ch)
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


def _table(
    rows: Sequence[RunSummary], columns: Sequence[Column], clean: Callable[[str], str]
) -> list[list[str]]:
    lines = [[column.label for column in columns]] + [[_cell(row, column) for column in columns] for row in rows]
    return [[clean(cell) for cell in line] for line in lines]


def _escape_md(cell: str) -> str:
    """Backslash first, so the backslash added for ``|`` is not doubled; line breaks end the row."""
    return cell.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _md_table(rows: Sequence[RunSummary], columns: Sequence[Column]) -> str:
    """Header and separator rows always; no runs gives the headers only."""
    header, *body = _table(rows, columns, lambda cell: printable(_escape_md(cell)))
    lines = [header, ["---"] * len(columns), *body]
    return "\n".join(f"| {' | '.join(line)} |" for line in lines)


def _text(rows: Sequence[RunSummary], columns: Sequence[Column]) -> str:
    table = _table(rows, columns, printable)
    widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
    return "\n".join(TEXT_GAP.join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip() for line in table)


def dumps_strict(obj: object, **kwargs: object) -> str:
    return json.dumps(obj, allow_nan=False, **kwargs)  # a leaked NaN fails loudly


def render(fmt: Format, rows: Sequence[RunSummary], columns: Sequence[Column]) -> str:
    """``columns`` only shapes the table formats. No runs: ``[]``, an empty string (jsonl), headers only."""
    match fmt:
        case Format.JSON:
            return dumps_strict([record(row) for row in rows], indent=2)
        case Format.JSONL:
            return "\n".join(dumps_strict(record(row)) for row in rows)
        case Format.MD_TABLE:
            return _md_table(rows, columns)
        case Format.TEXT:
            return _text(rows, columns)
