"""What a run contains, as plain data: metric statistics, downsampling, glob matching. No I/O."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from math import isnan
from typing import TypeVar

from mlflow.entities import Metric, Run

T = TypeVar("T")

MIN_HISTORY_POINTS = 2  # a downsampled history keeps its first and last point


@dataclass(frozen=True, slots=True)
class Point:
    step: int
    value: float


@dataclass(frozen=True, slots=True)
class MetricStats:
    """Statistics over the full history of one metric of one run."""

    min: float
    min_step: int
    last: float
    last_step: int
    count: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Read-only view of a run. ``metrics`` and ``history`` only hold the matched metric keys;
    ``latest`` holds every metric's latest value, as the run search returns it."""

    run_id: str
    name: str
    status: str
    start: datetime
    end: datetime | None  # None while the run has not ended
    params: Mapping[str, str]
    tags: Mapping[str, str]
    latest: Mapping[str, float]
    metrics: Mapping[str, MetricStats]
    history: Mapping[str, tuple[Point, ...]]


def matching_keys(globs: Iterable[str], keys: Iterable[str]) -> list[str]:
    """Keys matched by any of the ``fnmatch`` globs, sorted."""
    patterns = tuple(globs)
    return sorted(key for key in keys if any(fnmatchcase(key, pattern) for pattern in patterns))


def to_points(history: Iterable[Metric]) -> tuple[Point, ...]:
    """Points by step, then logging time; a step logged twice keeps both values."""
    ordered = sorted(history, key=lambda m: (m.step, m.timestamp))
    return tuple(Point(m.step, m.value) for m in ordered)


def summarise_metric(points: Sequence[Point]) -> MetricStats:
    """Minimum (earliest step on ties; NaN never wins), last point and count of a non-empty history."""
    if not points:
        raise ValueError("a metric history needs at least one point")
    lowest = min(points, key=lambda p: (isnan(p.value), p.value, p.step))
    final = points[-1]  # points are ordered by step
    return MetricStats(lowest.value, lowest.step, final.value, final.step, len(points))


def downsample(points: Sequence[T], limit: int) -> tuple[T, ...]:
    """At most ``limit`` points at an even stride, always keeping the first and the last."""
    if limit < MIN_HISTORY_POINTS:
        raise ValueError(f"limit must be at least {MIN_HISTORY_POINTS}, got {limit}")
    if len(points) <= limit:
        return tuple(points)
    # Integer floor keeps the indices strictly increasing: the stride is above 1 here.
    last_index = len(points) - 1
    return tuple(points[i * last_index // (limit - 1)] for i in range(limit))


def from_millis(millis: int) -> datetime:
    return datetime.fromtimestamp(millis / 1000, timezone.utc)


def summarise_run(
    run: Run, histories: Mapping[str, Sequence[Point]], history_limit: int | None = None
) -> RunSummary:
    """``histories`` holds the full ordered history of each matched metric key."""
    end_time = run.info.end_time
    return RunSummary(
        run_id=run.info.run_id,
        name=run.info.run_name or "",
        status=run.info.status,
        start=from_millis(run.info.start_time),
        end=None if end_time is None else from_millis(end_time),
        params=dict(run.data.params),
        tags=dict(run.data.tags),
        latest=dict(run.data.metrics),
        metrics={key: summarise_metric(points) for key, points in histories.items()},
        history={} if history_limit is None
        else {key: downsample(points, history_limit) for key, points in histories.items()},
    )


def in_display_order(rows: Iterable[RunSummary]) -> list[RunSummary]:
    return sorted(rows, key=lambda row: (row.start, row.run_id))
