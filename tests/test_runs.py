from __future__ import annotations

import json
import math
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import Result
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from mlflow_pull import (
    PARENT_RUN_TAG,
    MlflowGateway,
    RunGateway,
    downsample,
    matching_keys,
    resolve_run_list,
    summarise_metric,
    summarise_runs,
)
from mlflow_pull.render import default_columns, parse_columns
from mlflow_pull.selection import build_filter
from mlflow_pull.plan import MANIFEST_NAME
from mlflow_pull.summary import Point, to_points

Cli = Callable[..., Result]

START_STEP_MS = 1_000  # run n starts n * START_STEP_MS after the epoch: a deterministic order


@pytest.fixture
def metric_runs(client: MlflowClient, tmp_path: Path) -> dict[str, str]:
    """exp: best (FINISHED, key abc, val/loss minimum in the middle) -> child (nested),
    worse (FINISHED, key xyz), live (RUNNING, key abc).  other: one run without metrics."""
    starts = iter(range(START_STEP_MS, 100 * START_STEP_MS, START_STEP_MS))

    def make(
        experiment_id: str,
        name: str,
        metrics: dict[str, list[float]],
        status: str | None = "FINISHED",
        tags: dict[str, str] | None = None,
    ) -> str:
        run_id = client.create_run(
            experiment_id, start_time=next(starts), tags=tags or {}, run_name=name
        ).info.run_id
        for key, values in metrics.items():
            for step, value in enumerate(values):
                client.log_metric(run_id, key, value, step=step)
        if status is not None:  # None leaves the run RUNNING
            client.set_terminated(run_id, status)
        return run_id

    exp = client.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    other = client.create_experiment("other", artifact_location=(tmp_path / "art2").as_uri())
    best = make(
        exp, "best",
        {"val/loss": [1.0, 0.6, 0.2, 0.4, 0.3], "train/loss": [2.0, 1.0, 0.5]},
        tags={"run_key": "abc"},
    )
    client.log_param(best, "lr", "0.1")
    client.log_text(best, "x", "best.txt")
    child = make(exp, "child", {"val/loss": [0.5]}, tags={PARENT_RUN_TAG: best})
    worse = make(exp, "worse", {"val/loss": [1.0, 0.9]}, tags={"run_key": "xyz"})
    live = make(exp, "live", {"val/loss": [0.8, 0.7]}, status=None, tags={"run_key": "abc"})
    solo = make(other, "solo", {})
    return {"best": best, "child": child, "worse": worse, "live": live, "solo": solo}


class CountingGateway(MlflowGateway):
    """A real gateway that records every history read."""

    def __init__(self, client: MlflowClient) -> None:
        super().__init__(client)
        self.history_calls: list[tuple[str, str]] = []

    def metric_history(self, run_id: str, key: str) -> list[Metric]:
        self.history_calls.append((run_id, key))
        return super().metric_history(run_id, key)


def _summaries(gateway: RunGateway, globs: list[str], filter_string: str | None = None, **kwargs):
    return summarise_runs(gateway, resolve_run_list(gateway, ["exp"], filter_string), globs, **kwargs)


def _by_name(rows):
    return {row.name: row for row in rows}


# --- statistics ----------------------------------------------------------------------


def test_stats_come_from_the_full_history(gateway: RunGateway, metric_runs: dict[str, str]) -> None:
    stats = _by_name(_summaries(gateway, ["val/*"]))["best"].metrics["val/loss"]

    assert (stats.min, stats.min_step) == (0.2, 2)  # the minimum is not the last point
    assert (stats.last, stats.last_step, stats.count) == (0.3, 4, 5)


def test_running_and_nested_runs_are_listed(gateway: RunGateway, metric_runs: dict[str, str]) -> None:
    rows = _by_name(_summaries(gateway, ["val/loss"]))

    assert [row.name for row in _summaries(gateway, [])] == ["best", "child", "worse", "live"]
    assert rows["live"].status == "RUNNING" and rows["live"].end is None
    assert rows["live"].metrics["val/loss"].last == 0.7
    assert rows["child"].tags[PARENT_RUN_TAG] == metric_runs["best"]
    assert rows["best"].end is not None and rows["best"].params == {"lr": "0.1"}


NAN = float("nan")
INF = float("inf")


def test_summarise_metric_ties_take_the_earliest_step_and_nan_never_wins() -> None:
    stats = summarise_metric([Point(0, 2.0), Point(1, 1.0), Point(2, 1.0), Point(3, NAN)])
    assert (stats.min, stats.min_step, stats.count) == (1.0, 1, 4)

    with pytest.raises(ValueError):
        summarise_metric([])


@pytest.mark.parametrize(
    ("values", "min_step", "minimum"),
    [
        ([NAN, 3.0, 2.0], 2, 2.0),  # NaN first still loses
        ([INF, 1.0], 1, 1.0),
        ([INF, INF], 0, INF),
        ([1.0, -INF, -INF], 1, -INF),
        ([INF, NAN], 0, INF),  # infinity is a number, NaN is not
    ],
)
def test_summarise_metric_orders_non_finite_values(values: list[float], min_step: int, minimum: float) -> None:
    stats = summarise_metric([Point(step, value) for step, value in enumerate(values)])

    assert (stats.min, stats.min_step, stats.count) == (minimum, min_step, len(values))


def test_summarise_metric_all_nan_keeps_the_first_step() -> None:
    stats = summarise_metric([Point(4, NAN), Point(5, NAN)])

    assert math.isnan(stats.min) and stats.min_step == 4 and math.isnan(stats.last)


def test_summarise_metric_duplicate_step_keeps_both_points() -> None:
    stats = summarise_metric([Point(1, 2.0), Point(1, 1.0)])

    assert (stats.min, stats.min_step, stats.last, stats.last_step, stats.count) == (1.0, 1, 1.0, 1, 2)


def test_ordering_is_to_points_job_not_summarise_metrics() -> None:
    # Same step logged twice: the later timestamp is the last point, whatever order they arrive in.
    metrics = [Metric("k", 5.0, 200, 0), Metric("k", 1.0, 100, 0), Metric("k", 3.0, 50, 1)]
    ordered = to_points(metrics)

    assert ordered == (Point(0, 1.0), Point(0, 5.0), Point(1, 3.0))
    assert summarise_metric(ordered).last == 3.0
    # Unsorted input is taken as given: the final point is simply the last element.
    assert summarise_metric([Point(3, 1.0), Point(1, 2.0)]).last_step == 1


def test_glob_selects_metric_keys(gateway: RunGateway, metric_runs: dict[str, str]) -> None:
    assert matching_keys(["val/*"], ["val/loss", "train/loss", "val"]) == ["val/loss"]
    assert matching_keys(["*/loss", "val"], ["val/loss", "train/loss", "val"]) == ["train/loss", "val", "val/loss"]

    best = _by_name(_summaries(gateway, ["*/loss"]))["best"]
    assert set(best.metrics) == {"val/loss", "train/loss"}


def test_without_a_metric_glob_no_history_is_read(
    client: MlflowClient, metric_runs: dict[str, str]
) -> None:
    counting = CountingGateway(client)
    rows = _summaries(counting, [])

    assert counting.history_calls == []
    assert all(row.metrics == {} for row in rows)
    assert _by_name(rows)["best"].latest["val/loss"] == 0.3  # the latest values still come with the run


def test_one_history_read_per_run_per_matched_key(
    client: MlflowClient, metric_runs: dict[str, str]
) -> None:
    counting = CountingGateway(client)
    _summaries(counting, ["*/loss"], jobs=3)

    assert sorted(counting.history_calls) == sorted(
        [(metric_runs["best"], "val/loss"), (metric_runs["best"], "train/loss"),
         (metric_runs["child"], "val/loss"), (metric_runs["worse"], "val/loss"),
         (metric_runs["live"], "val/loss")]
    )


# --- downsampling --------------------------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, 2, 3, 7, 10, 99, 1000])
@pytest.mark.parametrize("limit", [2, 3, 5, 10, 100])
def test_downsample_respects_limit_and_keeps_both_ends(length: int, limit: int) -> None:
    points = list(range(length))
    kept = downsample(points, limit)

    assert len(kept) <= limit
    assert list(kept) == sorted(set(kept))  # strictly increasing: no point twice
    if points:
        assert kept[0] == points[0] and kept[-1] == points[-1]
    if length <= limit:
        assert list(kept) == points


def test_downsample_rejects_a_limit_below_two() -> None:
    with pytest.raises(ValueError):
        downsample([1, 2, 3], 1)


def test_history_is_downsampled_in_the_summary(gateway: RunGateway, metric_runs: dict[str, str]) -> None:
    best = _by_name(_summaries(gateway, ["val/loss"], history_limit=3))["best"]

    assert best.history["val/loss"] == (Point(0, 1.0), Point(2, 0.2), Point(4, 0.3))
    assert best.metrics["val/loss"].count == 5  # statistics still use every point


# --- CLI -----------------------------------------------------------------------------


def _json(cli: Cli, *args: str) -> list[dict]:
    result = cli("runs", "--format", "json", *args)
    assert result.exit_code == 0, result.output
    return _strict_json(result.stdout)


def test_cli_json_round_trips(cli: Cli, metric_runs: dict[str, str]) -> None:
    records = _json(cli, "-e", "exp", "--metric", "val/loss", "--history", "3")
    by_name = {record["name"]: record for record in records}

    assert [record["name"] for record in records] == ["best", "child", "worse", "live"]
    assert by_name["best"]["metrics"]["val/loss"] == {
        "min": 0.2, "min_step": 2, "last": 0.3, "last_step": 4, "count": 5,
    }
    assert by_name["best"]["history"]["val/loss"] == [[0, 1.0], [2, 0.2], [4, 0.3]]
    assert by_name["best"]["start"] == "1970-01-01T00:00:01+00:00"
    assert by_name["live"]["end"] is None and by_name["live"]["status"] == "RUNNING"


def _strict_json(text: str):
    """``json.loads`` that rejects the non-standard NaN, Infinity and -Infinity."""

    def reject(constant: str):
        raise AssertionError(f"non-standard JSON constant {constant}")

    return json.loads(text, parse_constant=reject)


def test_cli_jsonl_is_one_record_per_line(cli: Cli, metric_runs: dict[str, str]) -> None:
    result = cli("runs", "-e", "exp", "--format", "jsonl")

    assert result.exit_code == 0, result.output
    assert [json.loads(line)["name"] for line in result.stdout.splitlines()] == [
        "best", "child", "worse", "live",
    ]


def test_cli_without_experiment_reads_every_experiment(cli: Cli, metric_runs: dict[str, str]) -> None:
    names = [record["name"] for record in _json(cli)]
    assert names == ["best", "child", "worse", "live", "solo"]


def test_cli_key_filters_on_the_tag(cli: Cli, metric_runs: dict[str, str]) -> None:
    assert [r["name"] for r in _json(cli, "-e", "exp", "--key", "abc")] == ["best", "live"]
    assert [r["name"] for r in _json(cli, "-e", "exp", "--key", "xyz")] == ["worse"]
    empty = cli("runs", "--format", "json", "-e", "exp", "--key", "abc", "--tag-name", "missing")
    assert (empty.exit_code, _strict_json(empty.stdout)) == (3, [])


def test_cli_key_tag_name_comes_from_the_environment(
    cli: Cli, metric_runs: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLPULL_KEY_TAG", "mlflow.runName")
    assert [r["name"] for r in _json(cli, "-e", "exp", "--key", "worse")] == ["worse"]


def test_cli_experiment_accepts_key(cli: Cli, metric_runs: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("experiment", "exp", "--key", "abc", "-o", str(out))  # live is RUNNING: skipped

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.rglob("*") if p.is_file() and p.name != MANIFEST_NAME) == ["best.txt"]


def test_cli_md_table_exact(cli: Cli, metric_runs: dict[str, str]) -> None:
    result = cli(
        "runs", "-e", "exp", "--key", "abc", "--metric", "val/*", "--format", "md-table",
        "--columns", "name,status,val/loss:min,val/loss:last,val/loss:min_step,val/loss:count",
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "| name | status | val/loss:min | val/loss:last | val/loss:min_step | val/loss:count |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| best | FINISHED | 0.2 | 0.3 | 2 | 5 |\n"
        "| live | RUNNING | 0.7 | 0.7 | 1 | 2 |\n"
    )


def test_cli_default_columns_hold_min_and_last_of_every_matched_metric(
    cli: Cli, metric_runs: dict[str, str]
) -> None:
    result = cli("runs", "-e", "exp", "--key", "xyz", "--metric", "*/loss", "--format", "md-table")

    assert result.stdout.splitlines()[0] == (
        "| run_id | name | status | val/loss:min | val/loss:last |"
    )


def test_cli_text_without_metric_shows_latest_values(cli: Cli, metric_runs: dict[str, str]) -> None:
    result = cli("runs", "-e", "exp", "--key", "xyz", "--columns", "name,status")

    assert result.stdout == "name   status\nworse  FINISHED\n"

    latest = cli("runs", "-e", "exp", "--key", "xyz")
    assert latest.stdout.splitlines()[0].split() == ["run_id", "name", "status", "val/loss:last"]
    assert latest.stdout.splitlines()[1].split()[-1] == "0.9"


@pytest.mark.parametrize(
    "args",
    [
        ("--columns", "name,nonsense"),
        ("--columns", "val/loss:min"),  # no --metric glob covers the key
        ("--metric", "val/*", "--columns", "val/loss:median"),
        ("--metric", "val/*", "--columns", "train/loss:min"),
        ("--history", "5"),  # history needs --metric
        ("--metric", "val/*", "--history", "1"),
        ("--format", "yaml"),
        ("--tag", "no-equals-sign"),
    ],
)
def test_cli_usage_errors_exit_2(cli: Cli, metric_runs: dict[str, str], args: tuple[str, ...]) -> None:
    assert cli("runs", "-e", "exp", *args).exit_code == 2


def test_cli_unknown_experiment_exits_1(cli: Cli, metric_runs: dict[str, str], assert_fails_cleanly) -> None:
    result = cli("runs", "-e", "no-such-experiment")

    assert_fails_cleanly(result, 1)
    assert result.stdout == ""


def test_cli_experiment_glob_without_match_exits_3(cli: Cli, metric_runs: dict[str, str]) -> None:
    result = cli("runs", "-e", "nothing*")

    assert result.exit_code == 3
    assert "no experiment matches" in result.stderr


@pytest.mark.parametrize("fmt", ["json", "jsonl", "md-table", "text"])
def test_cli_experiment_glob_without_match_renders_what_an_empty_selection_renders(
    cli: Cli, metric_runs: dict[str, str], fmt: str
) -> None:
    empty = cli("runs", "-e", "exp", "--tag", "phase=nope", "--format", fmt)
    result = cli("runs", "-e", "nothing*", "--format", fmt)

    assert result.exit_code == empty.exit_code == 3
    assert result.stdout == empty.stdout
    assert "no experiment matches nothing*" in result.stderr


def test_cli_a_start_time_no_calendar_holds_fails_with_one_line(
    cli: Cli, client: MlflowClient, metric_runs: dict[str, str], assert_fails_cleanly
) -> None:
    client.create_run(client.get_experiment_by_name("exp").experiment_id, start_time=2**62)

    assert "year" in assert_fails_cleanly(cli("runs", "-e", "exp"), 1)


def test_cli_an_overflowing_start_time_fails_with_one_line(
    cli: Cli, metric_runs: dict[str, str], monkeypatch: pytest.MonkeyPatch, assert_fails_cleanly
) -> None:
    def overflow(*_args: object) -> None:
        raise OverflowError("timestamp out of range for platform time_t")

    monkeypatch.setattr("mlflow_pull.cli.summarise_runs", overflow)

    assert "time_t" in assert_fails_cleanly(cli("runs", "-e", "exp"), 1)


def test_cli_boto_errors_fail_with_one_line_when_botocore_is_installed(
    cli: Cli, metric_runs: dict[str, str], monkeypatch: pytest.MonkeyPatch, assert_fails_cleanly
) -> None:
    class BotoCoreError(Exception):
        pass

    class ClientError(Exception):
        pass

    botocore = types.ModuleType("botocore")
    exceptions = types.ModuleType("botocore.exceptions")
    exceptions.BotoCoreError, exceptions.ClientError = BotoCoreError, ClientError
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)

    def denied(*_args: object) -> None:
        raise ClientError("An error occurred (AccessDenied) when calling GetObject")

    monkeypatch.setattr("mlflow_pull.cli.summarise_runs", denied)

    assert "AccessDenied" in assert_fails_cleanly(cli("runs", "-e", "exp"), 1)


def test_cli_boto3_errors_fail_with_one_line_when_boto3_is_installed(
    cli: Cli, metric_runs: dict[str, str], monkeypatch: pytest.MonkeyPatch, assert_fails_cleanly
) -> None:
    """``download_file`` raises ``RetriesExceededError`` and kin, boto3's own family, when its retries run out."""

    class Boto3Error(Exception):
        pass

    class RetriesExceededError(Boto3Error):
        pass

    boto3 = types.ModuleType("boto3")
    exceptions = types.ModuleType("boto3.exceptions")
    exceptions.Boto3Error = Boto3Error
    monkeypatch.setitem(sys.modules, "botocore", None)  # only boto3 is mapped here
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "boto3.exceptions", exceptions)

    def exhausted(*_args: object) -> None:
        raise RetriesExceededError("Max Retries Exceeded")

    monkeypatch.setattr("mlflow_pull.cli.summarise_runs", exhausted)

    assert "Max Retries" in assert_fails_cleanly(cli("runs", "-e", "exp"), 1)


def test_cli_boto_errors_are_not_mapped_without_botocore_or_boto3(
    cli: Cli, metric_runs: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "botocore", None)  # makes the import fail
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setattr("mlflow_pull.cli.summarise_runs", lambda *_: (_ for _ in ()).throw(KeyError("boom")))

    assert isinstance(cli("runs", "-e", "exp").exception, KeyError)  # no blanket except Exception


# --- CLI edge cases ------------------------------------------------------------------


def _make_run(
    client: MlflowClient,
    experiment_id: str,
    name: str,
    start: int = START_STEP_MS,
    metrics: dict[str, list[float]] | None = None,
) -> str:
    run_id = client.create_run(experiment_id, start_time=start, run_name=name).info.run_id
    for key, values in (metrics or {}).items():
        for step, value in enumerate(values):
            client.log_metric(run_id, key, value, step=step)
    client.set_terminated(run_id, "FINISHED")
    return run_id


@pytest.fixture
def nan_runs(client: MlflowClient, tmp_path: Path) -> None:
    exp = client.create_experiment("nan", artifact_location=(tmp_path / "nan").as_uri())
    _make_run(client, exp, "mixed", start=1_000, metrics={"val/loss": [NAN, 0.5, 0.2]})
    _make_run(client, exp, "allnan", start=2_000, metrics={"val/loss": [NAN, NAN]})


def test_cli_json_has_no_non_finite_numbers(cli: Cli, nan_runs: None) -> None:
    by_name = {r["name"]: r for r in _json(cli, "-e", "nan", "--metric", "val/loss", "--history", "2")}

    assert by_name["mixed"]["metrics"]["val/loss"]["min"] == 0.2
    assert by_name["allnan"]["metrics"]["val/loss"]["min"] is None
    assert by_name["allnan"]["metrics"]["val/loss"]["last"] is None
    assert by_name["allnan"]["latest"]["val/loss"] is None
    assert by_name["allnan"]["history"]["val/loss"] == [[0, None], [1, None]]


def test_cli_jsonl_has_no_non_finite_numbers(cli: Cli, nan_runs: None) -> None:
    result = cli("runs", "-e", "nan", "--format", "jsonl", "--metric", "val/loss")

    assert result.exit_code == 0, result.output
    assert [_strict_json(line)["metrics"]["val/loss"]["min"] for line in result.stdout.splitlines()] == [0.2, None]


def test_cli_tables_show_nan_as_nan_and_floats_round_trip(
    cli: Cli, nan_runs: None, client: MlflowClient
) -> None:
    _make_run(client, client.get_experiment_by_name("nan").experiment_id, "precise", start=3_000, metrics={"val/loss": [0.1234567891]})
    result = cli("runs", "-e", "nan", "--metric", "val/loss", "--columns", "name,val/loss:min,val/loss:last")

    assert [line.split() for line in result.stdout.splitlines()] == [
        ["name", "val/loss:min", "val/loss:last"],
        ["mixed", "0.2", "0.2"],
        ["allnan", "nan", "nan"],
        ["precise", "0.1234567891", "0.1234567891"],
    ]


@pytest.fixture
def odd_names(client: MlflowClient, tmp_path: Path) -> None:
    exp = client.create_experiment("odd", artifact_location=(tmp_path / "odd").as_uri())
    _make_run(client, exp, "a|b\\c\nd\x1b[31mred\rx\x00")


def test_cli_md_table_escapes_cells(cli: Cli, odd_names: None) -> None:
    result = cli("runs", "-e", "odd", "--format", "md-table", "--columns", "name")

    assert result.stdout.splitlines() == [
        "| name |",
        "| --- |",
        "| a\\|b\\\\c d\\x1b[31mred x\\x00 |",
    ]


def test_cli_text_escapes_control_characters(cli: Cli, odd_names: None) -> None:
    result = cli("runs", "-e", "odd", "--columns", "name")

    assert result.stdout.splitlines() == ["name", "a|b\\c\\x0ad\\x1b[31mred\\x0dx\\x00"]
    assert "\x1b" not in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ("--key", "it's \"both\""),
        ("--key", "abc", "--tag-name", "bad`name"),
        ("--tag", "bad`name=v"),
        ("--param", "bad`name=v"),
    ],
)
def test_cli_unsafe_selection_exits_2(cli: Cli, metric_runs: dict[str, str], args: tuple[str, ...]) -> None:
    assert cli("runs", "-e", "exp", *args).exit_code == 2


def test_cli_key_tag_name_from_environment_is_checked(
    cli: Cli, metric_runs: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLPULL_KEY_TAG", "bad`name")

    assert cli("runs", "-e", "exp", "--key", "abc").exit_code == 2


def test_build_filter_rejects_a_backtick_in_any_key() -> None:
    for kwargs in ({"tags": [("a`b", "v")]}, {"params": [("a`b", "v")]}):
        with pytest.raises(ValueError, match="backtick"):
            build_filter(**kwargs)
    assert build_filter(tags=[("a b", "v")]) == "tags.`a b` = 'v'"


def test_cli_columns_split_on_the_last_colon(cli: Cli, client: MlflowClient, tmp_path: Path) -> None:
    exp = client.create_experiment("colon", artifact_location=(tmp_path / "colon").as_uri())
    _make_run(client, exp, "r", metrics={"a:b": [3.0, 1.0]})

    result = cli("runs", "-e", "colon", "--metric", "a:b", "--columns", "a:b:min,a:b:count")

    assert result.exit_code == 0, result.output
    assert [line.split() for line in result.stdout.splitlines()] == [["a:b:min", "a:b:count"], ["1.0", "2"]]


def test_cli_metric_glob_matching_nothing_adds_no_metric_columns(cli: Cli, metric_runs: dict[str, str]) -> None:
    result = cli("runs", "-e", "exp", "--key", "xyz", "--metric", "nope/*", "--format", "md-table")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == "| run_id | name | status |"


def test_cli_last_column_works_without_a_metric_glob(cli: Cli, metric_runs: dict[str, str]) -> None:
    result = cli("runs", "-e", "exp", "--key", "xyz", "--columns", "name,val/loss:last")

    assert result.exit_code == 0, result.output
    assert [line.split() for line in result.stdout.splitlines()] == [["name", "val/loss:last"], ["worse", "0.9"]]


@pytest.mark.parametrize("text", ["val/*:min", "val/*:last", "val/lo?s:last", "val/[l]oss:min", ":last"])
def test_parse_columns_rejects_glob_characters_in_the_key(text: str) -> None:
    with pytest.raises(ValueError, match="literal"):
        parse_columns(text, ["val/*"])


def test_parse_columns_only_last_works_without_a_glob() -> None:
    assert parse_columns("val/loss:last", [])
    for stat in ("min", "min_step", "count"):
        with pytest.raises(ValueError, match="no --metric glob"):
            parse_columns(f"val/loss:{stat}", [])
    with pytest.raises(ValueError, match="no --metric glob"):
        parse_columns("typo/loss:min", ["val/*"])


def test_default_columns_follow_the_globs_not_whether_anything_matched(
    gateway: RunGateway, metric_runs: dict[str, str]
) -> None:
    rows = _summaries(gateway, ["nope/*"])

    assert [c.label for c in default_columns(rows, ["nope/*"])] == ["run_id", "name", "status"]
    assert "val/loss:last" in [c.label for c in default_columns(rows)]


def test_cli_equal_start_times_are_ordered_by_run_id(cli: Cli, client: MlflowClient, tmp_path: Path) -> None:
    exp = client.create_experiment("ties", artifact_location=(tmp_path / "ties").as_uri())
    ids = [_make_run(client, exp, name, start=5_000) for name in "cab"]

    assert [r["run_id"] for r in _json(cli, "-e", "ties")] == sorted(ids)


def test_cli_deleted_experiment_is_excluded_by_default(
    cli: Cli, client: MlflowClient, metric_runs: dict[str, str], tmp_path: Path
) -> None:
    gone = client.create_experiment("gone", artifact_location=(tmp_path / "gone").as_uri())
    _make_run(client, gone, "ghost")
    client.delete_experiment(gone)

    assert "ghost" not in [r["name"] for r in _json(cli)]


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("json", "[]\n"),
        ("md-table", "| run_id | name | status |\n| --- | --- | --- |\n"),
        ("text", "run_id  name  status\n"),
        ("jsonl", ""),  # nothing at all, not a blank line
    ],
)
def test_cli_empty_result_per_format(cli: Cli, metric_runs: dict[str, str], fmt: str, expected: str) -> None:
    result = cli("runs", "-e", "exp", "--key", "no-such-key", "--format", fmt)

    assert result.exit_code == 3, result.output  # no run matched
    assert result.stdout == expected
