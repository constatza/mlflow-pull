"""Structured output and exit codes: ``--format jsonl`` on ``run`` and ``experiment``, distinct exit codes."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import Result
from mlflow.tracking import MlflowClient

from mlflow_pull import PARENT_RUN_TAG, MlflowGateway
from mlflow_pull.exitcodes import ExitCode
from mlflow_pull.plan import MANIFEST_NAME

Cli = Callable[..., Result]

FIELDS = ["run_id", "name", "status", "dir", "outcome", "files", "bytes", "unsupported"]


@dataclass(frozen=True, slots=True)
class Store:
    client: MlflowClient
    parent: str  # FINISHED, one 3-byte artifact
    child: str  # FINISHED, nested under parent, one 7-byte artifact
    failed_child: str  # FAILED, nested under parent
    failed_only: str  # the experiment "broken" holds this one FAILED top-level run


def _strict_lines(text: str) -> list[dict[str, object]]:
    """Every stdout line must be one strict JSON object: NaN, blank lines and stray notes fail."""

    def refuse(constant: str) -> None:
        raise AssertionError(f"not strict JSON: {constant}")

    lines = text.splitlines()
    assert all(lines), "blank line in the JSON lines"
    return [json.loads(line, parse_constant=refuse) for line in lines]


def _make_run(
    client: MlflowClient, experiment_id: str, name: str, text: str, status: str, parent: str | None = None
) -> str:
    tags = {} if parent is None else {PARENT_RUN_TAG: parent}
    run_id = client.create_run(experiment_id, tags=tags, run_name=name).info.run_id
    client.log_text(run_id, text, f"{name}.txt")
    client.set_terminated(run_id, status)
    return run_id


@pytest.fixture
def store(client: MlflowClient, tmp_path: Path) -> Store:
    exp = client.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    broken = client.create_experiment("broken", artifact_location=(tmp_path / "art-broken").as_uri())
    parent = _make_run(client, exp, "parent", "abc", "FINISHED")
    child = _make_run(client, exp, "child", "x" * 7, "FINISHED", parent)
    failed_child = _make_run(client, exp, "bad", "zz", "FAILED", parent)
    failed_only = _make_run(client, broken, "dead", "zz", "FAILED")
    return Store(client, parent, child, failed_child, failed_only)


def _out(tmp_path: Path) -> Path:
    return (tmp_path / "out").resolve()


def _dir(out: Path, name: str, run_id: str) -> str:
    return str(out / f"{name}-{run_id[:8]}")


# --- the records ----------------------------------------------------------------------------


def test_jsonl_one_record_per_run_with_notes_on_stderr(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    result = cli("run", store.parent, "--format", "jsonl", "-o", str(out))
    parent_dir = _dir(out, "parent", store.parent)

    assert result.exit_code == ExitCode.OK, result.stderr
    records = _strict_lines(result.stdout)
    assert [list(r) for r in records] == [FIELDS] * 3
    assert records == [
        {"run_id": store.parent, "name": "parent", "status": "FINISHED", "dir": parent_dir,
         "outcome": "downloaded", "files": 1, "bytes": 3, "unsupported": 0},
        {"run_id": store.child, "name": "child", "status": "FINISHED", "dir": _dir(Path(parent_dir), "child", store.child),
         "outcome": "downloaded", "files": 1, "bytes": 7, "unsupported": 0},
        {"run_id": store.failed_child, "name": "bad", "status": "FAILED",
         "dir": _dir(Path(parent_dir), "bad", store.failed_child),
         "outcome": "skipped_status", "files": 0, "bytes": 0, "unsupported": 0},
    ]
    assert f"skipped bad-{store.failed_child[:8]}: FAILED" in result.stderr
    assert "Downloaded 2 run(s), 2 file(s), 10 B" in result.stderr
    assert "Downloaded" not in result.stdout and "skipped" not in result.stdout.replace("skipped_status", "")


def test_jsonl_second_pull_reports_up_to_date_and_still_exits_0(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    assert cli("run", store.parent, "--format", "jsonl", "-o", str(out)).exit_code == ExitCode.OK

    again = cli("run", store.parent, "--format", "jsonl", "-o", str(out))

    assert again.exit_code == ExitCode.OK, again.stderr
    assert [(r["name"], r["outcome"], r["files"], r["bytes"]) for r in _strict_lines(again.stdout)] == [
        ("parent", "up_to_date", 1, 3), ("child", "up_to_date", 1, 7), ("bad", "skipped_status", 0, 0),
    ]
    assert "2 run(s) already up to date" in again.stderr


def test_jsonl_dry_run_lists_would_download_and_writes_nothing(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    result = cli("run", store.parent, "--dry-run", "--format", "jsonl", "-o", str(out))

    assert result.exit_code == ExitCode.OK, result.stderr
    assert [(r["name"], r["outcome"], r["files"], r["bytes"]) for r in _strict_lines(result.stdout)] == [
        ("parent", "would_download", 1, 3), ("child", "would_download", 1, 7), ("bad", "skipped_status", 0, 0),
    ]
    assert "Would fetch 2 file(s), 10 B" in result.stderr
    assert not out.exists()


def test_jsonl_dry_run_after_a_pull_reports_up_to_date(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    assert cli("run", store.parent, "-o", str(out)).exit_code == ExitCode.OK

    result = cli("run", store.parent, "--dry-run", "--format", "jsonl", "-o", str(out))

    assert [r["outcome"] for r in _strict_lines(result.stdout)] == ["up_to_date", "up_to_date", "skipped_status"]


def test_jsonl_dry_run_still_reports_a_size_refusal(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    result = cli("run", store.parent, "--dry-run", "--max-size", "5", "--format", "jsonl", "-o", str(out))

    assert result.exit_code == ExitCode.REFUSED
    assert "refused (--max-size)" in result.stderr
    assert [r["outcome"] for r in _strict_lines(result.stdout)] == ["would_download"] * 2 + ["skipped_status"]
    assert not out.exists()


def test_jsonl_experiment_uses_the_same_records(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    result = cli("experiment", "exp", "--format", "jsonl", "-o", str(out))

    assert result.exit_code == ExitCode.OK, result.stderr
    records = _strict_lines(result.stdout)
    assert [r["name"] for r in records] == ["parent", "child", "bad"]
    assert records[0]["dir"] == str(out / "exp" / f"parent-{store.parent[:8]}")


def test_text_format_is_unchanged_and_keeps_notes_apart(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    result = cli("run", store.parent, "-o", str(out))

    assert result.exit_code == ExitCode.OK
    assert result.stdout.strip() == f"Downloaded 2 run(s), 2 file(s), 10 B to {out}"
    assert result.stderr.strip() == f"skipped bad-{store.failed_child[:8]}: FAILED"


# --- exit codes ---------------------------------------------------------------------------------


def test_exit_codes_are_the_documented_numbers() -> None:
    assert [int(c) for c in (ExitCode.OK, ExitCode.REFUSED, ExitCode.USAGE, ExitCode.NOTHING_MATCHED)] == [0, 1, 2, 3]


@pytest.mark.parametrize("fmt", ["text", "jsonl"])
def test_a_filter_matching_nothing_exits_3(store: Store, cli: Cli, tmp_path: Path, fmt: str) -> None:
    out = _out(tmp_path)
    result = cli("experiment", "exp", "--tag", "phase=nope", "--format", fmt, "-o", str(out))

    assert result.exit_code == ExitCode.NOTHING_MATCHED
    assert "nothing matched" in result.stderr
    assert result.stdout == ""
    assert not out.exists()


def test_every_candidate_skipped_by_status_exits_3_and_names_them(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    result = cli("experiment", "broken", "--format", "jsonl", "-o", str(out))

    assert result.exit_code == ExitCode.NOTHING_MATCHED
    [record] = _strict_lines(result.stdout)
    assert (record["run_id"], record["outcome"], record["status"]) == (store.failed_only, "skipped_status", "FAILED")
    assert not out.exists()


def test_an_experiment_glob_without_match_exits_3(store: Store, cli: Cli, tmp_path: Path) -> None:
    result = cli("experiment", "nothing*", "-o", str(_out(tmp_path)))

    assert result.exit_code == ExitCode.NOTHING_MATCHED
    assert "no experiment matches nothing*" in result.stderr


def test_an_unknown_experiment_name_exits_1(store: Store, cli: Cli, tmp_path: Path) -> None:
    result = cli("experiment", "no-such-experiment", "-o", str(_out(tmp_path)))

    assert result.exit_code == ExitCode.REFUSED


@pytest.mark.parametrize("fmt", ["text", "jsonl"])
def test_a_refused_status_exits_1_and_writes_no_record(store: Store, cli: Cli, tmp_path: Path, fmt: str) -> None:
    out = _out(tmp_path)
    result = cli("run", store.failed_child, "--format", fmt, "-o", str(out))

    assert result.exit_code == ExitCode.REFUSED
    assert "refused (override with -u, -f or -a)" in result.stderr
    assert result.stdout == "" and not out.exists()


def test_a_size_refusal_exits_1(store: Store, cli: Cli, tmp_path: Path) -> None:
    result = cli("run", store.parent, "--max-size", "5", "--format", "jsonl", "-o", str(_out(tmp_path)))

    assert result.exit_code == ExitCode.REFUSED
    assert result.stdout == "" and "refused (--max-size)" in result.stderr


def test_a_reserved_artifact_name_exits_1(store: Store, cli: Cli, tmp_path: Path) -> None:
    store.client.log_text(store.child, "mine", MANIFEST_NAME)
    result = cli("run", store.child, "-o", str(_out(tmp_path)))

    assert result.exit_code == ExitCode.REFUSED
    assert "collides" in result.stderr


def test_a_pull_with_everything_up_to_date_exits_0(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = _out(tmp_path)
    cli("run", store.child, "-o", str(out))

    assert cli("run", store.child, "-o", str(out)).exit_code == ExitCode.OK


@pytest.mark.parametrize(
    ("args", "code", "stdout"),
    [
        (("--format", "jsonl", "-e", "exp", "--tag", "phase=nope"), ExitCode.NOTHING_MATCHED, ""),
        (("--format", "json", "-e", "exp", "--tag", "phase=nope"), ExitCode.NOTHING_MATCHED, "[]\n"),
        (("--format", "jsonl", "-e", "exp"), ExitCode.OK, None),
        (("--format", "jsonl", "-e", "no-such-experiment"), ExitCode.REFUSED, ""),
    ],
)
def test_runs_exit_codes(store: Store, cli: Cli, args: tuple[str, ...], code: ExitCode, stdout: str | None) -> None:
    result = cli("runs", *args)

    assert result.exit_code == code
    if stdout is not None:
        assert result.stdout == stdout


def test_history_with_text_format_is_a_usage_error(store: Store, cli: Cli) -> None:
    result = cli("runs", "-e", "exp", "--metric", "loss", "--history", "3", "--format", "text")

    assert result.exit_code == ExitCode.USAGE
    assert "only json and jsonl output carry history" in result.stderr


def test_run_and_experiment_reject_an_unknown_format(store: Store, cli: Cli) -> None:
    assert cli("run", store.parent, "--format", "json").exit_code == ExitCode.USAGE
    assert cli("experiment", "exp", "--format", "md-table").exit_code == ExitCode.USAGE


# --- experiment search paging -----------------------------------------------------------------------


class SmallPageClient(MlflowClient):
    """A real client that serves one experiment per page, so the gateway has to follow page tokens."""

    pages_served = 0

    def search_experiments(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        type(self).pages_served += 1
        return super().search_experiments(*args, **{**kwargs, "max_results": 1})


def test_search_experiments_follows_page_tokens(store: Store) -> None:
    SmallPageClient.pages_served = 0
    paged = MlflowGateway(SmallPageClient(tracking_uri=store.client.tracking_uri))

    found = paged.search_experiments("")

    assert sorted(e.name for e in found) == ["Default", "broken", "exp"]
    assert SmallPageClient.pages_served == 3
