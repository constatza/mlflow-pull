from __future__ import annotations

import os
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from mlflow_pull import (
    PARENT_RUN_TAG,
    RunStatusError,
    allowed_statuses,
    app,
    build_filter,
    download_experiments,
    download_runs,
    parse_when,
    ssh_command,
    ssh_tunnel,
)

EVERY_STATUS = allowed_statuses(unfinished=True, failed=True)


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")


@pytest.fixture
def tree(client: MlflowClient, tmp_path: Path) -> dict[str, str]:
    """exp: parent -> (child_a[report] -> grandchild[report], child_b FAILED),
    other (param lr=0.1), live RUNNING.  exp-two: one run."""

    def make(
        experiment_id: str,
        name: str,
        parent: str | None,
        files: dict[str, str],
        status: str | None = "FINISHED",
        tags: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> str:
        all_tags = dict(tags or {}) | ({} if parent is None else {PARENT_RUN_TAG: parent})
        run_id = client.create_run(experiment_id, tags=all_tags, run_name=name).info.run_id
        for artifact_path, text in files.items():
            client.log_text(run_id, text, artifact_path)
        for key, value in (params or {}).items():
            client.log_param(run_id, key, value)
        if status is not None:  # None leaves the run RUNNING, as create_run makes it
            client.set_terminated(run_id, status)
        return run_id

    exp = client.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    exp_two = client.create_experiment("exp-two", artifact_location=(tmp_path / "art2").as_uri())
    report = {"phase": "report"}
    parent = make(exp, "parent", None, {"top.txt": "p", "dir/nested/deep.txt": "d"})
    child_a = make(exp, "child_a", parent, {"a.txt": "a"}, tags=report)
    make(exp, "grandchild", child_a, {"g.txt": "g"}, tags=report)
    make(exp, "child_b", parent, {"b.txt": "b"}, status="FAILED")
    other = make(exp, "other", None, {"o.txt": "o"}, params={"lr": "0.1"})
    live = make(exp, "live", None, {"l.txt": "l"}, status=None)
    make(exp_two, "solo", None, {"s.txt": "s"})
    return {"parent": parent, "child_a": child_a, "other": other, "live": live}


def _files(root: Path) -> list[str]:
    return sorted(p.name for p in root.rglob("*") if p.is_file())


# --- download_runs -------------------------------------------------------------


def test_run_tree_keeps_layout_and_skips_failed_child(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    count = download_runs(client, [tree["parent"]], out)

    parent_dir = out / f"parent-{tree['parent'][:8]}"
    assert count == 3
    assert (parent_dir / "dir" / "nested" / "deep.txt").is_file()
    assert (parent_dir / f"child_a-{tree['child_a'][:8]}" / "a.txt").is_file()
    assert _files(out) == ["a.txt", "deep.txt", "g.txt", "top.txt"]


def test_run_tree_with_every_status_includes_failed_child(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    assert download_runs(client, [tree["parent"]], tmp_path / "out", EVERY_STATUS) == 4
    assert "b.txt" in _files(tmp_path / "out")


def test_unfinished_named_run_is_refused_before_any_download(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    with pytest.raises(RunStatusError, match="RUNNING"):
        download_runs(client, [tree["other"], tree["live"]], out)
    assert not out.exists()


def test_several_runs_in_parallel(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert download_runs(client, [tree["parent"], tree["other"]], out, jobs=2) == 4
    assert "o.txt" in _files(out)


# --- download_experiments ----------------------------------------------------------


def test_experiment_skips_unfinished_and_failed_runs(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert download_experiments(client, ["exp"], out) == 4
    assert _files(out) == ["a.txt", "deep.txt", "g.txt", "o.txt", "top.txt"]


def test_experiment_with_every_status(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    assert download_experiments(client, ["exp"], tmp_path / "out", EVERY_STATUS) == 6


def test_tag_filter_picks_nested_match_once_with_its_subtree(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    count = download_experiments(
        client, ["exp"], out, filter_string=build_filter(tags=[("phase", "report")])
    )

    child_dir = out / "exp" / f"child_a-{tree['child_a'][:8]}"
    assert count == 2  # grandchild matches too but comes inside child_a's subtree
    assert (child_dir / "a.txt").is_file()
    assert _files(out) == ["a.txt", "g.txt"]


def test_param_filter(client: MlflowClient, tree: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "out"
    filter_string = build_filter(params=[("lr", "0.1")])
    assert download_experiments(client, ["exp"], out, filter_string=filter_string) == 1
    assert _files(out) == ["o.txt"]


def test_since_in_the_future_selects_nothing(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    filter_string = build_filter(since=tomorrow)
    assert download_experiments(client, ["exp"], tmp_path / "out", filter_string=filter_string) == 0


def test_experiment_glob_matches_several(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert download_experiments(client, ["exp*"], out) == 5
    assert {p.name for p in out.iterdir()} == {"exp", "exp-two"}


def test_experiment_glob_without_match_is_an_error(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no experiment"):
        download_experiments(client, ["nothing*"], tmp_path / "out")


# --- filter building -------------------------------------------------------------


def test_build_filter_joins_clauses() -> None:
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = build_filter(
        since=since, tags=[("mlflow.user", "me")], params=[("p", "it's")], raw="metrics.x < 1"
    )
    assert result == (
        f"attributes.start_time >= {int(since.timestamp() * 1000)}"
        " and tags.`mlflow.user` = 'me'"
        ' and params.`p` = "it\'s"'
        " and metrics.x < 1"
    )
    assert build_filter() is None


def test_build_filter_rejects_value_with_both_quotes() -> None:
    with pytest.raises(ValueError, match="both quote"):
        build_filter(tags=[("k", "a'b\"c")])


def test_parse_when_relative_and_iso() -> None:
    now = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
    assert parse_when("2d", now) == now - timedelta(days=2)
    assert parse_when("30m", now) == now - timedelta(minutes=30)
    assert parse_when("2026-11-18T00:00+00:00", now) == datetime(2026, 11, 18, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        parse_when("yesterday", now)


# --- CLI -----------------------------------------------------------------------------


@pytest.fixture
def cli(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", client.tracking_uri)
    monkeypatch.delenv("MLPULL_SSH", raising=False)
    monkeypatch.delenv("MLPULL_OUT", raising=False)
    return lambda *args: CliRunner().invoke(app, list(args))


def test_cli_run_with_out_option(cli, tree: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "cli-out"
    result = cli("run", tree["other"], "-o", str(out))

    assert result.exit_code == 0, result.output
    assert _files(out) == ["o.txt"]


def test_cli_refuses_unfinished_run_and_all_flag_overrides(
    cli, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "cli-out"
    refused = cli("run", tree["live"], "-o", str(out))
    assert refused.exit_code == 1
    assert "override with -u, -f or -a" in refused.output

    assert cli("run", tree["live"], "-a", "-o", str(out)).exit_code == 0
    assert _files(out) == ["l.txt"]


def test_cli_experiment_tag_filter(cli, tree: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "cli-out"
    result = cli("experiment", "exp", "--tag", "phase=report", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert _files(out) == ["a.txt", "g.txt"]


@pytest.mark.parametrize(
    "args",
    [
        ("run", "x", "-z"),  # compression without a tunnel
        ("experiment", "exp", "--tag", "no-equals-sign"),
        ("experiment", "exp", "--since", "yesterday"),
    ],
)
def test_cli_usage_errors(cli, args: tuple[str, ...]) -> None:
    assert cli(*args).exit_code == 2


# --- SSH tunnel (fake ssh only: no test may reach a real host) -------------------------

# Stand-in for ssh: listens on the -L local port until terminated.
_FAKE_SSH_LISTEN = """
import socket, sys
port = sys.argv[sys.argv.index("-L") + 1].split(":")[1]  # bind:port:host:hostport
srv = socket.create_server(("127.0.0.1", int(port)))
while True:
    srv.accept()[0].close()
"""
_FAKE_SSH_FAIL = """
import sys
sys.stderr.write("Permission denied (publickey).\\n")
sys.exit(255)
"""


@pytest.fixture
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``ssh`` with the given Python body first on PATH."""

    def install(body: str) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        script = bin_dir / "ssh"
        script.write_text(f"#!{sys.executable}\n{body}")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    return install


def test_ssh_command_forwards_local_port() -> None:
    cmd = ssh_command("test.invalid", "localhost:5000", 5055)

    assert cmd[0] == "ssh" and cmd[-1] == "test.invalid"
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:5055:localhost:5000"
    assert "ExitOnForwardFailure=yes" in cmd and "BatchMode=yes" in cmd
    assert "-C" not in cmd
    assert "-C" in ssh_command("test.invalid", "localhost:5000", 5055, compress=True)


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
def test_tunnel_opens_and_closes(fake_ssh) -> None:
    fake_ssh(_FAKE_SSH_LISTEN)

    with ssh_tunnel("test.invalid") as uri:
        host, port = uri.removeprefix("http://").split(":")
        socket.create_connection((host, int(port)), 1).close()

    with pytest.raises(OSError):
        socket.create_connection((host, int(port)), 1)


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
def test_tunnel_reports_ssh_failure(fake_ssh) -> None:
    fake_ssh(_FAKE_SSH_FAIL)

    with pytest.raises(SystemExit, match="Permission denied"):
        with ssh_tunnel("test.invalid"):
            pass
