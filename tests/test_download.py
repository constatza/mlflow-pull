from __future__ import annotations

import os
import queue
import signal
import socket
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from mlflow_pull.plan import MANIFEST_NAME
from mlflow_pull import (
    PARENT_RUN_TAG,
    RemoteAddr,
    RunGateway,
    RunStatusError,
    SshHost,
    allowed_statuses,
    build_filter,
    download_experiments,
    download_runs,
    parse_when,
    ssh_command,
    ssh_tunnel,
)

EVERY_STATUS = allowed_statuses(unfinished=True, failed=True)


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
    return sorted(p.name for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)


# --- download_runs -------------------------------------------------------------


def test_run_tree_keeps_layout_and_skips_failed_child(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    count = download_runs(gateway, [tree["parent"]], out)

    parent_dir = out / f"parent-{tree['parent'][:8]}"
    assert count == 3
    assert (parent_dir / "dir" / "nested" / "deep.txt").is_file()
    assert (parent_dir / f"child_a-{tree['child_a'][:8]}" / "a.txt").is_file()
    assert _files(out) == ["a.txt", "deep.txt", "g.txt", "top.txt"]


def test_run_tree_with_every_status_includes_failed_child(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    assert download_runs(gateway, [tree["parent"]], tmp_path / "out", EVERY_STATUS) == 4
    assert "b.txt" in _files(tmp_path / "out")


def test_unfinished_named_run_is_refused_before_any_download(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    with pytest.raises(RunStatusError, match="RUNNING"):
        download_runs(gateway, [tree["other"], tree["live"]], out)
    assert not out.exists()


def test_several_runs_in_parallel(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert download_runs(gateway, [tree["parent"], tree["other"]], out, jobs=2) == 4
    assert "o.txt" in _files(out)


# --- download_experiments ----------------------------------------------------------


def test_experiment_skips_unfinished_and_failed_runs(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert download_experiments(gateway, ["exp"], out) == 4
    assert _files(out) == ["a.txt", "deep.txt", "g.txt", "o.txt", "top.txt"]


def test_experiment_with_every_status(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    assert download_experiments(gateway, ["exp"], tmp_path / "out", EVERY_STATUS) == 6


def test_tag_filter_picks_nested_match_once_with_its_subtree(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    count = download_experiments(
        gateway, ["exp"], out, filter_string=build_filter(tags=[("phase", "report")])
    )

    child_dir = out / "exp" / f"child_a-{tree['child_a'][:8]}"
    assert count == 2  # grandchild matches too but comes inside child_a's subtree
    assert (child_dir / "a.txt").is_file()
    assert _files(out) == ["a.txt", "g.txt"]


def test_param_filter(gateway: RunGateway, tree: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "out"
    filter_string = build_filter(params=[("lr", "0.1")])
    assert download_experiments(gateway, ["exp"], out, filter_string=filter_string) == 1
    assert _files(out) == ["o.txt"]


def test_since_in_the_future_selects_nothing(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    filter_string = build_filter(since=tomorrow)
    assert download_experiments(gateway, ["exp"], tmp_path / "out", filter_string=filter_string) == 0


def test_experiment_glob_matches_several(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert download_experiments(gateway, ["exp*"], out) == 5
    assert {p.name for p in out.iterdir()} == {"exp", "exp-two"}


def test_experiment_glob_without_match_is_an_error(
    gateway: RunGateway, tree: dict[str, str], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no experiment"):
        download_experiments(gateway, ["nothing*"], tmp_path / "out")


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
_FAKE_SSH_RECORD_PID = """
import os, socket, sys
with open(os.environ["FAKE_SSH_PID_FILE"], "w") as pid_file:
    pid_file.write(str(os.getpid()))
port = sys.argv[sys.argv.index("-L") + 1].split(":")[1]
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
    cmd = ssh_command(SshHost("test.invalid"), RemoteAddr("localhost", 5000), 5055)

    assert cmd[0] == "ssh" and cmd[-1] == "test.invalid" and cmd[-2] == "--"
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:5055:localhost:5000"
    assert "ExitOnForwardFailure=yes" in cmd and "BatchMode=yes" in cmd
    assert "-C" not in cmd
    assert "-C" in ssh_command(SshHost("test.invalid"), RemoteAddr("localhost", 5000), 5055, compress=True)


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
def test_tunnel_opens_and_closes(fake_ssh) -> None:
    fake_ssh(_FAKE_SSH_LISTEN)

    with ssh_tunnel(SshHost("test.invalid")) as uri:
        host, port = uri.removeprefix("http://").split(":")
        socket.create_connection((host, int(port)), 1).close()

    with pytest.raises(OSError):
        socket.create_connection((host, int(port)), 1)


_FAKE_SSH_HOSTILE_BANNER = """
import sys, time
sys.stderr.write("welcome\\x1b[2J\\u202e\\nPermission denied (publickey).\\n")
sys.stderr.flush()
if {stall}:
    time.sleep(60)  # alive, never listening: the tunnel times out
sys.exit(255)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
@pytest.mark.parametrize("stall", [False, True])
def test_a_remote_banner_in_a_tunnel_failure_is_escaped_and_on_one_line(fake_ssh, stall: bool) -> None:
    fake_ssh(_FAKE_SSH_HOSTILE_BANNER.format(stall=stall))

    with pytest.raises(SystemExit) as caught, ssh_tunnel(SshHost("test.invalid"), timeout=2.0):
        pass

    message = str(caught.value)
    assert "\x1b" not in message and "\u202e" not in message and "\n" not in message, repr(message)
    assert "\\x1b[2J" in message and "\\u202e" in message and "welcome" in message and "Permission denied" in message


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
def test_tunnel_reports_ssh_failure(fake_ssh) -> None:
    fake_ssh(_FAKE_SSH_FAIL)

    with pytest.raises(SystemExit, match="Permission denied"):
        with ssh_tunnel(SshHost("test.invalid")):
            pass


TUNNEL_HOLDER = """
import sys
from mlflow_pull import SshHost, ssh_tunnel
with ssh_tunnel(SshHost("test.invalid"), timeout=10) as uri:
    print("ready", flush=True)
    sys.stdin.read()
"""
SIGTERM_EXIT_CODE = 128 + signal.SIGTERM
WAIT_S = 60.0  # a cold import of mlflow in the holder's subprocess alone can take tens of seconds


def _await_ready(holder: subprocess.Popen[str], timeout: float = WAIT_S) -> None:
    """Fail, with the holder's stderr, unless it prints ``ready`` within ``timeout``.

    ``readline`` has no timeout and would hang the suite if the holder stalled, so it runs on a thread.
    """
    first_line: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: first_line.put(holder.stdout.readline()), daemon=True).start()
    try:
        line = first_line.get(timeout=timeout).strip()
    except queue.Empty:
        line = f"nothing within {timeout}s"
    if line != "ready":
        holder.kill()  # also ends the reader thread: the pipe closes
        holder.wait()
        pytest.fail(f"tunnel holder is not ready ({line!r}); stderr: {holder.stderr.read()!r}")


def _holder(code: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_a_holder_that_dies_before_ready_fails_fast_with_its_stderr() -> None:
    holder = _holder("import sys; sys.exit('no tunnel for you')")

    with pytest.raises(pytest.fail.Exception, match="no tunnel for you"):
        _await_ready(holder)


def test_a_holder_that_never_says_ready_fails_at_the_deadline() -> None:
    holder = _holder("import time; time.sleep(60)")

    with pytest.raises(pytest.fail.Exception, match="nothing within 0.2s"):
        _await_ready(holder, timeout=0.2)
    assert holder.returncode is not None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM and a shebang fake ssh are POSIX")
def test_sigterm_closes_the_tunnel_instead_of_orphaning_ssh(
    fake_ssh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "ssh.pid"
    monkeypatch.setenv("FAKE_SSH_PID_FILE", str(pid_file))
    fake_ssh(_FAKE_SSH_RECORD_PID)
    holder = _holder(TUNNEL_HOLDER)
    ssh_pid = None
    try:
        _await_ready(holder)
        ssh_pid = int(pid_file.read_text())
        holder.send_signal(signal.SIGTERM)

        assert holder.wait(WAIT_S) == SIGTERM_EXIT_CODE
        assert not _alive(ssh_pid)
    finally:
        holder.kill()
        holder.wait()
        if ssh_pid is not None and _alive(ssh_pid):
            os.kill(ssh_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM and a shebang fake ssh are POSIX")
def test_the_sigterm_handler_lives_only_while_the_tunnel_is_open(fake_ssh) -> None:
    fake_ssh(_FAKE_SSH_LISTEN)
    before = signal.getsignal(signal.SIGTERM)

    with ssh_tunnel(SshHost("test.invalid")):
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
def test_a_tunnel_opened_off_the_main_thread_leaves_signals_alone(fake_ssh) -> None:
    fake_ssh(_FAKE_SSH_LISTEN)
    before = signal.getsignal(signal.SIGTERM)
    seen: list[object] = []

    def open_tunnel() -> None:
        with ssh_tunnel(SshHost("test.invalid")):
            seen.append(signal.getsignal(signal.SIGTERM))

    worker = threading.Thread(target=open_tunnel)
    worker.start()
    worker.join(WAIT_S)

    assert seen == [before]


def test_a_missing_ssh_binary_fails_with_one_line(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assert_fails_cleanly
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    assert "ssh" in assert_fails_cleanly(cli("run", "x", "--ssh", "test.invalid"), 1)


def test_a_failing_ssh_fails_with_one_line(cli, fake_ssh, assert_fails_cleanly) -> None:
    fake_ssh(_FAKE_SSH_FAIL)

    assert "Permission denied" in assert_fails_cleanly(cli("run", "x", "--ssh", "test.invalid"), 1)


def test_no_free_local_port_fails_with_one_line(cli, monkeypatch: pytest.MonkeyPatch, assert_fails_cleanly) -> None:
    def refuse() -> int:
        raise OSError("no free port")

    monkeypatch.setattr("mlflow_pull.tunnel._free_port", refuse)

    assert "no free port" in assert_fails_cleanly(cli("run", "x", "--ssh", "test.invalid"), 1)


def test_parse_when_accepts_a_trailing_z() -> None:
    now = datetime(2026, 11, 23, tzinfo=timezone.utc)

    assert parse_when("2026-11-23T10:00:00Z", now) == datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
