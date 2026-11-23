"""Regression tests for the hardening in the credential/security audit."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from mlflow_pull import (
    FINISHED,
    PARENT_RUN_TAG,
    MlflowGateway,
    RemoteAddr,
    SshHost,
    app,
    download,
    download_experiments,
    resolve_experiments,
    resolve_runs,
    ssh_command,
    ssh_tunnel,
)
from mlflow_pull import localfs, paths
from mlflow_pull.localfs import (
    NotRegularFileError,
    SymlinkError,
    ensure_directory,
    prepare_destination,
    read_small_text,
    stat_file,
    write_atomically,
)
from mlflow_pull.plan import OnDiskFile
from mlflow_pull.ports import UNKNOWN_SIZE, InvalidSizeError, KnownSize, size_from_listing
from mlflow_pull.tree import Resolution, RunTarget
from mlflow_pull.paths import MAX_COMPONENT_BYTES, MAX_PATH_BYTES, RelPath, UnsafePathError, require_within, safe_component

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="needs real symlinks and FIFOs")

# --- S1: server-controlled names must not escape --out --------------------------------


@pytest.mark.parametrize("name", ["..", ".", "", " ", "...", "a/b", "a\\b", "x\x01y", "con", "NUL.txt"])
def test_safe_component_is_a_single_plain_path_component(name: str) -> None:
    result = safe_component(name)
    assert result not in {"", ".", ".."}
    assert "/" not in result and "\\" not in result
    assert not any(ord(c) < 32 for c in result)
    assert result.split(".")[0].lower() not in {"con", "nul"}


def test_safe_component_keeps_ordinary_names() -> None:
    assert safe_component("my-project v2") == "my-project v2"


def test_require_within_rejects_escape(tmp_path: Path) -> None:
    require_within(tmp_path, tmp_path / "a" / "b")
    with pytest.raises(UnsafePathError):
        require_within(tmp_path, tmp_path / ".." / "elsewhere")


def test_experiment_named_dotdot_stays_under_out(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    eid = client.create_experiment("..", artifact_location=(tmp_path / "art").as_uri())
    run_id = client.create_run(eid, run_name="r").info.run_id
    client.log_text(run_id, "x", "f.txt")
    client.set_terminated(run_id, "FINISHED")

    out = tmp_path / "work" / "out"
    download_experiments(MlflowGateway(client), [".."], out)

    # Unfixed, the experiment directory was ``out/..`` = ``work`` and the run landed beside ``out``.
    assert [p.name for p in (tmp_path / "work").iterdir()] == ["out"]
    assert [p.name for p in out.iterdir()] == ["_"]
    assert [p.name for p in out.rglob("f.txt")] == ["f.txt"]


def test_download_refuses_a_target_outside_root_before_writing(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    eid = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    run_id = client.create_run(eid, run_name="r").info.run_id
    client.set_terminated(run_id, "FINISHED")
    out = tmp_path / "out"
    good = resolve_experiments(MlflowGateway(client), ["e"], out, FINISHED)
    target = good.trees[0][0]
    hostile = Resolution(out, ((RunTarget(target.run, out / ".." / "escaped"),),), ())

    with pytest.raises(UnsafePathError):
        download(MlflowGateway(client), hostile)
    assert not out.exists() and not (tmp_path / "escaped").exists()


def test_failed_run_hides_its_finished_descendant(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    eid = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    failed = client.create_run(eid, run_name="failed").info.run_id
    client.set_terminated(failed, "FAILED")
    child = client.create_run(eid, tags={PARENT_RUN_TAG: failed}, run_name="child").info.run_id
    client.log_text(child, "x", "c.txt")
    client.set_terminated(child, "FINISHED")

    resolution = resolve_experiments(MlflowGateway(client), ["e"], tmp_path / "out", FINISHED)

    assert resolution.run_count == 0
    assert [s.status for s in resolution.skipped] == ["FAILED"]


def test_parent_tag_cycle_does_not_hang(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    eid = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    a = client.create_run(eid, run_name="a").info.run_id
    b = client.create_run(eid, tags={PARENT_RUN_TAG: a}, run_name="b").info.run_id
    client.set_tag(a, PARENT_RUN_TAG, b)  # a <-> b, no top-level run
    c = client.create_run(eid, tags={PARENT_RUN_TAG: a, "x": "1"}, run_name="c").info.run_id
    for run_id in (a, b, c):
        client.set_terminated(run_id, "FINISHED")
    gateway = MlflowGateway(client)

    # c is the only match; walking its ancestors enters the a <-> b cycle.
    matched = resolve_experiments(gateway, ["e"], tmp_path / "out", FINISHED, filter_string="tags.x = '1'")
    assert [t.run.info.run_id for tree in matched.trees for t in tree] == [c]
    # Naming a cycle member walks its subtree, which contains the cycle.
    named = resolve_runs(gateway, [a], tmp_path / "out", FINISHED)
    assert sorted(t.run.info.run_id for tree in named.trees for t in tree) == sorted([a, b, c])


# --- S3: ssh argument injection --------------------------------------------------------


@pytest.mark.parametrize("host", ["-oProxyCommand=touch pwned", "-J x", "", "a b", "host\n"])
def test_ssh_host_rejects_option_like_and_blank_values(host: str) -> None:
    with pytest.raises(ValueError):
        SshHost(host)


@pytest.mark.parametrize("host", ["myhost", "myhost.alias", "user@host.example.org", "10.0.0.1"])
def test_ssh_host_accepts_aliases_and_user_at_host(host: str) -> None:
    assert SshHost(host).name == host


@pytest.mark.parametrize("text", ["nohost", "host:", "host:0", "host:70000", "-x:5000", "a b:1", "host:5000;id"])
def test_remote_addr_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError):
        RemoteAddr.parse(text)


def test_remote_addr_round_trips() -> None:
    assert str(RemoteAddr.parse("127.0.0.1:5050")) == "127.0.0.1:5050"
    assert str(RemoteAddr.parse("[::1]:5050")) == "[::1]:5050"


def test_ssh_command_ends_options_before_host_and_ignores_control_master() -> None:
    cmd = ssh_command(SshHost("test.invalid"), RemoteAddr("localhost", 5000), 5055)

    assert cmd[-2:] == ["--", "test.invalid"]
    assert "ControlMaster=no" in cmd and "ControlPath=none" in cmd


def test_cli_rejects_option_like_ssh_host_without_running_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []
    monkeypatch.setattr("mlflow_pull.cli.ssh_tunnel", lambda *a, **k: called.append(a))
    result = CliRunner().invoke(app, ["run", "x", "--ssh=-oProxyCommand=touch pwned"])

    assert result.exit_code == 2
    assert not called


@pytest.mark.parametrize("when", ["99999999999d", "yesterday"])
def test_cli_rejects_unparseable_or_overflowing_time(when: str) -> None:
    assert CliRunner().invoke(app, ["experiment", "e", "--since", when]).exit_code == 2


# --- S4: a chatty ssh must not stall the tunnel ---------------------------------------

_FAKE_SSH_CHATTY = """
import socket, sys
sys.stderr.write("channel open failed\\n" * 20000)  # ~400 KB, far over a 64 KB pipe
sys.stderr.flush()
port = sys.argv[sys.argv.index("-L") + 1].split(":")[1]
srv = socket.create_server(("127.0.0.1", int(port)))
while True:
    srv.accept()[0].close()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="fake ssh is a shebang script")
def test_tunnel_survives_large_ssh_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ssh"
    script.write_text(f"#!{sys.executable}\n{_FAKE_SSH_CHATTY}")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    with ssh_tunnel(SshHost("test.invalid"), timeout=5) as uri:
        assert uri.startswith("http://127.0.0.1:")


# --- S6: no process-global environment writes ------------------------------------------


def test_ssh_run_uses_tunnel_uri_without_touching_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    eid = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    run_id = client.create_run(eid, run_name="r").info.run_id
    client.log_text(run_id, "x", "f.txt")
    client.set_terminated(run_id, "FINISHED")

    @contextmanager
    def fake_tunnel(*_args, **_kwargs):
        yield client.tracking_uri  # stands in for the http://127.0.0.1:<port> tunnel URI

    monkeypatch.setattr("mlflow_pull.cli.ssh_tunnel", fake_tunnel)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # must be ignored
    out = tmp_path / "out"

    result = CliRunner().invoke(app, ["run", run_id, "--ssh", "tunnelled", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert [p.name for p in out.rglob("f.txt")] == ["f.txt"]
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:1"


# --- plan/execute split: resolving reads only -------------------------------------------


def test_resolving_experiments_writes_nothing(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    eid = client.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    run_id = client.create_run(eid, run_name="r").info.run_id
    client.set_terminated(run_id, "FINISHED")
    out = tmp_path / "out"

    resolution = resolve_experiments(MlflowGateway(client), ["e"], out, FINISHED)

    assert [t.run.info.run_id for tree in resolution.trees for t in tree] == [run_id]
    assert not out.exists()


# --- RelPath: what a file system can hold ------------------------------------------------------------------

_LONGEST_PATH = ["a" * MAX_COMPONENT_BYTES] * 3 + ["a" * (MAX_COMPONENT_BYTES - 1), "a"]  # exactly MAX_PATH_BYTES


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("a\ud800b", "UTF-8"),
        ("\udc80", "UTF-8"),
        ("a" * (MAX_COMPONENT_BYTES + 1), "255 bytes"),
        ("\u00e9" * 128, "255 bytes"),  # 256 bytes, 128 characters
        ("d/" + "a" * 256, "255 bytes"),
        ("/".join([*_LONGEST_PATH, "b"]), f"{MAX_PATH_BYTES} bytes"),
    ],
)
def test_rel_path_rejects_names_no_file_system_can_hold(monkeypatch: pytest.MonkeyPatch, text: str, reason: str) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", False)
    with pytest.raises(UnsafePathError, match=reason):
        RelPath(text)


def test_rel_path_accepts_the_longest_representable_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", False)
    assert len("/".join(_LONGEST_PATH).encode()) == MAX_PATH_BYTES
    RelPath("/".join(_LONGEST_PATH))
    RelPath("a" * MAX_COMPONENT_BYTES)
    RelPath("\u00e9" * 127)


@pytest.mark.parametrize(
    "text",
    ["dir/a:b", "con", "x/NUL.txt", "a.", "a ", "dir/b .", "a<b", "a|b", "a?b", "a*b", 'a"b', "COM1"],
)
def test_on_windows_rel_path_rejects_names_windows_cannot_create(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", True)
    with pytest.raises(UnsafePathError, match="drive letter|Windows"):
        RelPath(text)
    monkeypatch.setattr(paths, "IS_WINDOWS", False)
    RelPath(text)  # POSIX takes all of them


@pytest.mark.parametrize("text", ["a.b", "dir/ok", "\u00e9.txt", "console", "a b"])
def test_on_windows_rel_path_still_accepts_ordinary_names(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", True)
    assert RelPath(text).value == text


def test_is_windows_is_one_named_constant() -> None:
    assert paths.IS_WINDOWS == (os.name == "nt")


# --- sizes at the boundary ----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, -(10**12), True, False, 1.5, "7"])
def test_a_known_size_is_a_non_negative_integer(bad: object) -> None:
    with pytest.raises(InvalidSizeError):
        KnownSize(bad)  # type: ignore[arg-type]


def test_listed_sizes_are_known_unknown_or_rejected() -> None:
    path = RelPath("a.bin")
    assert size_from_listing(0, path) == KnownSize(0)
    assert size_from_listing(12, path) == KnownSize(12)
    assert size_from_listing(None, path) is UNKNOWN_SIZE
    with pytest.raises(InvalidSizeError, match=r"a\.bin"):
        size_from_listing(-3, path)


# --- the local file system: no symlinks, no surprises --------------------------------------------------------


@POSIX_ONLY
def test_atomic_write_does_not_follow_a_planted_temp_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    (tmp_path / ".mlpull.json.tmp").symlink_to(victim)
    (tmp_path / "manifest.json.tmp").symlink_to(tmp_path / "nowhere" / "dangling")

    write_atomically(tmp_path / ".mlpull.json", "new")
    write_atomically(tmp_path / "manifest.json", "new")

    assert victim.read_text() == "precious"
    assert (tmp_path / ".mlpull.json").read_text() == "new" and (tmp_path / "manifest.json").read_text() == "new"
    assert not (tmp_path / "nowhere").exists()


def test_atomic_writes_replace_and_leave_nothing_behind(tmp_path: Path) -> None:
    target = tmp_path / "m.json"
    for text in ("one", "two", "three"):
        write_atomically(target, text)

    assert target.read_text() == "three" and [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_concurrent_writers_of_one_file_do_not_collide(tmp_path: Path) -> None:
    target = tmp_path / "m.json"
    texts = [f"writer {i} " * 500 for i in range(8)]
    barrier = threading.Barrier(len(texts))

    def write(text: str) -> None:
        barrier.wait()
        for _ in range(20):
            write_atomically(target, text)

    with ThreadPoolExecutor(len(texts)) as pool:
        list(pool.map(write, texts))  # raises if any writer raised

    assert target.read_text() in texts  # one whole version, never a mixture
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_atomic_write_syncs_the_file_and_on_posix_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    write_atomically(tmp_path / "m.json", "x")

    assert len(synced) == (1 if sys.platform == "win32" else 2)


def test_a_failed_atomic_write_removes_its_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "replace", refuse)

    with pytest.raises(OSError, match="disk on fire"):
        write_atomically(tmp_path / "m.json", "x")

    assert list(tmp_path.iterdir()) == []


def test_small_text_is_read_whole(tmp_path: Path) -> None:
    (tmp_path / "m").write_text("hello \u00e9", encoding="utf-8")
    assert read_small_text(tmp_path / "m") == "hello \u00e9"


def test_small_text_over_the_limit_is_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(localfs, "MAX_MANIFEST_BYTES", 100)
    (tmp_path / "ok").write_bytes(b"x" * 100)
    (tmp_path / "big").write_bytes(b"x" * 101)

    assert read_small_text(tmp_path / "ok") == "x" * 100
    assert read_small_text(tmp_path / "big") is None


def test_the_manifest_limit_is_16_mebibytes() -> None:
    assert localfs.MAX_MANIFEST_BYTES == 16 * 1024**2


def test_small_text_that_is_missing_or_not_utf8_is_nothing(tmp_path: Path) -> None:
    (tmp_path / "binary").write_bytes(b"\xff\xfe\x00")
    assert read_small_text(tmp_path / "binary") is None
    assert read_small_text(tmp_path / "missing") is None
    assert read_small_text(tmp_path) is None  # a directory


@POSIX_ONLY
def test_small_text_never_opens_a_fifo_or_follows_a_symlink(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "real").write_text("secret")
    (tmp_path / "link").symlink_to(tmp_path / "real")

    assert read_small_text(tmp_path / "fifo") is None  # returns at once: no writer needed
    assert read_small_text(tmp_path / "link") is None


def test_stat_file_describes_regular_files_only(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.txt").write_text("12345")
    found = stat_file(tmp_path, RelPath("d/f.txt"))
    info = (tmp_path / "d" / "f.txt").stat()

    assert found == OnDiskFile(5, info.st_mtime_ns)
    assert stat_file(tmp_path, RelPath("d")) is None  # a directory
    assert stat_file(tmp_path, RelPath("missing")) is None
    assert stat_file(tmp_path, RelPath("d/f.txt/deeper")) is None


@POSIX_ONLY
def test_stat_file_trusts_no_symlink_on_the_way(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "f.txt").write_text("12345")
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    (tmp_path / "file-link").symlink_to(tmp_path / "real" / "f.txt")
    os.mkfifo(tmp_path / "fifo")

    assert stat_file(tmp_path, RelPath("real/f.txt")) is not None
    assert stat_file(tmp_path, RelPath("linked/f.txt")) is None
    assert stat_file(tmp_path, RelPath("file-link")) is None
    assert stat_file(tmp_path, RelPath("fifo")) is None


def test_stat_file_survives_a_file_system_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "f").write_text("x")

    def refuse(*_: object, **__: object) -> None:
        raise PermissionError("no")

    monkeypatch.setattr(os, "lstat", refuse)

    assert stat_file(tmp_path, RelPath("f")) is None


def test_ensure_directory_creates_what_is_missing_below_the_root(tmp_path: Path) -> None:
    root = tmp_path / "does" / "not" / "exist"

    ensure_directory(root, root / "a" / "b")
    ensure_directory(root, root / "a" / "b")  # again: fine

    assert (root / "a" / "b").is_dir()


def test_ensure_directory_refuses_to_leave_the_root(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        ensure_directory(tmp_path / "root", tmp_path / "elsewhere")


def test_ensure_directory_refuses_a_file_in_the_way(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("x")
    with pytest.raises(UnsafePathError, match="not a directory"):
        ensure_directory(tmp_path, tmp_path / "f" / "g")


@POSIX_ONLY
def test_ensure_directory_refuses_every_symlinked_component(tmp_path: Path) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "root" / "a").mkdir(parents=True)
    (tmp_path / "root" / "a" / "b").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

    with pytest.raises(SymlinkError, match="b"):
        ensure_directory(tmp_path / "root", tmp_path / "root" / "a" / "b" / "c")

    assert list((tmp_path / "elsewhere").iterdir()) == []


@POSIX_ONLY
def test_the_root_itself_may_be_a_symlink_that_is_the_users_choice(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "root").symlink_to(tmp_path / "real", target_is_directory=True)

    ensure_directory(tmp_path / "root", tmp_path / "root" / "a")

    assert (tmp_path / "real" / "a").is_dir()


def test_a_destination_may_be_absent_or_an_existing_regular_file(tmp_path: Path) -> None:
    (tmp_path / "there").write_text("old")

    prepare_destination(tmp_path, tmp_path / "there")
    prepare_destination(tmp_path, tmp_path / "new" / "deep" / "file")

    assert (tmp_path / "new" / "deep").is_dir() and not (tmp_path / "new" / "deep" / "file").exists()


@POSIX_ONLY
def test_a_destination_that_is_a_symlink_or_a_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "victim").write_text("precious")
    (tmp_path / "link").symlink_to(tmp_path / "victim")
    (tmp_path / "dangling").symlink_to(tmp_path / "nothing")
    (tmp_path / "dir").mkdir()

    for name in ("link", "dangling"):
        with pytest.raises(SymlinkError, match=name):
            prepare_destination(tmp_path, tmp_path / name)
    with pytest.raises(NotRegularFileError, match="dir"):
        prepare_destination(tmp_path, tmp_path / "dir")
