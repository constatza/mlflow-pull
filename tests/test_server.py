"""Read and download paths against a real ``mlflow server`` with basic auth.

The sqlite-store tests bypass HTTP; these cover ``HttpStore``, search paging,
proxied ``mlflow-artifacts:`` downloads and credential handling.
Run with ``pytest -m slow`` (deselected by default).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from mlflow_pull import PARENT_RUN_TAG, MlflowGateway, app, download_experiments, download_runs
from mlflow_pull.plan import MANIFEST_NAME

pytestmark = pytest.mark.slow

ADMIN_USER = "admin"
ADMIN_PASSWORD = "test-only-admin-password"
STARTUP_TIMEOUT_S = 60
BOGUS_URI = "http://127.0.0.1:1"


@dataclass(slots=True, frozen=True)
class Server:
    uri: str


@dataclass(slots=True, frozen=True)
class Seeded:
    """Runs created through the server: parent -> child -> grandchild, each with one file."""

    experiment: str
    parent: str
    child: str
    grandchild: str


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_healthy(uri: str, proc: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{uri}/health", timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    proc.terminate()
    pytest.fail(f"mlflow server did not become healthy:\n{log.read_text()}")


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Server]:
    pytest.importorskip("flask_wtf", reason="basic-auth needs mlflow[auth]")
    root = tmp_path_factory.mktemp("server")
    auth_ini = root / "basic_auth.ini"
    auth_ini.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = sqlite:///{root / 'auth.db'}\n"
        f"admin_username = {ADMIN_USER}\n"
        f"admin_password = {ADMIN_PASSWORD}\n"
    )
    auth_ini.chmod(0o600)
    port = _free_port()
    uri = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "MLFLOW_AUTH_CONFIG_PATH": str(auth_ini),
        "MLFLOW_FLASK_SERVER_SECRET_KEY": "test-only-csrf-secret",  # basic-auth refuses to start without it
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    log = root / "server.log"
    with log.open("wb") as sink:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "mlflow", "server",
                "--app-name", "basic-auth",
                "--host", "127.0.0.1", "--port", str(port),
                "--backend-store-uri", f"sqlite:///{root / 'mlflow.db'}",
                "--artifacts-destination", str(root / "art"),
                "--serve-artifacts",
            ],
            env=env, stdout=sink, stderr=subprocess.STDOUT,
        )
        try:
            _wait_healthy(uri, proc, log)
            yield Server(uri)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credentials only via the environment, plus a bogus ambient URI the code must not use."""
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", ADMIN_USER)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", ADMIN_PASSWORD)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", BOGUS_URI)


@pytest.fixture(scope="module")
def seeded(server: Server) -> Iterator[Seeded]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_USERNAME", ADMIN_USER)
        mp.setenv("MLFLOW_TRACKING_PASSWORD", ADMIN_PASSWORD)
        client = MlflowClient(tracking_uri=server.uri)
        experiment = client.create_experiment("served")  # default location: mlflow-artifacts:

        def make(name: str, parent: str | None) -> str:
            tags = {} if parent is None else {PARENT_RUN_TAG: parent}
            run_id = client.create_run(experiment, tags=tags, run_name=name).info.run_id
            client.log_text(run_id, f"content of {name}", f"sub/{name}.txt")
            client.set_terminated(run_id, "FINISHED")
            return run_id

        parent = make("parent", None)
        child = make("child", parent)
        grandchild = make("grandchild", child)
        assert client.get_run(parent).info.artifact_uri.startswith("mlflow-artifacts:")
        yield Seeded(experiment, parent, child, grandchild)


def _contents(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file() and p.name != MANIFEST_NAME}


def test_download_over_http_with_proxied_artifacts(
    server: Server, seeded: Seeded, credentials: None, tmp_path: Path
) -> None:
    gateway = MlflowGateway.connect(server.uri)  # must beat the bogus MLFLOW_TRACKING_URI
    parent_dir = Path(f"parent-{seeded.parent[:8]}")
    child_dir = parent_dir / f"child-{seeded.child[:8]}"
    grandchild_dir = child_dir / f"grandchild-{seeded.grandchild[:8]}"
    expected = {
        str(parent_dir / "sub" / "parent.txt"): "content of parent",
        str(child_dir / "sub" / "child.txt"): "content of child",
        str(grandchild_dir / "sub" / "grandchild.txt"): "content of grandchild",
    }

    by_experiment = tmp_path / "by-experiment"
    assert download_experiments(gateway, ["served"], by_experiment) == 3
    assert _contents(by_experiment / "served") == expected

    by_run = tmp_path / "by-run"
    assert download_runs(gateway, [seeded.parent], by_run) == 3
    assert _contents(by_run) == expected


def test_cli_wrong_password_exits_one_with_one_line_and_writes_nothing(
    server: Server,
    seeded: Seeded,
    credentials: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_fails_cleanly,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", server.uri)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "wrong-password")
    out = tmp_path / "out"

    result = CliRunner().invoke(app, ["run", seeded.parent, "-o", str(out)])

    assert_fails_cleanly(result, 1)
    assert not out.exists()
