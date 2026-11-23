"""What a download costs the tracking server, measured over real HTTP against ``mlflow server``.

Run with ``pytest -m slow`` (deselected by default).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
from mlflow.tracking import MlflowClient

from mlflow_pull import MlflowGateway, RelPath, download_runs
from test_server import ADMIN_PASSWORD, ADMIN_USER, Server, credentials, server  # noqa: F401  (fixtures)

pytestmark = pytest.mark.slow

FILES = 20
MAX_REQUESTS_PER_FILE = 3  # the artifact repository path costs about 2; the per-file client path cost 5.15
NAMES = [f"dir{i % 4}/file{i}.txt" for i in range(FILES)]


class RequestCounter:
    def __init__(self) -> None:
        self.count = 0


@pytest.fixture
def requests_made(monkeypatch: pytest.MonkeyPatch) -> RequestCounter:
    counter = RequestCounter()
    real = requests.Session.request

    def counting(self: requests.Session, *args: object, **kwargs: object) -> requests.Response:
        counter.count += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "request", counting)
    return counter


@pytest.fixture
def served_run(server: Server) -> Iterator[str]:  # noqa: F811
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_TRACKING_USERNAME", ADMIN_USER)
        mp.setenv("MLFLOW_TRACKING_PASSWORD", ADMIN_PASSWORD)
        client = MlflowClient(tracking_uri=server.uri)
        experiment = client.create_experiment(f"cost-{uuid.uuid4().hex}")
        run_id = client.create_run(experiment, run_name="many").info.run_id
        for name in NAMES:
            client.log_text(run_id, f"content of {name}", name)
        client.set_terminated(run_id, "FINISHED")
        yield run_id


def test_a_download_costs_few_requests_per_file(
    server: Server, served_run: str, credentials: None, requests_made: RequestCounter, tmp_path: Path  # noqa: F811
) -> None:
    gateway = MlflowGateway.connect(server.uri)
    destination = tmp_path / "dst"
    requests_made.count = 0

    for name in NAMES:
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        gateway.download_artifact(served_run, RelPath(name), destination)

    per_file = requests_made.count / FILES
    print(f"requests per file: {per_file:.2f}")
    assert per_file <= MAX_REQUESTS_PER_FILE
    assert {p.relative_to(destination).as_posix(): p.read_text() for p in destination.rglob("*") if p.is_file()} == {
        name: f"content of {name}" for name in NAMES
    }


def test_the_whole_pull_lays_files_out_as_before(
    server: Server, served_run: str, credentials: None, tmp_path: Path  # noqa: F811
) -> None:
    gateway = MlflowGateway.connect(server.uri)

    assert download_runs(gateway, [served_run], tmp_path / "out") == 1

    run_dir = next((tmp_path / "out").iterdir())
    assert (run_dir / "dir1" / "file1.txt").read_text() == "content of dir1/file1.txt"
