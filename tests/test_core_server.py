"""What a download costs the tracking server, measured over real HTTP against ``mlflow server``.

Run with ``pytest -m slow`` (deselected by default).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlflow_pull import Downloaded, MlflowGateway, PullOptions, RelPath, download_runs, pull, resolve_runs
from mlflow_pull.plan import MANIFEST_NAME
from mlflow_pull.selection import FINISHED
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

    with gateway.artifact_session(served_run):
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


# --- names an HTTP-served store cannot carry ----------------------------------------------------------------
# The proxy and ``HttpArtifactRepository`` join the raw path into the endpoint URL (checked against MLflow 3.17):
# ``?`` cuts the URL, ``%41`` is decoded by the server and ``#`` is refused by MLflow's own listing check.

SUPPORTED = {"ok.txt": "fine", "dir/fine.txt": "also fine"}
OTHER_FILE = "aAb.txt"  # what ``a%41b.txt`` is decoded to


@pytest.fixture
def plant_odd_names(server: Server, served_run: str) -> Callable[[dict[str, str]], str]:  # noqa: F811
    """Make a FINISHED run whose store also holds names no client can upload over HTTP: they are planted on disk."""

    def make(planted: dict[str, str]) -> str:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MLFLOW_TRACKING_USERNAME", ADMIN_USER)
            mp.setenv("MLFLOW_TRACKING_PASSWORD", ADMIN_PASSWORD)
            client = MlflowClient(tracking_uri=server.uri)
            run_id = client.create_run(client.get_run(served_run).info.experiment_id, run_name="odd").info.run_id
            for name, text in SUPPORTED.items():
                client.log_text(run_id, text, name)
            client.set_terminated(run_id, "FINISHED")
        root = next(server.artifacts.glob(f"*/{run_id}/artifacts"))
        for name, text in planted.items():
            (root / name).write_text(text)
        return run_id

    return make


def test_the_http_route_fetches_nothing_or_another_file_for_the_names_the_plan_skips(
    server: Server, plant_odd_names: Callable[..., str], credentials: None, tmp_path: Path  # noqa: F811
) -> None:
    run_id = plant_odd_names({"a?b.txt": "q", OTHER_FILE: "the other file", "a%41b.txt": "mine"})
    gateway = MlflowGateway.connect(server.uri)
    (tmp_path / "dst").mkdir()

    with gateway.artifact_session(run_id):
        with pytest.raises(MlflowException, match="failures occurred"):
            gateway.download_artifact(run_id, RelPath("a?b.txt"), tmp_path / "dst")
        gateway.download_artifact(run_id, RelPath("a%41b.txt"), tmp_path / "dst")

    assert (tmp_path / "dst" / "a%41b.txt").read_text() == "the other file"  # aAb.txt, not a%41b.txt


def test_a_name_with_a_question_mark_is_reported_and_the_rest_downloads_over_http(
    server: Server, plant_odd_names: Callable[..., str], credentials: None, tmp_path: Path  # noqa: F811
) -> None:
    run_id = plant_odd_names({"a?b.txt": "q", "sub?dir.txt": "q"})
    gateway = MlflowGateway.connect(server.uri)
    out = tmp_path / "out"

    (outcome,) = pull(gateway, resolve_runs(gateway, [run_id], out, FINISHED), PullOptions())

    assert isinstance(outcome, Downloaded)
    assert (outcome.files, outcome.unsupported) == (len(SUPPORTED), 2)
    run_dir = out / f"odd-{run_id[:8]}"
    fetched = {p.relative_to(run_dir).as_posix(): p.read_text() for p in run_dir.rglob("*") if p.is_file() and p.name != MANIFEST_NAME}
    assert fetched == SUPPORTED


def test_a_name_with_a_hash_makes_the_http_listing_fail_before_anything_is_written(
    server: Server, plant_odd_names: Callable[..., str], credentials: None, tmp_path: Path  # noqa: F811
) -> None:
    run_id = plant_odd_names({"a#b.txt": "h"})
    gateway = MlflowGateway.connect(server.uri)
    out = tmp_path / "out"

    with pytest.raises(MlflowException, match="Invalid path"):  # MLflow's own listing check, not ours to skip
        pull(gateway, resolve_runs(gateway, [run_id], out, FINISHED), PullOptions())

    assert not out.exists()
