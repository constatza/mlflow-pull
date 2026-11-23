"""The gateway's artifact route: one repository per run, built like MLflow's own client, released after the run."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.tracking import MlflowClient

from mlflow_pull import MlflowGateway, PullOptions, RelPath, pull, resolve_runs
from mlflow_pull import gateway as gateway_module
from mlflow_pull.selection import FINISHED

DATABRICKS_ARTIFACT_URI = "dbfs:/databricks/mlflow-tracking/1/abc/artifacts"
FILES = {"a.txt": "alpha", "sub/b.txt": "beta"}


@dataclass(slots=True)
class Built:
    """What the gateway asked the artifact-repository registry for, and what it got back."""

    calls: list[tuple[str, str | None]] = field(default_factory=list)
    repositories: list[ArtifactRepository] = field(default_factory=list)


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> Built:
    record = Built()
    real = gateway_module.get_artifact_repository

    def recording(artifact_uri: str, tracking_uri: str | None = None) -> ArtifactRepository:
        repository = real(artifact_uri, tracking_uri=tracking_uri)
        record.calls.append((artifact_uri, tracking_uri))
        record.repositories.append(repository)
        return repository

    monkeypatch.setattr(gateway_module, "get_artifact_repository", recording)
    return record


def _is_released(repository: ArtifactRepository) -> bool:
    """A shut-down pool refuses new work."""
    try:
        repository.thread_pool.submit(int)
    except RuntimeError:
        return True
    return False


@pytest.fixture
def two_runs(client: MlflowClient, tmp_path: Path) -> list[str]:
    experiment = client.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    run_ids = []
    for name in ("first", "second"):
        run_id = client.create_run(experiment, run_name=name).info.run_id
        for path, text in FILES.items():
            client.log_text(run_id, text, path)
        client.set_terminated(run_id, "FINISHED")
        run_ids.append(run_id)
    return run_ids


def test_a_pull_builds_one_repository_per_run_from_the_clients_tracking_uri_and_releases_it(
    client: MlflowClient, gateway: MlflowGateway, built: Built, two_runs: list[str], tmp_path: Path
) -> None:
    out = tmp_path / "out"

    pull(gateway, resolve_runs(gateway, two_runs, out, FINISHED), PullOptions(), jobs=2)

    assert sorted(uri for uri, _ in built.calls) == sorted(client.get_run(r).info.artifact_uri for r in two_runs)
    assert {tracking for _, tracking in built.calls} == {client.tracking_uri}  # never the environment's
    assert all(_is_released(repository) for repository in built.repositories)
    assert {p.name: p.read_text() for p in out.rglob("*.txt")} == {"a.txt": "alpha", "b.txt": "beta"}


def test_many_files_of_a_run_share_its_repository(
    gateway: MlflowGateway, built: Built, two_runs: list[str], tmp_path: Path
) -> None:
    (tmp_path / "sub").mkdir()

    with gateway.artifact_session(two_runs[0]):
        for name in FILES:
            gateway.download_artifact(two_runs[0], RelPath(name), tmp_path)

    assert len(built.calls) == 1
    assert _is_released(built.repositories[0])


def test_a_session_is_released_when_its_block_fails(gateway: MlflowGateway, built: Built, two_runs: list[str]) -> None:
    with pytest.raises(ConnectionError), gateway.artifact_session(two_runs[0]):
        raise ConnectionError("the server went away")

    assert _is_released(built.repositories[0])
    with gateway.artifact_session(two_runs[0]):  # and the run can be opened again
        pass


def test_a_download_outside_a_session_is_refused(gateway: MlflowGateway, two_runs: list[str], tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no open artifact session"):
        gateway.download_artifact(two_runs[0], RelPath("a.txt"), tmp_path)


def test_one_run_cannot_have_two_sessions(gateway: MlflowGateway, two_runs: list[str]) -> None:
    with gateway.artifact_session(two_runs[0]), pytest.raises(RuntimeError, match="already has an open"):
        with gateway.artifact_session(two_runs[0]):
            pass


# --- Databricks profiles ----------------------------------------------------------------------------------


def _gateway_of_a_stub(tracking_uri: str, artifact_uri: str, monkeypatch: pytest.MonkeyPatch) -> tuple[MlflowGateway, list[str]]:
    """A gateway over a client that only knows one run, and the artifact URIs it asked the registry for."""
    asked: list[str] = []

    def registry(artifact_uri: str, tracking_uri: str | None = None) -> SimpleNamespace:
        asked.append(artifact_uri)
        return SimpleNamespace(thread_pool=ThreadPoolExecutor(1))

    monkeypatch.setattr(gateway_module, "get_artifact_repository", registry)
    client = SimpleNamespace(
        tracking_uri=tracking_uri, get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(artifact_uri=artifact_uri))
    )
    return MlflowGateway(cast(MlflowClient, client)), asked


@pytest.mark.parametrize(
    ("tracking_uri", "expected"),
    [
        ("databricks://prof", "dbfs://prof@databricks/databricks/mlflow-tracking/1/abc/artifacts"),
        ("databricks", "dbfs://databricks/databricks/mlflow-tracking/1/abc/artifacts"),
        ("http://127.0.0.1:5000", DATABRICKS_ARTIFACT_URI),
        ("sqlite:///x.db", DATABRICKS_ARTIFACT_URI),
    ],
)
def test_a_databricks_tracking_uri_adds_its_profile_to_the_artifact_uri_like_the_mlflow_client_does(
    tracking_uri: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, asked = _gateway_of_a_stub(tracking_uri, DATABRICKS_ARTIFACT_URI, monkeypatch)

    with gateway.artifact_session("abc"):
        pass

    assert asked == [expected]


def test_a_listing_failure_names_the_run_and_keeps_its_error_code(tmp_path: Path) -> None:
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    class FailingListing(MlflowClient):
        def list_artifacts(self, run_id, path=None):
            raise MlflowException("Invalid path", error_code=INVALID_PARAMETER_VALUE)

    gateway = MlflowGateway(FailingListing(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"))

    with pytest.raises(MlflowException, match=r"^run abc123: Invalid path") as caught:
        gateway.list_artifacts("abc123", None)
    assert caught.value.error_code == "INVALID_PARAMETER_VALUE"
