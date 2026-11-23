"""MLflow adapter: the only module that touches ``MlflowClient``. Effects: network."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from mlflow.entities import Experiment, FileInfo, Metric, Run
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import MlflowClient
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri

from mlflow_pull.paths import RelPath, UnsupportedNameError
from mlflow_pull.ports import ArtifactDir, ArtifactFile, ListedEntry, UnsupportedEntry, size_from_listing

SEARCH_PAGE_SIZE = 50_000  # MLflow's server-side maximum per search call


def entry_from_info(info: FileInfo) -> ListedEntry:
    """The boundary where a server-supplied path becomes a validated ``RelPath``.

    A hostile path raises ``UnsafePathError``; one this machine cannot create becomes an ``UnsupportedEntry``.
    """
    try:
        path = RelPath(info.path)
    except UnsupportedNameError:
        return UnsupportedEntry.named(info.path)
    if info.is_dir:
        return ArtifactDir(path)
    return ArtifactFile(path, size_from_listing(info.file_size, path))


class MlflowGateway:
    def __init__(self, client: MlflowClient) -> None:
        self._client = client
        # One per worker thread and run inside ``artifact_session``, gone after it: a run planned twice
        # (named twice, or also inside another named run) is downloaded by two workers at once.
        self._open: dict[tuple[int, str], ArtifactRepository] = {}

    @classmethod
    def connect(cls, tracking_uri: str | None) -> MlflowGateway:
        """``None`` means the standard MLflow environment (``MLFLOW_TRACKING_URI``, credentials)."""
        return cls(MlflowClient(tracking_uri=tracking_uri))

    def get_run(self, run_id: str) -> Run:
        return self._client.get_run(run_id)

    def search_runs(self, experiment_id: str, filter_string: str | None = None) -> list[Run]:
        return list(
            self._client.search_runs(
                [experiment_id], filter_string or "", max_results=SEARCH_PAGE_SIZE
            )
        )

    def find_experiment(self, name_or_id: str) -> Experiment:
        return self._client.get_experiment_by_name(name_or_id) or self._client.get_experiment(name_or_id)

    def search_experiments(self, filter_string: str) -> list[Experiment]:
        found: list[Experiment] = []
        token: str | None = None
        while True:  # one page holds at most 1000 experiments
            page = self._client.search_experiments(filter_string=filter_string, page_token=token)
            found += page
            token = page.token
            if not token:
                return found

    def metric_history(self, run_id: str, key: str) -> list[Metric]:
        return list(self._client.get_metric_history(run_id, key))

    def list_artifacts(self, run_id: str, path: RelPath | None) -> list[ListedEntry]:
        try:
            infos = self._client.list_artifacts(run_id, None if path is None else path.value)
        except MlflowException as err:  # the server's message names no run
            raise MlflowException(f"run {run_id}: {err.message}", error_code=ErrorCode.Value(err.error_code)) from err
        return [entry_from_info(info) for info in infos]

    @contextmanager
    def artifact_session(self, run_id: str) -> Iterator[None]:
        """Downloads of ``run_id`` inside the block, on the thread that opened it, share one artifact repository, closed when it ends.

        Each thread has its own session: one worker per planned run, and a run can be planned twice.

        ``MlflowClient.download_artifact`` resolves a ``runs:/`` URI on every call (about 5 requests per
        file over HTTP); a repository costs about 2 (a listing and the transfer). The repository is built
        as ``MlflowClient._get_artifact_repo`` builds it, from the client's own tracking URI and never the
        environment, so an SSH tunnel's address is honoured. MLflow gives it a thread pool that is never
        shut down by anything but garbage collection (it has no ``close``), so the pool is shut down here:
        a pull over thousands of runs must not keep a thread and a store client alive per run.
        """
        key = (threading.get_ident(), run_id)
        if key in self._open:
            raise RuntimeError(f"run {run_id} already has an open artifact session on this thread")
        artifact_uri = add_databricks_profile_info_to_artifact_uri(
            self._client.get_run(run_id).info.artifact_uri, self._client.tracking_uri
        )
        repository = get_artifact_repository(artifact_uri, tracking_uri=self._client.tracking_uri)
        self._open[key] = repository
        try:
            yield
        finally:
            del self._open[key]
            # A repository from a plugin may have no thread pool.
            if isinstance(pool := getattr(repository, "thread_pool", None), ThreadPoolExecutor):
                pool.shutdown()

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        # The repository writes the file at ``dest_dir / path``, which ``prepare_destination`` has made room for.
        if (repository := self._open.get((threading.get_ident(), run_id))) is None:
            raise RuntimeError(f"run {run_id} has no open artifact session on this thread")
        repository.download_artifacts(path.value, str(dest_dir))
