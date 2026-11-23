"""MLflow adapter: the only module that touches ``MlflowClient``. Effects: network."""

from __future__ import annotations

from pathlib import Path

from mlflow.entities import Experiment, FileInfo, Metric, Run
from mlflow.tracking import MlflowClient

from mlflow_pull.paths import RelPath
from mlflow_pull.ports import ArtifactDir, ArtifactEntry, ArtifactFile, size_from_listing

SEARCH_PAGE_SIZE = 50_000  # MLflow's server-side maximum per search call


def entry_from_info(info: FileInfo) -> ArtifactEntry:
    """The boundary where a server-supplied path becomes a validated ``RelPath``."""
    path = RelPath(info.path)
    if info.is_dir:
        return ArtifactDir(path)
    return ArtifactFile(path, size_from_listing(info.file_size, path))


class MlflowGateway:
    def __init__(self, client: MlflowClient) -> None:
        self._client = client

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

    def list_artifacts(self, run_id: str, path: RelPath | None) -> list[ArtifactEntry]:
        infos = self._client.list_artifacts(run_id, None if path is None else path.value)
        return [entry_from_info(info) for info in infos]

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        # MLflow writes the file under its own name inside the directory it is given.
        self._client.download_artifacts(run_id, path.value, str(path.under(dest_dir).parent))
