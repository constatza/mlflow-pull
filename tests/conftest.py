"""Fixtures shared by the test modules: a sqlite-backed MLflow store, its gateway and the CLI."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import Result
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from mlflow_pull import MlflowGateway, RunGateway, app
from mlflow_pull.paths import RelPath
from mlflow_pull.ports import ListedEntry

AMBIENT_MLFLOW_VARIABLES = frozenset({
    "MLFLOW_EXPERIMENT_ID", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_TRACKING_URI",
})


@pytest.fixture(autouse=True)
def scrubbed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's own settings must not leak into a test; a test that needs one sets it."""
    for name in list(os.environ):
        if name.startswith("MLPULL_") or name in AMBIENT_MLFLOW_VARIABLES:
            monkeypatch.delenv(name)


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")


@pytest.fixture
def gateway(client: MlflowClient) -> RunGateway:
    return MlflowGateway(client)


@pytest.fixture
def cli(client: MlflowClient, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Result]:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", client.tracking_uri)
    return lambda *args: CliRunner().invoke(app, list(args))


@pytest.fixture
def assert_fails_cleanly() -> Callable[..., str]:
    """The CLI contract for a failure: the exit code, one line (the last, which is returned) and no escaped exception.

    ``notes_before`` allows earlier human notes, such as a ``skipped`` line, ahead of the failure line.
    """

    def check(result: Result, code: int, notes_before: bool = False) -> str:
        assert result.exit_code == code, repr(result.exception)
        assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
        # CliRunner prints a SystemExit message on stdout; a real process prints it on stderr.
        lines = result.output.strip().splitlines()
        assert (len(lines) == 1 or notes_before) and lines and "Traceback" not in result.output, result.output
        return lines[-1]

    return check


Listing = Callable[[RelPath | None], list[ListedEntry]]


class ScriptedListing(MlflowGateway):
    """A real gateway whose artifact listing is a function: a server holding odd names, or one that fails."""

    def __init__(self, client: MlflowClient, listing: Listing) -> None:
        super().__init__(client)
        self._listing = listing

    def list_artifacts(self, run_id: str, path: RelPath | None) -> list[ListedEntry]:
        return self._listing(path)


@pytest.fixture
def serve_listing(client: MlflowClient, monkeypatch: pytest.MonkeyPatch) -> Callable[[Listing], None]:
    """Make the CLI talk to ``ScriptedListing(listing)`` instead of connecting to a server."""

    def serve(listing: Listing) -> None:
        gateway = ScriptedListing(client, listing)
        monkeypatch.setattr(MlflowGateway, "connect", classmethod(lambda cls, uri: gateway))

    return serve
