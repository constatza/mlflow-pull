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

AMBIENT_MLFLOW_VARIABLES = frozenset({
    "MLFLOW_EXPERIMENT_ID", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN",
})


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")


@pytest.fixture
def gateway(client: MlflowClient) -> RunGateway:
    return MlflowGateway(client)


@pytest.fixture
def cli(client: MlflowClient, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Result]:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", client.tracking_uri)
    for name in list(os.environ):  # the developer's own settings must not leak into a test
        if name.startswith("MLPULL_") or name in AMBIENT_MLFLOW_VARIABLES:
            monkeypatch.delenv(name)
    return lambda *args: CliRunner().invoke(app, list(args))
