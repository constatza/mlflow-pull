from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from mlflow_pull import PARENT_RUN_TAG, download_experiment, download_run_tree, main


@pytest.fixture
def client(tmp_path: Path) -> MlflowClient:
    return MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")


@pytest.fixture
def tree(client: MlflowClient, tmp_path: Path) -> dict[str, str]:
    """parent -> (child_a -> grandchild, child_b); plus an unrelated root run."""
    experiment_id = client.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())

    def make(name: str, parent: str | None, files: dict[str, str]) -> str:
        tags = {} if parent is None else {PARENT_RUN_TAG: parent}
        run_id = client.create_run(experiment_id, tags=tags, run_name=name).info.run_id
        for artifact_path, text in files.items():
            client.log_text(run_id, text, artifact_path)
        return run_id

    parent = make("parent", None, {"top.txt": "p", "dir/nested/deep.txt": "d"})
    child_a = make("child_a", parent, {"a.txt": "a"})
    make("grandchild", child_a, {"g.txt": "g"})
    make("child_b", parent, {"b.txt": "b"})
    other = make("other", None, {"o.txt": "o"})
    return {"parent": parent, "other": other}


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_run_tree_includes_nested_runs_and_directories(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    count = download_run_tree(client, tree["parent"], out)

    parent_dir = f"parent-{tree['parent'][:8]}"
    names = {p.name.rsplit("-", 1)[0] for p in out.rglob("*") if p.is_dir()}
    assert count == 4
    assert {"parent", "child_a", "grandchild", "child_b", "dir", "nested"} <= names
    files = _files(out)
    assert {f.split("/", 1)[1] for f in files if f.startswith(parent_dir)} >= {
        "top.txt",
        "dir/nested/deep.txt",
    }
    assert sorted(f.rsplit("/", 1)[1] for f in files) == [
        "a.txt", "b.txt", "deep.txt", "g.txt", "top.txt",
    ]


def test_experiment_downloads_every_top_level_run(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    count = download_experiment(client, "exp", out)

    assert count == 5
    assert sorted(f.rsplit("/", 1)[1] for f in _files(out)) == [
        "a.txt", "b.txt", "deep.txt", "g.txt", "o.txt", "top.txt",
    ]


def test_out_option_follows_subcommand(
    client: MlflowClient, tree: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", client.tracking_uri)
    out = tmp_path / "cli-out"
    main(["run", tree["other"], "-o", str(out)])

    assert sorted(f.rsplit("/", 1)[1] for f in _files(out)) == ["o.txt"]
