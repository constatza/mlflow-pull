"""Selective, cheaper downloads: path globs, size limit, dry run, manifest, ``--what``."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import Result
from mlflow.entities import FileInfo
from mlflow.tracking import MlflowClient

from mlflow_pull import (
    PARENT_RUN_TAG,
    ArtifactDir,
    ArtifactEntry,
    ArtifactFile,
    Content,
    Downloaded,
    MlflowGateway,
    PathFilter,
    PullOptions,
    RelPath,
    RunGateway,
    SizeLimitError,
    UpToDate,
    What,
    download_runs,
    execute_plan,
    parse_size,
    plan_pull,
    plan_run,
    pull,
    resolve_runs,
)
from mlflow_pull import execute, paths
from mlflow_pull.execute import MAX_DEPTH, ListingLimitError
from mlflow_pull.gateway import entry_from_info
from mlflow_pull.localfs import STALE_TEMP_SECONDS, NotRegularFileError, SizeMismatchError, SymlinkError, temp_name
from mlflow_pull.paths import UnsafePathError
from mlflow_pull.plan import (
    MANIFEST_NAME,
    SUMMARY_NAME,
    UNRECORDED,
    DuplicatePathError,
    Manifest,
    OnDiskFile,
    PrefixConflictError,
    ReservedNameError,
    UnknownSizeError,
    parse_manifest,
)
from mlflow_pull.ports import UNKNOWN_SIZE, FileSize, InvalidSizeError, KnownSize, UnsupportedEntry
from mlflow_pull.report import render_outcomes
from mlflow_pull.selection import FINISHED, RunStatus, allowed_statuses
from mlflow_pull.tree import RunTarget

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="needs real symlinks and FIFOs")

Cli = Callable[..., Result]

SIZES = {
    "config/a.yaml": 10,
    "config/sub/b.yaml": 20,
    "plots/p.png": 30,
    "plots/deep/q.png": 40,
    "model/w.bin": 500,
    "top.txt": 5,
}
TOTAL = sum(SIZES.values())
EVERYTHING_PULLED = frozenset(SIZES)


@dataclass(frozen=True, slots=True)
class Store:
    client: MlflowClient
    experiment_id: str
    parent: str  # FINISHED, with ``SIZES`` as artifacts
    child: str  # FINISHED, nested under parent, one artifact
    live: str  # RUNNING, one artifact, one metric, a param and a tag


class CountingGateway(MlflowGateway):
    """A real gateway that records every listing and every file download."""

    def __init__(self, client: MlflowClient) -> None:
        super().__init__(client)
        self.listed: list[str | None] = []
        self.downloaded: list[str] = []

    def list_artifacts(self, run_id: str, path: RelPath | None):
        self.listed.append(None if path is None else path.value)
        return super().list_artifacts(run_id, path)

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        self.downloaded.append(path.value)
        super().download_artifact(run_id, path, dest_dir)


def _make_run(
    client: MlflowClient, experiment_id: str, name: str, files: dict[str, int], parent: str | None = None
) -> str:
    tags = {} if parent is None else {PARENT_RUN_TAG: parent}
    run_id = client.create_run(experiment_id, tags=tags, run_name=name).info.run_id
    for path, size in files.items():
        client.log_text(run_id, "x" * size, path)
    return run_id


@pytest.fixture
def store(client: MlflowClient, tmp_path: Path) -> Store:
    experiment_id = client.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    parent = _make_run(client, experiment_id, "parent", SIZES)
    child = _make_run(client, experiment_id, "child", {"c.txt": 7}, parent)
    live = _make_run(client, experiment_id, "live", {"l.txt": 3})
    client.log_metric(live, "loss", 0.5, step=1)
    client.log_metric(live, "loss", 0.25, step=2)
    client.log_param(live, "lr", "0.1")
    client.set_tag(live, "phase", "train")
    client.set_terminated(parent, "FINISHED")
    client.set_terminated(child, "FINISHED")
    return Store(client, experiment_id, parent, child, live)


@pytest.fixture
def counting(store: Store) -> CountingGateway:
    return CountingGateway(store.client)


def _tree(root: Path) -> frozenset[str]:
    """Relative paths of the files under ``root``, mlpull's own files left out."""
    return frozenset(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME
    )


def _on_disk(directory: Path, sizes: dict[str, int]) -> dict[str, dict[str, int]]:
    """What the manifest must say about each file: its size and mtime as the file system reports them now."""
    stats = {name: (directory / name).stat() for name in sizes}
    assert {name: found.st_size for name, found in stats.items()} == sizes
    return {name: {"size": found.st_size, "mtime_ns": found.st_mtime_ns} for name, found in stats.items()}


def _parent_dir(out: Path, store: Store) -> Path:
    return out / f"parent-{store.parent[:8]}"


def _pull(
    gateway: RunGateway, run_ids: list[str], out: Path, options: PullOptions = PullOptions(),
    allowed: frozenset[str] = FINISHED, jobs: int = execute.DEFAULT_JOBS,
):
    return pull(gateway, resolve_runs(gateway, run_ids, out, allowed), options, jobs)


def _options(*globs: str, max_bytes: int | None = None) -> PullOptions:
    return PullOptions(path_filter=PathFilter.parse(globs), max_bytes=max_bytes)


# --- path globs ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("globs", "path", "accepted"),
    [
        ([], "anything/at/all.txt", True),
        (["config/**"], "config/a.yaml", True),
        (["config/**"], "config/sub/b.yaml", True),
        (["config/**"], "configs/a.yaml", False),
        (["config/**"], "plots/p.png", False),
        (["plots/*"], "plots/p.png", True),
        (["plots/*"], "plots/deep/q.png", False),  # * does not cross /
        (["plots/*.png"], "plots/p.png", True),
        (["plots/*.png"], "plots/p.txt", False),
        (["plots/**"], "plots/deep/q.png", True),
        (["**/q.png"], "plots/deep/q.png", True),
        (["**/q.png"], "q.png", True),  # ** also matches zero directories
        (["plots/**/q.png"], "plots/q.png", True),
        (["plots/**/q.png"], "plots/a/b/q.png", True),
        (["plots/**/q.png"], "plotsx/q.png", False),
        (["top.tx?"], "top.txt", True),
        (["a.b"], "aXb", False),  # a dot is a dot
        (["*"], "top.txt", True),
        (["*"], "config/a.yaml", False),
        (["**"], "config/a.yaml", True),
        (["config/**", "top.txt"], "top.txt", True),  # includes add up
        (["config/**", "!config/sub/**"], "config/sub/b.yaml", False),  # excludes win
        (["config/**", "!config/sub/**"], "config/a.yaml", True),
        (["!model/**"], "model/w.bin", False),  # only excludes: everything else
        (["!model/**"], "top.txt", True),
        (["!**/*.png", "plots/**"], "plots/p.png", False),
    ],
)
def test_path_filter_accepts(globs: list[str], path: str, accepted: bool) -> None:
    assert PathFilter.parse(globs).accepts(RelPath(path)) is accepted


@pytest.mark.parametrize(
    ("glob", "reason"),
    [
        ("", "empty"),
        ("!", "empty"),
        ("/abs/**", "relative paths only"),
        ("a//b", "relative paths only"),
        ("a\\b", "use / as the separator"),
        ("../x", r"'\.\.' never matches"),
        ("./x", r"'\.' never matches"),
        ("a/./b", r"'\.' never matches"),
        (".", r"'\.' never matches"),
    ],
)
def test_path_filter_rejects_malformed_globs(glob: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        PathFilter.parse([glob])


@pytest.mark.parametrize("glob", ["plots/", "a/b/", "!plots/"])
def test_a_trailing_slash_glob_says_what_to_write_instead(glob: str) -> None:
    directory = glob.removeprefix("!").rstrip("/")
    with pytest.raises(ValueError, match=rf"trailing /.*'{directory}/\*\*'"):
        PathFilter.parse([glob])


def test_a_bracket_is_a_literal_character_in_a_glob() -> None:
    assert PathFilter.parse(["a[1].txt"]).accepts(RelPath("a[1].txt"))
    assert not PathFilter.parse(["a[1].txt"]).accepts(RelPath("a1.txt"))


@pytest.mark.parametrize(
    ("globs", "directory", "listed"),
    [
        ([], "plots", True),
        (["config/**"], "config", True),
        (["config/**"], "config/sub", True),
        (["config/**"], "plots", False),  # provably nothing below plots matches
        (["plots/*.png"], "plots", True),
        (["plots/*.png"], "plots/deep", False),  # * cannot reach into a subdirectory
        (["plots/**/q.png"], "plots/deep", True),
        (["**/q.png"], "anything/deep", True),
        (["*.txt"], "plots", False),
        (["config/**", "plots/*.png"], "plots", True),
        (["a/b/c.txt"], "a", True),
        (["a/b/c.txt"], "a/b", True),
        (["a/b/c.txt"], "a/x", False),
        (["a/b/c.txt"], "a/b/c.txt", False),
        (["**", "!model/**"], "model", False),  # a fully excluded directory is not listed
        (["**", "!model/**"], "plots", True),
        (["!**/cache/**"], "a/b/cache", False),
        (["!model/*"], "model", True),  # only direct children are excluded: model/x/y stays wanted
    ],
)
def test_path_filter_prunes_directories_only_when_safe(globs: list[str], directory: str, listed: bool) -> None:
    assert PathFilter.parse(globs).descends_into(RelPath(directory)) is listed


def test_pruning_never_drops_an_accepted_file() -> None:
    """The safety property: a file the filter accepts is always reachable through descends_into."""
    globs = [["config/**"], ["plots/*.png"], ["**/q.png"], ["a/*/c"], ["**", "!model/**"], ["!plots/**"]]
    paths = [RelPath(p) for p in (*SIZES, "a/x/c", "a/x/y/c", "model/deep/z", "plots/deep/deeper/q.png")]
    for spec in globs:
        path_filter = PathFilter.parse(spec)
        for path in paths:
            if not path_filter.accepts(path):
                continue
            ancestors = [RelPath("/".join(path.parts[:i])) for i in range(1, len(path.parts))]
            assert all(path_filter.descends_into(d) for d in ancestors), (spec, path)


# --- sizes -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("1", 1),
        ("1234", 1234),
        ("1K", 1024),
        ("500M", 500 * 1024**2),
        ("1G", 1024**3),
        ("1.5G", 3 * 1024**3 // 2),
        ("0.5k", 512),
        ("2T", 2 * 1024**4),
        ("2GB", 2 * 1024**3),
        ("2GiB", 2 * 1024**3),
        ("10B", 10),
        (" 7M ", 7 * 1024**2),
        ("1.0000001", 1),  # rounds down
    ],
)
def test_parse_size(text: str, size: int) -> None:
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "G", "-1", "1.5.2G", "1 G G", "abc", "0", "0G", "1e3", "1X"])
def test_parse_size_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_size(text)


# --- RelPath: hostile artifact paths ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "../x", "..", "a/../../b", "/abs", "a//b", "a/", "./a", "a/./b", "a\x00b", "a\nb"],
)
def test_rel_path_rejects_hostile_values(text: str) -> None:
    with pytest.raises(UnsafePathError):
        RelPath(text)


def test_rel_path_accepts_plain_paths() -> None:
    path = RelPath("a/b/c.txt")
    assert path.parts == ("a", "b", "c.txt")
    assert path.is_child_of(RelPath("a/b")) and not path.is_child_of(RelPath("a")) and not path.is_child_of(None)
    assert RelPath("top.txt").is_child_of(None)


def test_a_colon_inside_a_component_is_fine_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", False)
    assert RelPath("dir/a:b").value == "dir/a:b"  # a colon is only a drive letter in the first component


class ListingClient(MlflowClient):
    """A real client whose artifact listing is scripted, standing in for a hostile server."""

    listing: dict[str, list[FileInfo]]

    def list_artifacts(self, run_id: str, path: str | None = None):
        return self.listing[path or ""]


@pytest.mark.parametrize(
    ("hostile", "reason"),
    [
        (FileInfo("../x", False, 1), r"'\.' or '\.\.' component"),
        (FileInfo("/etc/passwd", False, 1), "absolute"),
        (FileInfo("a/../../b", False, 1), r"'\.' or '\.\.' component"),
        (FileInfo("a\x00b", False, 1), "control character"),
        (FileInfo("dir", True, None), r"unsafe artifact path 'dir/\.\./\.\./x'"),  # hostile child, below
    ],
)
def test_hostile_listing_fails_the_plan_before_anything_is_written(
    store: Store, tmp_path: Path, hostile: FileInfo, reason: str
) -> None:
    client = ListingClient(tracking_uri=store.client.tracking_uri)
    client.listing = {"": [FileInfo("ok.txt", False, 1), hostile], "dir": [FileInfo("dir/../../x", False, 1)]}
    out = tmp_path / "out"

    with pytest.raises(UnsafePathError, match=reason):
        download_runs(MlflowGateway(client), [store.parent], out)

    assert not out.exists()


def test_listing_entry_outside_its_directory_is_rejected(store: Store, tmp_path: Path) -> None:
    client = ListingClient(tracking_uri=store.client.tracking_uri)
    client.listing = {"": [FileInfo("a", True, None)], "a": [FileInfo("a", True, None)]}  # a lists itself
    out = tmp_path / "out"

    with pytest.raises(UnsafePathError, match="not directly inside"):
        download_runs(MlflowGateway(client), [store.parent], out)

    assert not out.exists()


def test_cli_reports_a_hostile_listing_in_one_line(
    store: Store, cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = ListingClient(tracking_uri=store.client.tracking_uri)
    listing.listing = {"": [FileInfo("../x", False, 1)]}
    monkeypatch.setattr(MlflowGateway, "connect", classmethod(lambda cls, uri: cls(listing)))
    out = tmp_path / "out"

    result = cli("run", store.parent, "-o", str(out))

    assert result.exit_code == 1
    assert "unsafe artifact path" in result.output
    assert not out.exists()


# --- selecting by --path ---------------------------------------------------------------------


def test_cli_path_includes_and_excludes(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli(
        "run", store.parent, "--path", "config/**", "--path", "plots/*.png", "--path", "!config/sub/**",
        "-o", str(out),
    )

    assert result.exit_code == 0, result.output
    assert _tree(_parent_dir(out, store)) == {"config/a.yaml", "plots/p.png"}  # plots/deep/q.png: * stops at /


def test_cli_without_path_fetches_everything_and_applies_to_nested_runs(
    store: Store, cli: Cli, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert cli("run", store.parent, "-o", str(out)).exit_code == 0
    assert _tree(_parent_dir(out, store)) == EVERYTHING_PULLED | {f"child-{store.child[:8]}/c.txt"}

    only_c = tmp_path / "only-c"
    assert cli("run", store.parent, "--path", "c.txt", "-o", str(only_c)).exit_code == 0
    assert _tree(only_c) == {f"parent-{store.parent[:8]}/child-{store.child[:8]}/c.txt"}


def test_cli_experiment_takes_the_same_options(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("experiment", "exp", "--path", "top.txt", "--max-size", "1K", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert _tree(out) == {f"exp/parent-{store.parent[:8]}/top.txt"}


def test_listing_skips_directories_no_include_glob_can_reach(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    _pull(counting, [store.parent], tmp_path / "out", _options("config/**"))

    parent_listings = [d for d in counting.listed if d is None or d.split("/")[0] in {"config", "plots", "model"}]
    assert sorted(d for d in parent_listings if d) == ["config", "config/sub"]  # never plots or model
    assert sorted(counting.downloaded) == ["config/a.yaml", "config/sub/b.yaml"]


def test_excluded_directory_is_not_listed(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    _pull(counting, [store.parent], tmp_path / "out", _options("!model/**"))

    assert "model" not in counting.listed
    assert "model/w.bin" not in counting.downloaded and "top.txt" in counting.downloaded


# --- --max-size ------------------------------------------------------------------------------


def test_cli_max_size_refuses_before_writing_anything(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("run", store.parent, "--max-size", "100", "-o", str(out))

    assert result.exit_code == 1
    assert "refused (--max-size)" in result.output
    assert "612 B" in result.output and "100 B" in result.output  # the total (parent 605 + child 7) and the limit
    assert not out.exists()


def test_max_size_counts_only_what_would_be_fetched(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(SizeLimitError):
        _pull(counting, [store.parent], out, _options("plots/**", max_bytes=69))
    assert counting.downloaded == [] and not out.exists()

    _pull(counting, [store.parent], out, _options("plots/**", max_bytes=70))  # 30 + 40: exactly the limit
    assert sorted(counting.downloaded) == ["plots/deep/q.png", "plots/p.png"]

    _pull(counting, [store.parent], out, _options("plots/**", max_bytes=1))  # nothing left to fetch
    assert len(counting.downloaded) == 2


@pytest.mark.parametrize("value", ["lots", "0", "-5M"])
def test_cli_rejects_a_bad_max_size(store: Store, cli: Cli, value: str) -> None:
    assert cli("run", store.parent, "--max-size", value).exit_code == 2


def test_cli_rejects_a_bad_path_glob(store: Store, cli: Cli) -> None:
    assert cli("run", store.parent, "--path", "a//b").exit_code == 2


# --- --dry-run ---------------------------------------------------------------------------------


def test_cli_dry_run_prints_files_with_sizes_and_writes_nothing(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("run", store.parent, "--path", "config/**", "--dry-run", "-o", str(out))

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines()]
    assert "10 B  config/a.yaml" in lines and "20 B  config/sub/b.yaml" in lines
    assert not any("plots" in line for line in lines)
    assert "Would fetch 2 file(s), 30 B from 2 run(s)" in lines[-1]
    assert not out.exists()


def test_cli_dry_run_applies_and_reports_the_size_limit(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    refused = cli("run", store.parent, "--dry-run", "--max-size", "100", "-o", str(out))

    assert refused.exit_code == 1
    assert "model/w.bin" in refused.output and "refused (--max-size)" in refused.output
    assert not out.exists()
    assert cli("run", store.parent, "--dry-run", "--max-size", "1M", "-o", str(out)).exit_code == 0


def test_cli_dry_run_after_a_pull_lists_nothing_to_fetch(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert cli("run", store.parent, "-o", str(out)).exit_code == 0
    before = sorted(out.rglob("*"))

    result = cli("run", store.parent, "--dry-run", "-o", str(out))

    assert "Would fetch 0 file(s), 0 B" in result.output and "7 file(s) already up to date" in result.output
    assert sorted(out.rglob("*")) == before


# --- the manifest ----------------------------------------------------------------------------


def test_manifest_records_run_status_and_file_sizes(store: Store, gateway: RunGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(gateway, [store.parent], out)

    manifest = json.loads((_parent_dir(out, store) / MANIFEST_NAME).read_text())
    assert manifest == {"version": 2, "run_id": store.parent, "status": "FINISHED", "files": _on_disk(_parent_dir(out, store), SIZES)}
    child_dir = _parent_dir(out, store) / f"child-{store.child[:8]}"
    child_manifest = json.loads((child_dir / MANIFEST_NAME).read_text())
    assert child_manifest["files"] == _on_disk(child_dir, {"c.txt": 7}) and child_manifest["run_id"] == store.child
    assert not list(out.rglob("*.tmp"))


def test_second_pull_makes_no_download_call(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = _pull(counting, [store.parent], out)
    assert len(counting.downloaded) == 7
    assert [type(o) for o in first] == [Downloaded, Downloaded]

    counting.downloaded.clear()
    second = _pull(counting, [store.parent], out)

    assert counting.downloaded == []
    assert [type(o) for o in second] == [UpToDate, UpToDate]
    assert sum(o.files for o in second) == 7 and sum(o.bytes for o in second) == TOTAL + 7


def test_cli_final_line_counts_runs_files_and_bytes(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = cli("run", store.parent, "-o", str(out))
    assert f"Downloaded 2 run(s), 7 file(s), {TOTAL + 7} B to {out.resolve()}" in first.output.strip()
    assert "already up to date" not in first.output

    second = cli("run", store.parent, "-o", str(out))
    assert f"Downloaded 0 run(s), 0 file(s), 0 B to {out.resolve()} (2 run(s) already up to date)" in second.output


def test_a_run_pulled_while_running_is_fully_refetched_once_finished(
    store: Store, counting: CountingGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    everything = frozenset({"RUNNING", "FINISHED"})
    _pull(counting, [store.live], out, allowed=everything)
    assert counting.downloaded == ["l.txt"]

    counting.downloaded.clear()
    _pull(counting, [store.live], out, allowed=everything)  # still RUNNING: files may be changing
    assert counting.downloaded == ["l.txt"]

    store.client.set_terminated(store.live, "FINISHED")
    counting.downloaded.clear()
    _pull(counting, [store.live], out)
    assert counting.downloaded == ["l.txt"]

    counting.downloaded.clear()
    _pull(counting, [store.live], out)
    assert counting.downloaded == []


def test_a_changed_path_set_fetches_only_the_missing_files(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.parent], out, _options("config/**"))
    assert sorted(counting.downloaded) == ["config/a.yaml", "config/sub/b.yaml"]

    counting.downloaded.clear()
    _pull(counting, [store.parent], out, _options("config/**", "plots/**"))
    assert sorted(counting.downloaded) == ["plots/deep/q.png", "plots/p.png"]

    counting.downloaded.clear()
    _pull(counting, [store.parent], out, _options("plots/**"))  # narrower: config stays recorded, nothing to do
    assert counting.downloaded == []

    counting.downloaded.clear()
    _pull(counting, [store.parent], out)  # everything: only what no earlier pull fetched
    assert sorted(counting.downloaded) == ["c.txt", "model/w.bin", "top.txt"]
    assert _tree(_parent_dir(out, store)) == EVERYTHING_PULLED | {f"child-{store.child[:8]}/c.txt"}


def test_deleted_or_truncated_files_are_fetched_again(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.parent], out)
    (_parent_dir(out, store) / "top.txt").unlink()
    (_parent_dir(out, store) / "config" / "a.yaml").write_text("short")  # recorded size is 10

    counting.downloaded.clear()
    _pull(counting, [store.parent], out)

    assert sorted(counting.downloaded) == ["config/a.yaml", "top.txt"]
    assert (_parent_dir(out, store) / "config" / "a.yaml").stat().st_size == 10


def test_a_file_the_server_changed_in_size_is_fetched_again(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    store.client.log_text(store.child, "longer text", "c.txt")

    counting.downloaded.clear()
    _pull(counting, [store.child], out)

    assert counting.downloaded == ["c.txt"]
    assert (out / f"child-{store.child[:8]}" / "c.txt").read_text() == "longer text"


GOOD_FILES = '{"c.txt": {"size": 7, "mtime_ns": 5}}'


@pytest.mark.parametrize(
    "damage",
    [
        "",
        "not json",
        "[]",
        '{"version": 1, "run_id": "R", "status": "FINISHED", "files": {"c.txt": 7}}',  # the old format
        '{"version": 2, "run_id": "R", "status": "FINISHED"}',
        f'{{"version": true, "run_id": "R", "status": "FINISHED", "files": {GOOD_FILES}}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": {"c.txt": 7}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": {"c.txt": {"size": "7", "mtime_ns": 5}}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": {"c.txt": {"size": -7, "mtime_ns": 5}}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": {"c.txt": {"size": 7}}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": {"c.txt": {"size": 7, "mtime_ns": true}}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": {"../c.txt": {"size": 7, "mtime_ns": 5}}}',
        '{"version": 2, "run_id": "R", "status": "FINISHED", "files": ["c.txt"]}',
        f'{{"version": 2, "run_id": "someone else", "status": "FINISHED", "files": {GOOD_FILES}}}',
        "\xff\xfe binary \x00",
    ],
)
def test_an_unusable_manifest_means_nothing_is_recorded(
    store: Store, counting: CountingGateway, tmp_path: Path, damage: str
) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    (out / f"child-{store.child[:8]}" / MANIFEST_NAME).write_text(damage.replace("R", store.child, 1))

    counting.downloaded.clear()
    _pull(counting, [store.child], out)

    assert counting.downloaded == ["c.txt"]
    repaired = json.loads((out / f"child-{store.child[:8]}" / MANIFEST_NAME).read_text())
    assert repaired["files"]["c.txt"]["size"] == 7


def test_a_binary_manifest_is_not_a_crash(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    manifest = out / f"child-{store.child[:8]}" / MANIFEST_NAME
    manifest.write_bytes(b"\xff\xfe\x00")
    counting.downloaded.clear()
    _pull(counting, [store.child], out)
    assert counting.downloaded == ["c.txt"]


def test_parse_manifest_validates_the_file() -> None:
    good = '{"version": 2, "run_id": "r", "status": "FINISHED", "files": {"a/b.txt": {"size": 3, "mtime_ns": 9}}}'
    parsed = parse_manifest(good)
    assert isinstance(parsed, Manifest) and parsed.files == {RelPath("a/b.txt"): OnDiskFile(3, 9)}
    assert parsed.status is RunStatus.FINISHED
    assert parse_manifest(good.replace('"version": 2', '"version": true')) == UNRECORDED  # True == 1 in Python
    assert parse_manifest(good.replace('"version": 2', '"version": 2.0')) == UNRECORDED
    assert parse_manifest(good.replace('"size": 3', '"size": true')) == UNRECORDED  # a bool is not a size


def test_an_unknown_run_status_in_a_manifest_is_not_terminal() -> None:
    text = '{"version": 2, "run_id": "r", "status": "WEIRD", "files": {}}'
    parsed = parse_manifest(text)
    assert isinstance(parsed, Manifest) and parsed.status is RunStatus.UNKNOWN and not parsed.status.terminal
    assert [s.terminal for s in (RunStatus.FINISHED, RunStatus.FAILED, RunStatus.KILLED)] == [True] * 3
    assert not any(s.terminal for s in (RunStatus.RUNNING, RunStatus.SCHEDULED, RunStatus.UNKNOWN))
    assert RunStatus.parse("FINISHED") is RunStatus.FINISHED and RunStatus.parse("nope") is RunStatus.UNKNOWN


def test_restricted_to_keeps_only_files_whose_recorded_state_is_on_disk() -> None:
    kept, moved, gone = RelPath("kept"), RelPath("moved"), RelPath("gone")
    manifest = Manifest("r", RunStatus.FINISHED, {kept: OnDiskFile(1, 10), moved: OnDiskFile(1, 10), gone: OnDiskFile(1, 10)})

    restricted = manifest.restricted_to({kept: OnDiskFile(1, 10), moved: OnDiskFile(1, 11)})  # moved: new mtime

    assert restricted.files == {kept: OnDiskFile(1, 10)}


# --- names mlpull writes itself --------------------------------------------------------------------


def test_an_artifact_named_like_the_manifest_is_rejected_unless_excluded(
    store: Store, gateway: RunGateway, tmp_path: Path
) -> None:
    store.client.log_text(store.child, "mine", MANIFEST_NAME)
    store.client.log_text(store.child, "nested is fine", f"sub/{MANIFEST_NAME}")
    out = tmp_path / "out"

    with pytest.raises(ReservedNameError, match=r"--path '!\.mlpull\.json'"):
        _pull(gateway, [store.child], out)
    assert not out.exists()

    _pull(gateway, [store.child], out, _options(f"!{MANIFEST_NAME}"))
    run_dir = out / f"child-{store.child[:8]}"
    assert (run_dir / "c.txt").is_file() and (run_dir / "sub" / MANIFEST_NAME).read_text() == "nested is fine"
    assert json.loads((run_dir / MANIFEST_NAME).read_text())["files"] == _on_disk(
        run_dir, {"c.txt": 7, f"sub/{MANIFEST_NAME}": 14}
    )


def test_an_artifact_named_run_json_collides_only_when_metadata_is_written(
    store: Store, gateway: RunGateway, tmp_path: Path
) -> None:
    store.client.log_text(store.child, "mine", SUMMARY_NAME)
    assert _pull(gateway, [store.child], tmp_path / "plain")  # fine without --what metadata

    with pytest.raises(ReservedNameError, match="run.json"):
        _pull(gateway, [store.child], tmp_path / "both", PullOptions(content=Content.parse(["metrics,artifacts"])))


# --- --what ----------------------------------------------------------------------------------------


def test_content_parses_comma_lists_and_repeats() -> None:
    assert Content.parse(["metrics,tags", "artifacts"]).what == {What.METRICS, What.TAGS, What.ARTIFACTS}
    assert Content.parse(["params"]).what == {What.PARAMS}
    for bad in (["nope"], [""], ["metrics,"]):
        with pytest.raises(ValueError, match="--what"):
            Content.parse(bad)


def test_status_gate_applies_to_artifacts_only() -> None:
    assert Content.parse(["artifacts"]).statuses(False, False) == FINISHED
    assert Content.parse(["metrics,artifacts"]).statuses(False, False) == FINISHED
    assert "RUNNING" in Content.parse(["artifacts"]).statuses(True, False)
    assert {"RUNNING", "FAILED", "FINISHED"} <= Content.parse(["metrics"]).statuses(False, False)


def test_cli_what_metrics_reads_a_running_run_without_flags(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("run", store.live, "--what", "metrics", "-o", str(out))

    assert result.exit_code == 0, result.output
    run_dir = next(out.iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == [SUMMARY_NAME]  # no artifacts, no manifest
    record = json.loads((run_dir / SUMMARY_NAME).read_text())
    assert record["run_id"] == store.live and record["name"] == "live" and record["status"] == "RUNNING"
    assert record["latest"] == {"loss": 0.25}
    assert record["params"] == {"lr": "0.1"} and record["tags"]["phase"] == "train"
    assert record["start"] and record["end"] is None
    assert "Downloaded 1 run(s), 0 file(s)" in result.output


def test_cli_what_accepts_repeats_and_still_gates_artifacts(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    refused = cli("run", store.live, "--what", "params", "--what", "artifacts", "-o", str(out))
    assert refused.exit_code == 1 and "override with -u, -f or -a" in refused.output and not out.exists()

    allowed = cli("run", store.live, "-u", "--what", "tags,artifacts", "-o", str(out))
    assert allowed.exit_code == 0, allowed.output
    run_dir = next(out.iterdir())
    assert _tree(run_dir) == {"l.txt", SUMMARY_NAME} and (run_dir / MANIFEST_NAME).is_file()


def test_cli_default_what_is_artifacts_and_writes_no_run_json(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert cli("run", store.child, "-o", str(out)).exit_code == 0
    assert _tree(out) == {f"child-{store.child[:8]}/c.txt"}


def test_cli_experiment_metadata_covers_every_selected_run_of_any_status(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("experiment", "exp", "--what", "metrics", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.rglob(SUMMARY_NAME)) == [SUMMARY_NAME] * 3  # parent, child, live


def test_cli_dry_run_with_metadata_lists_run_json_and_writes_nothing(store: Store, cli: Cli, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = cli("run", store.live, "--what", "params", "--dry-run", "-o", str(out))

    assert result.exit_code == 0 and SUMMARY_NAME in result.output
    assert not out.exists()


def test_cli_rejects_an_unknown_what(store: Store, cli: Cli) -> None:
    assert cli("run", store.parent, "--what", "logs").exit_code == 2


def test_cli_help_lists_the_new_options(cli: Cli) -> None:
    for command in ("run", "experiment"):
        output = cli(command, "--help").output
        for option in ("--path", "--max-size", "--dry-run", "--what"):
            assert option in output


def test_outcomes_carry_counts_and_bytes(store: Store, gateway: RunGateway, tmp_path: Path) -> None:
    first = _pull(gateway, [store.parent], tmp_path / "out", _options("config/**"))
    parent, child = first

    assert isinstance(parent, Downloaded) and (parent.files, parent.bytes) == (2, 30)
    assert isinstance(child, UpToDate) and (child.files, child.bytes) == (0, 0)  # c.txt is not under config/


def test_listing_entries_are_files_or_directories(store: Store, gateway: RunGateway) -> None:
    root = gateway.list_artifacts(store.parent, None)
    assert {type(e) for e in root} == {ArtifactDir, ArtifactFile}
    assert ArtifactFile(RelPath("top.txt"), KnownSize(5)) in root and ArtifactDir(RelPath("config")) in root
    assert sorted(e.path.value for e in gateway.list_artifacts(store.parent, RelPath("config"))) == [
        "config/a.yaml",
        "config/sub",
    ]


# --- a hostile server cannot run the listing walk away -----------------------------------------------


class ScriptedGateway(MlflowGateway):
    """A real gateway whose artifact listing is entirely scripted: a hostile server."""

    def __init__(self, client: MlflowClient, script: Callable[[RelPath | None], list[ArtifactEntry]]) -> None:
        super().__init__(client)
        self.script = script
        self.listed: list[RelPath | None] = []

    def list_artifacts(self, run_id: str, path: RelPath | None) -> list[ArtifactEntry]:
        self.listed.append(path)
        return self.script(path)


def _below(path: RelPath | None, name: str) -> RelPath:
    return RelPath(name if path is None else f"{path}/{name}")


def _depth(path: RelPath | None) -> int:
    return 0 if path is None else len(path.parts)


def _plan_child(gateway: MlflowGateway, store: Store, out: Path):
    return plan_pull(gateway, resolve_runs(MlflowGateway(store.client), [store.child], out, FINISHED))


def test_a_listing_that_always_has_a_fresh_subdirectory_stops_at_the_depth_limit(store: Store, tmp_path: Path) -> None:
    gateway = ScriptedGateway(store.client, lambda path: [ArtifactDir(_below(path, "d"))])

    with pytest.raises(ListingLimitError, match=f"deeper than {MAX_DEPTH} levels"):
        _plan_child(gateway, store, tmp_path / "out")

    assert len(gateway.listed) == MAX_DEPTH + 1


def test_a_chain_exactly_as_deep_as_the_limit_is_allowed(store: Store, tmp_path: Path) -> None:
    def chain(levels: int) -> Callable[[RelPath | None], list[ArtifactEntry]]:
        def script(path: RelPath | None) -> list[ArtifactEntry]:
            if _depth(path) == levels - 1:
                return [ArtifactFile(_below(path, "f"), KnownSize(1))]
            return [ArtifactDir(_below(path, "d"))]

        return script

    plan = _plan_child(ScriptedGateway(store.client, chain(MAX_DEPTH)), store, tmp_path / "out")
    assert [len(f.path.parts) for f in plan.runs[0].files] == [MAX_DEPTH]

    with pytest.raises(ListingLimitError, match="deeper than"):
        _plan_child(ScriptedGateway(store.client, chain(MAX_DEPTH + 1)), store, tmp_path / "out")


def test_a_directory_listed_twice_per_level_is_walked_once(store: Store, tmp_path: Path) -> None:
    levels = 12

    def doubled(path: RelPath | None) -> list[ArtifactEntry]:
        if _depth(path) == levels:
            return [ArtifactFile(_below(path, "f"), KnownSize(1))]
        directory = ArtifactDir(_below(path, "d"))
        return [directory, directory]

    gateway = ScriptedGateway(store.client, doubled)
    plan = _plan_child(gateway, store, tmp_path / "out")

    assert len(gateway.listed) == levels + 1  # unfixed: 2 ** levels
    assert len(plan.runs[0].files) == 1


def test_a_file_listed_twice_is_one_file(store: Store, tmp_path: Path) -> None:
    twice = ArtifactFile(RelPath("f"), KnownSize(1))
    plan = _plan_child(ScriptedGateway(store.client, lambda path: [twice, twice]), store, tmp_path / "out")
    assert plan.runs[0].files == (twice,)


def test_a_listing_with_too_many_entries_is_refused(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execute, "MAX_ENTRIES", 50)

    def wide(path: RelPath | None) -> list[ArtifactEntry]:
        if path is None:  # ten directories ...
            return [ArtifactDir(RelPath(f"d{i}")) for i in range(10)]
        return [ArtifactFile(_below(path, f"f{i}"), KnownSize(1)) for i in range(10)]  # ... of ten files each

    with pytest.raises(ListingLimitError, match="more than 50 entries"):
        _plan_child(ScriptedGateway(store.client, wide), store, tmp_path / "out")


def test_one_huge_listing_is_refused_too(store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execute, "MAX_ENTRIES", 50)
    files = [ArtifactFile(RelPath(f"f{i}"), KnownSize(1)) for i in range(51)]

    with pytest.raises(ListingLimitError, match="more than 50 entries"):
        _plan_child(ScriptedGateway(store.client, lambda path: files), store, tmp_path / "out")


def test_cli_reports_a_runaway_listing_in_one_line(
    store: Store, cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = ScriptedGateway(store.client, lambda path: [ArtifactDir(_below(path, "d"))])
    monkeypatch.setattr(MlflowGateway, "connect", classmethod(lambda cls, uri: gateway))
    out = tmp_path / "out"

    result = cli("run", store.child, "-o", str(out))

    assert result.exit_code == 1 and "deeper than" in result.output
    assert not out.exists()


# --- symlinks in the output directory are never followed ------------------------------------------------


def _plant(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


@POSIX_ONLY
def test_a_symlinked_directory_inside_a_run_dir_is_not_written_through(
    store: Store, gateway: RunGateway, tmp_path: Path
) -> None:
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    elsewhere.mkdir()
    _plant(_parent_dir(out, store) / "config", elsewhere)

    with pytest.raises(SymlinkError, match="config"):
        _pull(gateway, [store.parent], out)

    assert list(elsewhere.iterdir()) == []


@POSIX_ONLY
@pytest.mark.parametrize("dangling", [False, True], ids=["existing target", "dangling target"])
def test_a_symlinked_file_destination_is_not_written_through(
    store: Store, gateway: RunGateway, tmp_path: Path, dangling: bool
) -> None:
    out, victim = tmp_path / "out", tmp_path / "victim.txt"
    if not dangling:
        victim.write_text("precious")
    _plant(_parent_dir(out, store) / "top.txt", victim)

    with pytest.raises(SymlinkError, match="top.txt"):
        _pull(gateway, [store.parent], out)

    assert victim.read_text() == "precious" if not dangling else not victim.exists()


@POSIX_ONLY
def test_a_symlinked_nested_run_directory_is_not_written_through(
    store: Store, gateway: RunGateway, tmp_path: Path
) -> None:
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    elsewhere.mkdir()
    _plant(_parent_dir(out, store) / f"child-{store.child[:8]}", elsewhere)

    with pytest.raises(SymlinkError, match=f"child-{store.child[:8]}"):
        _pull(gateway, [store.parent], out)

    assert list(elsewhere.iterdir()) == []


@POSIX_ONLY
def test_cli_refuses_a_symlinked_destination_with_exit_1(store: Store, cli: Cli, tmp_path: Path) -> None:
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    elsewhere.mkdir()
    _plant(_parent_dir(out, store) / "model", elsewhere)

    result = cli("run", store.parent, "-o", str(out))

    assert result.exit_code == 1 and "symlink" in result.output and "model" in result.output
    assert list(elsewhere.iterdir()) == []


@POSIX_ONLY
def test_a_symlink_never_counts_as_a_recorded_file(
    store: Store, counting: CountingGateway, tmp_path: Path
) -> None:
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    _pull(counting, [store.child], out)
    run_dir = out / f"child-{store.child[:8]}"
    elsewhere.mkdir()
    (elsewhere / "c.txt").write_text("x" * 7)  # the same size and a recorded name, but outside the run dir
    (run_dir / "c.txt").unlink()
    (run_dir / "c.txt").symlink_to(elsewhere / "c.txt")
    counting.downloaded.clear()

    with pytest.raises(SymlinkError):
        _pull(counting, [store.child], out)

    assert counting.downloaded == []  # it was planned for a fetch, then refused before the download call


# --- sizes: the server's numbers are checked, not trusted -----------------------------------------------


class ResizingGateway(MlflowGateway):
    """A real gateway whose listing reports every file size as ``size_of(real size)``."""

    def __init__(self, client: MlflowClient, size_of: Callable[[FileSize], FileSize]) -> None:
        super().__init__(client)
        self.size_of = size_of

    def list_artifacts(self, run_id: str, path: RelPath | None) -> list[ArtifactEntry]:
        return [
            ArtifactFile(entry.path, self.size_of(entry.size)) if isinstance(entry, ArtifactFile) else entry
            for entry in super().list_artifacts(run_id, path)
        ]


def _unknown(_: FileSize) -> FileSize:
    return UNKNOWN_SIZE


def _one_more(size: FileSize) -> FileSize:
    assert isinstance(size, KnownSize)
    return KnownSize(size.bytes + 1)


def test_listing_sizes_become_validated_sizes_at_the_gateway() -> None:
    assert entry_from_info(FileInfo("a.txt", False, 5)) == ArtifactFile(RelPath("a.txt"), KnownSize(5))
    assert entry_from_info(FileInfo("a.txt", False, 0)) == ArtifactFile(RelPath("a.txt"), KnownSize(0))
    assert entry_from_info(FileInfo("a.txt", False, None)) == ArtifactFile(RelPath("a.txt"), UNKNOWN_SIZE)
    assert entry_from_info(FileInfo("d", True, None)) == ArtifactDir(RelPath("d"))
    with pytest.raises(InvalidSizeError, match=r"a\.txt.*-5"):
        entry_from_info(FileInfo("a.txt", False, -5))


def test_a_negative_listed_size_cannot_defeat_max_size(store: Store, tmp_path: Path) -> None:
    client = ListingClient(tracking_uri=store.client.tracking_uri)
    client.listing = {"": [FileInfo("big.bin", False, -(10**9)), FileInfo("ok.txt", False, 1)]}
    out = tmp_path / "out"

    with pytest.raises(InvalidSizeError):
        pull(MlflowGateway(client), resolve_runs(MlflowGateway(client), [store.child], out), _options(max_bytes=100))

    assert not out.exists()


def test_max_size_refuses_a_plan_with_a_file_of_unknown_size(store: Store, tmp_path: Path) -> None:
    gateway = ResizingGateway(store.client, _unknown)
    out = tmp_path / "out"

    with pytest.raises(UnknownSizeError, match=r"1 file\(s\).*no size.*c\.txt"):
        pull(gateway, resolve_runs(gateway, [store.child], out), _options(max_bytes=10**9))

    assert not out.exists()


def test_without_max_size_unknown_sizes_are_pulled_and_the_manifest_records_the_file_on_disk(
    store: Store, tmp_path: Path
) -> None:
    gateway = ResizingGateway(store.client, _unknown)
    out = tmp_path / "out"

    (outcome,) = pull(gateway, resolve_runs(gateway, [store.child], out), PullOptions())

    run_dir = out / f"child-{store.child[:8]}"
    assert isinstance(outcome, Downloaded) and outcome.files == 1
    assert json.loads((run_dir / MANIFEST_NAME).read_text())["files"] == _on_disk(run_dir, {"c.txt": 7})


def test_cli_max_size_with_an_unknown_size_exits_1_and_says_why(
    store: Store, cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = ResizingGateway(store.client, _unknown)
    monkeypatch.setattr(MlflowGateway, "connect", classmethod(lambda cls, uri: gateway))
    out = tmp_path / "out"

    result = cli("run", store.child, "--max-size", "1G", "-o", str(out))

    assert result.exit_code == 1 and "refused (--max-size)" in result.output and "no size" in result.output
    assert not out.exists()
    assert cli("run", store.child, "-o", str(out)).exit_code == 0  # without the limit it is fine


def test_a_file_whose_size_differs_from_the_listing_is_an_error(store: Store, tmp_path: Path) -> None:
    gateway = ResizingGateway(store.client, _one_more)
    out = tmp_path / "out"

    with pytest.raises(SizeMismatchError, match=r"c\.txt.*listed as 8 bytes.*7 bytes"):
        pull(gateway, resolve_runs(gateway, [store.child], out), PullOptions())

    assert not (out / f"child-{store.child[:8]}" / MANIFEST_NAME).exists()  # nothing vouches for the file


class DirectoryInsteadGateway(MlflowGateway):
    """The server lists a file, the download is a directory (mlflow decides that itself, and recurses)."""

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        path.under(dest_dir).mkdir(parents=True)


def test_a_listed_file_that_downloads_as_a_directory_is_an_error(store: Store, tmp_path: Path) -> None:
    gateway = DirectoryInsteadGateway(store.client)
    out = tmp_path / "out"

    with pytest.raises(NotRegularFileError, match=r"c\.txt"):
        pull(gateway, resolve_runs(gateway, [store.child], out), PullOptions())

    assert not (out / f"child-{store.child[:8]}" / MANIFEST_NAME).exists()


# --- plan-time conflicts: nothing is written when two artifacts cannot both land -----------------------


def _target(store: Store, tmp_path: Path) -> RunTarget:
    return RunTarget(store.client.get_run(store.child), tmp_path / f"child-{store.child[:8]}")


def _files(*names: str) -> list[ArtifactFile]:
    return [ArtifactFile(RelPath(name), KnownSize(1)) for name in names]


def _plan(store: Store, tmp_path: Path, *names: str, content: Content = Content.parse(["artifacts"]),
          children: frozenset[str] = frozenset(), path_filter: PathFilter = PathFilter(), fold_names: bool = True):
    return plan_run(
        _target(store, tmp_path), _files(*names), UNRECORDED, path_filter, content.reserved, children, fold_names
    )


def test_a_file_and_a_directory_of_the_same_name_conflict(store: Store, tmp_path: Path) -> None:
    with pytest.raises(PrefixConflictError, match=r"a.*a/b"):
        _plan(store, tmp_path, "a", "a/b")
    with pytest.raises(PrefixConflictError):
        _plan(store, tmp_path, "x/y/z", "x/y")  # in any order, at any depth
    _plan(store, tmp_path, "a", "ab", "a.b", "b/a")  # a mere string prefix is fine


@pytest.mark.parametrize(
    "name", [MANIFEST_NAME, f"{MANIFEST_NAME}/x", f"{MANIFEST_NAME}/x/y", ".MLPULL.JSON", ".MLPULL.json/x"]
)
def test_the_manifest_name_is_reserved_as_a_first_component_in_any_case(
    store: Store, tmp_path: Path, name: str
) -> None:
    with pytest.raises(ReservedNameError, match=r"--path '!"):
        _plan(store, tmp_path, "ok.txt", name)


@pytest.mark.parametrize("name", [SUMMARY_NAME, f"{SUMMARY_NAME}/x", "RUN.json", "Run.Json/y"])
def test_run_json_is_reserved_only_when_metadata_is_written(store: Store, tmp_path: Path, name: str) -> None:
    _plan(store, tmp_path, name)  # artifacts only: run.json is not written
    with pytest.raises(ReservedNameError, match="--path"):
        _plan(store, tmp_path, name, content=Content.parse(["metrics,artifacts"]))


def test_a_nested_reserved_name_is_fine(store: Store, tmp_path: Path) -> None:
    assert len(_plan(store, tmp_path, f"sub/{MANIFEST_NAME}", "sub/run.json").files) == 2


def test_a_reserved_name_can_still_be_excluded_with_path(store: Store, tmp_path: Path) -> None:
    plan = _plan(store, tmp_path, f"{MANIFEST_NAME}/x", "ok", path_filter=PathFilter.parse([f"!{MANIFEST_NAME}/**"]))
    assert [f.path.value for f in plan.files] == ["ok"]


@pytest.mark.parametrize(
    "names",
    [("A.txt", "a.txt"), ("é.txt", "é.txt"), ("d/x", "D/x"), ("straße", "STRASSE")],
    ids=["case", "normalisation", "directory case", "casefold"],
)
def test_names_that_are_the_same_file_after_case_and_normalisation_conflict(
    store: Store, tmp_path: Path, names: tuple[str, str]
) -> None:
    with pytest.raises(DuplicatePathError, match="after case/Unicode folding"):
        _plan(store, tmp_path, *names)
    _plan(store, tmp_path, names[0], f"other/{names[1]}")


def test_a_file_and_a_directory_conflict_after_folding(store: Store, tmp_path: Path) -> None:
    with pytest.raises(PrefixConflictError):
        _plan(store, tmp_path, "Config", "config/a.yaml")


def test_the_directory_of_a_nested_run_is_reserved_for_that_run(store: Store, tmp_path: Path) -> None:
    child_dir = f"child-{store.child[:8]}"
    for name in (f"{child_dir}/{MANIFEST_NAME}", child_dir.upper() + "/x", child_dir):
        with pytest.raises(ReservedNameError, match="nested run"):
            _plan(store, tmp_path, "ok", name, children=frozenset({child_dir}))
    _plan(store, tmp_path, f"other/{child_dir}/x", children=frozenset({child_dir}))


def test_a_parent_artifact_cannot_land_in_its_nested_runs_directory(
    store: Store, gateway: RunGateway, tmp_path: Path
) -> None:
    store.client.log_text(store.parent, "planted", f"child-{store.child[:8]}/{MANIFEST_NAME}")
    out = tmp_path / "out"

    with pytest.raises(ReservedNameError, match="nested run"):
        _pull(gateway, [store.parent], out)

    assert not out.exists()


def test_a_file_and_a_directory_listed_under_one_name_are_refused_before_any_write(
    store: Store, tmp_path: Path
) -> None:
    def script(path: RelPath | None) -> list[ArtifactEntry]:
        if path is None:
            return [ArtifactFile(RelPath("a"), KnownSize(1)), ArtifactDir(RelPath("a"))]
        return [ArtifactFile(RelPath("a/b"), KnownSize(1))]

    gateway = ScriptedGateway(store.client, script)
    out = tmp_path / "out"

    with pytest.raises(PrefixConflictError):
        pull(gateway, resolve_runs(gateway, [store.child], out), PullOptions())

    assert not out.exists()


# --- names MLflow's runs:/ URI cannot carry are skipped and counted ---------------------------------------

ODD_NAMES = ["a;b.txt", "a%b.txt", "a%2.txt", "a b.txt", "é.txt", "a+b.txt", "sub/a;b.txt"]
UNSUPPORTED_NAMES = ["a#b.txt", "a?b.txt", "a%41b.txt", "a%23b.txt", "d#x/f.txt"]


@pytest.fixture
def odd_run(store: Store) -> str:
    """A FINISHED run holding names MLflow downloads fine and names its runs:/ URI cannot carry."""
    run_id = _make_run(store.client, store.experiment_id, "odd", {n: 3 for n in (*ODD_NAMES, *UNSUPPORTED_NAMES)})
    store.client.set_terminated(run_id, "FINISHED")
    return run_id


def test_names_the_runs_uri_cannot_carry_are_skipped_and_counted_not_fatal(
    store: Store, odd_run: str, gateway: RunGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    (outcome,) = _pull(gateway, [odd_run], out)

    run_dir = out / f"odd-{odd_run[:8]}"
    assert _tree(run_dir) == frozenset(ODD_NAMES)
    assert all((run_dir / name).read_text() == "xxx" for name in ODD_NAMES)
    assert isinstance(outcome, Downloaded) and (outcome.files, outcome.unsupported) == (len(ODD_NAMES), len(UNSUPPORTED_NAMES))
    assert set(json.loads((run_dir / MANIFEST_NAME).read_text())["files"]) == set(ODD_NAMES)


def test_unsupported_names_do_not_count_against_max_size_or_the_manifest(
    store: Store, odd_run: str, gateway: RunGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _pull(gateway, [odd_run], out, _options(max_bytes=3 * len(ODD_NAMES)))  # exactly the supported bytes

    (outcome,) = _pull(gateway, [odd_run], out, _options(max_bytes=1))  # nothing left to fetch
    assert isinstance(outcome, UpToDate) and outcome.unsupported == len(UNSUPPORTED_NAMES)


def test_cli_reports_unsupported_names_on_stderr_and_in_the_jsonl_and_summary(
    store: Store, odd_run: str, cli: Cli, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = cli("run", odd_run, "--format", "jsonl", "-o", str(out))

    assert result.exit_code == 0, result.output
    (record,) = [json.loads(line) for line in result.stdout.splitlines()]
    assert record["unsupported"] == len(UNSUPPORTED_NAMES) and record["files"] == len(ODD_NAMES)
    for name in UNSUPPORTED_NAMES:
        assert f"unsupported name skipped: odd-{odd_run[:8]}/{name}" in result.stderr
    assert f"{len(UNSUPPORTED_NAMES)} unsupported name(s) skipped" in result.stderr

    dry = cli("run", odd_run, "--dry-run", "--format", "jsonl", "-o", str(tmp_path / "dry"))
    assert json.loads(dry.stdout)["unsupported"] == len(UNSUPPORTED_NAMES)
    assert "unsupported name skipped" in dry.stderr


def test_a_run_without_unsupported_names_says_nothing_about_them(store: Store, cli: Cli, tmp_path: Path) -> None:
    result = cli("run", store.child, "-o", str(tmp_path / "out"))
    assert "unsupported name" not in result.output


# --- the manifest vouches for a file only while the file on disk is the one it recorded -------------------


def test_the_manifest_records_size_and_mtime_of_what_landed_on_disk(
    store: Store, gateway: RunGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _pull(gateway, [store.child], out)
    run_dir = out / f"child-{store.child[:8]}"

    entry = json.loads((run_dir / MANIFEST_NAME).read_text())["files"]["c.txt"]

    assert entry == {"size": 7, "mtime_ns": (run_dir / "c.txt").stat().st_mtime_ns}


def test_a_same_size_file_replaced_later_is_fetched_again(
    store: Store, counting: CountingGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    target = out / f"child-{store.child[:8]}" / "c.txt"
    target.write_text("y" * 7)  # the recorded size, other content
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    counting.downloaded.clear()

    _pull(counting, [store.child], out)

    assert counting.downloaded == ["c.txt"] and target.read_text() == "x" * 7


@pytest.mark.parametrize("status", ["RUNNING", "SOMETHING_NEW", "UNKNOWN"])
def test_a_manifest_status_that_is_not_terminal_vouches_for_nothing(
    store: Store, counting: CountingGateway, tmp_path: Path, status: str
) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    manifest = out / f"child-{store.child[:8]}" / MANIFEST_NAME
    manifest.write_text(manifest.read_text().replace('"FINISHED"', f'"{status}"'))
    counting.downloaded.clear()

    _pull(counting, [store.child], out)

    assert counting.downloaded == ["c.txt"]


@POSIX_ONLY
def test_a_fifo_for_a_manifest_means_nothing_is_recorded(
    store: Store, counting: CountingGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    manifest = out / f"child-{store.child[:8]}" / MANIFEST_NAME
    manifest.unlink()
    os.mkfifo(manifest)
    counting.downloaded.clear()

    _pull(counting, [store.child], out)  # must not block on the FIFO

    assert counting.downloaded == ["c.txt"]
    assert manifest.is_file() and json.loads(manifest.read_text())["run_id"] == store.child


@POSIX_ONLY
def test_a_symlink_for_a_manifest_to_a_huge_file_is_not_read(
    store: Store, counting: CountingGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    manifest = out / f"child-{store.child[:8]}" / MANIFEST_NAME
    good = manifest.read_text()
    big = tmp_path / "big.json"
    with big.open("wb") as handle:
        handle.write(good.encode())
        handle.truncate(64 * 1024**2)  # a valid manifest followed by a sparse 64 MiB
    manifest.unlink()
    manifest.symlink_to(big)
    counting.downloaded.clear()

    _pull(counting, [store.child], out)

    assert counting.downloaded == ["c.txt"]
    assert not manifest.is_symlink() and big.stat().st_size == 64 * 1024**2  # replaced, not written through


# --- case and normalisation folding only where the file system folds -----------------------------------------


@pytest.mark.parametrize(("platform", "folds"), [("win32", True), ("darwin", True), ("linux", False), ("freebsd14", False), ("cygwin", False)])
def test_only_the_platforms_whose_file_systems_fold_names_fold_them(platform: str, folds: bool) -> None:
    assert paths.names_fold_on(platform) is folds


def test_names_differing_in_case_are_two_files_where_the_file_system_keeps_case(store: Store, tmp_path: Path) -> None:
    plan = _plan(store, tmp_path, "Model.pkl", "model.pkl", "d/x", "D/x", fold_names=False)
    assert [f.path.value for f in plan.files] == ["D/x", "Model.pkl", "d/x", "model.pkl"]
    _plan(store, tmp_path, "Config", "config/a.yaml", fold_names=False)  # a file and a directory, in two cases


def test_reserved_names_are_exact_where_the_file_system_keeps_case(store: Store, tmp_path: Path) -> None:
    both = Content.parse(["metrics,artifacts"])
    assert len(_plan(store, tmp_path, "RUN.json", ".MLPULL.JSON", content=both, fold_names=False).files) == 2
    for name in (SUMMARY_NAME, MANIFEST_NAME, f"{SUMMARY_NAME}/x"):
        with pytest.raises(ReservedNameError, match=r"--path '!"):
            _plan(store, tmp_path, name, content=both, fold_names=False)
    child_dir = f"child-{store.child[:8]}"
    _plan(store, tmp_path, f"{child_dir.upper()}/x", children=frozenset({child_dir}), fold_names=False)
    with pytest.raises(ReservedNameError, match="nested run"):
        _plan(store, tmp_path, f"{child_dir}/x", children=frozenset({child_dir}), fold_names=False)


def test_unicode_equal_names_still_conflict_where_the_file_system_keeps_case(store: Store, tmp_path: Path) -> None:
    with pytest.raises(DuplicatePathError, match=r"--path '!"):
        _plan(store, tmp_path, "é.txt", "é.txt", fold_names=False)


def test_both_name_conflicts_say_how_to_leave_one_out(store: Store, tmp_path: Path) -> None:
    with pytest.raises(DuplicatePathError, match=r"--path '!a\.txt'"):
        _plan(store, tmp_path, "A.txt", "a.txt")  # the later name is the one to leave out
    with pytest.raises(ReservedNameError, match=r"--path '!RUN\.json'"):
        _plan(store, tmp_path, "RUN.json", content=Content.parse(["metrics,artifacts"]))


def test_a_clash_found_by_folding_can_be_left_out_with_path(store: Store, tmp_path: Path) -> None:
    plan = _plan(store, tmp_path, "A.txt", "a.txt", path_filter=PathFilter.parse(["!a.txt"]))
    assert [f.path.value for f in plan.files] == ["A.txt"]


def test_the_fold_setting_reaches_the_plan_of_a_pull(store: Store, tmp_path: Path) -> None:
    def listing(path: RelPath | None) -> list[ArtifactEntry]:
        return [ArtifactFile(RelPath("Model.pkl"), KnownSize(1)), ArtifactFile(RelPath("model.pkl"), KnownSize(1))]

    gateway = ScriptedGateway(store.client, listing)
    resolution = resolve_runs(gateway, [store.child], tmp_path / "out", FINISHED)

    kept = plan_pull(gateway, resolution, PullOptions(fold_names=False))
    assert len(kept.runs[0].files) == 2
    with pytest.raises(DuplicatePathError):
        plan_pull(gateway, resolution, PullOptions(fold_names=True))


# --- names this platform cannot hold are reported, not fatal -------------------------------------------------


def _unsupported_listing(path: RelPath | None) -> list[ArtifactEntry]:
    assert path is None, f"the unsupported directory must not be listed: {path}"
    return [
        ArtifactFile(RelPath("c.txt"), KnownSize(7)),
        UnsupportedEntry.named("k:3.json"),
        UnsupportedEntry.named("too-long-dir"),
    ]


def test_unsupported_entries_are_skipped_counted_and_never_listed(store: Store, tmp_path: Path) -> None:
    gateway = ScriptedGateway(store.client, _unsupported_listing)
    out = tmp_path / "out"

    first, second = (_pull(gateway, [store.child], out) + _pull(gateway, [store.child], out))

    assert isinstance(first, Downloaded) and (first.files, first.unsupported) == (1, 2)
    assert isinstance(second, UpToDate) and second.unsupported == 2
    assert gateway.listed == [None, None]  # the directory entry pruned its subtree
    assert _tree(out / f"child-{store.child[:8]}") == frozenset({"c.txt"})


def test_unsupported_entries_are_reported_like_other_unsupported_names(store: Store, tmp_path: Path) -> None:
    gateway = ScriptedGateway(store.client, _unsupported_listing)
    plan = plan_pull(gateway, resolve_runs(gateway, [store.child], tmp_path / "out", FINISHED))

    (run,) = plan.runs
    assert plan.unsupported == 2
    assert sorted(str(entry.path.under(run.target.dir).relative_to(tmp_path / "out")) for entry in run.unsupported) == [
        f"child-{store.child[:8]}/k:3.json",
        f"child-{store.child[:8]}/too-long-dir",
    ]


def test_path_can_leave_an_unsupported_entry_out_of_the_count(store: Store, tmp_path: Path) -> None:
    gateway = ScriptedGateway(store.client, _unsupported_listing)
    plan = plan_pull(
        gateway, resolve_runs(gateway, [store.child], tmp_path / "out", FINISHED), _options("!k:3.json", "!too-long-dir")
    )
    assert plan.unsupported == 0 and len(plan.runs[0].files) == 1


def test_a_hostile_name_among_unsupported_ones_still_fails_the_listing(store: Store, tmp_path: Path) -> None:
    client = ListingClient(tracking_uri=store.client.tracking_uri)
    client.listing = {"": [FileInfo("a" * 300, False, 1), FileInfo("../x", False, 1)]}
    with pytest.raises(UnsafePathError):
        download_runs(MlflowGateway(client), [store.parent], tmp_path / "out")


def test_a_real_gateway_listing_with_a_long_name_is_reported_not_fatal(store: Store, tmp_path: Path) -> None:
    client = ListingClient(tracking_uri=store.client.tracking_uri)
    client.listing = {"": [FileInfo("a" * 300, False, 1), FileInfo("d" * 300, True, None)]}
    gateway = MlflowGateway(client)

    plan = plan_pull(gateway, resolve_runs(gateway, [store.child], tmp_path / "out", FINISHED))

    assert plan.unsupported == 2 and plan.fetch_files == 0


# --- progress survives a failure -------------------------------------------------------------------------------


class FlakyGateway(CountingGateway):
    """Fails the download of one named file, once."""

    def __init__(self, client: MlflowClient, fail_on: str) -> None:
        super().__init__(client)
        self.fail_on = fail_on

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        if path.value == self.fail_on:
            raise ConnectionError("the server went away")
        super().download_artifact(run_id, path, dest_dir)


LAST_FILE = sorted(SIZES)[-1]


def test_a_failure_on_the_last_file_keeps_the_manifest_of_the_files_before_it(
    store: Store, counting: CountingGateway, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    flaky = FlakyGateway(store.client, LAST_FILE)

    with pytest.raises(ConnectionError, match="went away"):
        _pull(flaky, [store.parent], out, jobs=1)

    recorded = json.loads((_parent_dir(out, store) / MANIFEST_NAME).read_text())["files"]
    assert set(recorded) == set(SIZES) - {LAST_FILE}

    _pull(counting, [store.parent], out)

    assert [name for name in counting.downloaded if name != "c.txt"] == [LAST_FILE]  # only what was missing
    assert _tree(_parent_dir(out, store)) == EVERYTHING_PULLED | {"child-" + store.child[:8] + "/c.txt"}


def test_a_failing_manifest_write_does_not_hide_the_download_error(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, text: str) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(execute, "write_atomically", refuse)

    with pytest.raises(ConnectionError, match="went away"):
        _pull(FlakyGateway(store.client, LAST_FILE), [store.parent], tmp_path / "out", jobs=1)


def test_a_failure_before_any_file_landed_writes_no_manifest(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(ConnectionError):
        _pull(FlakyGateway(store.client, sorted(SIZES)[0]), [store.parent], out, jobs=1)
    assert not (_parent_dir(out, store) / MANIFEST_NAME).exists()


# --- an interrupt stops every worker after the file it is fetching --------------------------------------------

INTERRUPT_TIMEOUT_S = 30  # a hung test fails after this instead of hanging the suite


class InFlightGateway(CountingGateway):
    """Holds the first download of every healthy worker until the pull is told to stop, and says when all are in flight.

    The run ``failing`` (if any) fails its first download as soon as every healthy worker is in flight.
    """

    def __init__(self, client: MlflowClient, workers: int, stop_requested: threading.Event, failing: str | None = None) -> None:
        super().__init__(client)
        self.all_in_flight = threading.Event()
        self._workers = workers
        self._stop_requested = stop_requested
        self._failing = failing
        self._held: set[str] = set()
        self._lock = threading.Lock()

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        if run_id == self._failing:
            assert self.all_in_flight.wait(INTERRUPT_TIMEOUT_S), "the other workers never started"
            raise ConnectionError("the server went away")
        with self._lock:
            first = run_id not in self._held
            self._held.add(run_id)
            if len(self._held) == self._workers:
                self.all_in_flight.set()
        super().download_artifact(run_id, path, dest_dir)
        if first:
            assert self._stop_requested.wait(INTERRUPT_TIMEOUT_S), "the pull was never told to stop"


@pytest.fixture
def stop_requested(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Set the moment the pull raises its stop signal."""
    told = threading.Event()
    real_request = execute.StopSignal.request

    def request_and_tell(signal_: execute.StopSignal) -> None:
        real_request(signal_)
        told.set()

    monkeypatch.setattr(execute.StopSignal, "request", request_and_tell)
    return told


def _two_finished_runs(store: Store) -> tuple[str, str]:
    runs = tuple(_make_run(store.client, store.experiment_id, name, SIZES) for name in ("first", "second"))
    for run_id in runs:
        store.client.set_terminated(run_id, "FINISHED")
    return runs


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs a signal aimed at the main thread")
def test_an_interrupt_lets_each_worker_finish_its_file_and_keeps_what_landed(
    store: Store, tmp_path: Path, stop_requested: threading.Event
) -> None:
    out = tmp_path / "out"
    runs = _two_finished_runs(store)
    gateway = InFlightGateway(store.client, workers=2, stop_requested=stop_requested)
    plan = plan_pull(gateway, resolve_runs(gateway, list(runs), out, FINISHED), PullOptions())
    main = threading.get_ident()

    def interrupt_when_both_workers_are_downloading() -> None:
        if gateway.all_in_flight.wait(INTERRUPT_TIMEOUT_S):
            signal.pthread_kill(main, signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_when_both_workers_are_downloading)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            execute_plan(gateway, plan, jobs=2)
    finally:
        interrupter.join()

    first_file = sorted(SIZES)[0]
    assert gateway.downloaded == [first_file, first_file]  # one per worker, none after the interrupt
    for run_id, name in zip(runs, ("first", "second")):
        manifest = json.loads((out / f"{name}-{run_id[:8]}" / MANIFEST_NAME).read_text())
        assert manifest["files"].keys() == {first_file}


def test_a_failing_run_stops_the_other_workers_after_their_current_file(
    store: Store, tmp_path: Path, stop_requested: threading.Event
) -> None:
    out = tmp_path / "out"
    failing, healthy = _two_finished_runs(store)
    gateway = InFlightGateway(store.client, workers=1, stop_requested=stop_requested, failing=failing)
    plan = plan_pull(gateway, resolve_runs(gateway, [failing, healthy], out, FINISHED), PullOptions())

    with pytest.raises(ConnectionError):
        execute_plan(gateway, plan, jobs=2)

    assert gateway.downloaded == [sorted(SIZES)[0]]  # the healthy run's file in flight, no more


# --- a running run grows between listing and download ----------------------------------------------------------


class GrowingGateway(CountingGateway):
    """The run appends to its artifact after the listing: more bytes land than were listed."""

    def download_artifact(self, run_id: str, path: RelPath, dest_dir: Path) -> None:
        super().download_artifact(run_id, path, dest_dir)
        with path.under(dest_dir).open("a") as handle:
            handle.write("more")


RUNNING_ALLOWED = allowed_statuses(unfinished=True)


def test_a_running_run_that_grew_is_a_warning_not_a_failure(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "out"
    gateway = GrowingGateway(store.client)

    outcomes = _pull(gateway, [store.live], out, allowed=RUNNING_ALLOWED)

    (outcome,) = outcomes
    assert isinstance(outcome, Downloaded) and outcome.changed_during_pull == 1
    assert ", 1 file(s) changed while pulling" in render_outcomes(outcomes, out)
    assert (out / f"live-{store.live[:8]}" / "l.txt").read_text() == "xxxmore"
    manifest = json.loads((out / f"live-{store.live[:8]}" / MANIFEST_NAME).read_text())
    assert manifest["files"] == {}  # nothing vouches for the file


def test_a_finished_run_that_changed_size_is_still_an_error(store: Store, tmp_path: Path) -> None:
    with pytest.raises(SizeMismatchError, match=r"c\.txt"):
        _pull(GrowingGateway(store.client), [store.child], tmp_path / "out")


def test_outcomes_without_changes_report_none(store: Store, gateway: RunGateway, tmp_path: Path) -> None:
    (outcome,) = _pull(gateway, [store.live], tmp_path / "out", allowed=RUNNING_ALLOWED)
    assert isinstance(outcome, Downloaded) and outcome.changed_during_pull == 0


# --- stores that report no size ----------------------------------------------------------------------------------


class UnsizedGateway(CountingGateway):
    def list_artifacts(self, run_id: str, path: RelPath | None):
        return [
            ArtifactFile(entry.path, UNKNOWN_SIZE) if isinstance(entry, ArtifactFile) else entry
            for entry in super().list_artifacts(run_id, path)
        ]


def test_unknown_listed_sizes_do_not_refetch_an_intact_file_of_a_finished_run(store: Store, tmp_path: Path) -> None:
    gateway = UnsizedGateway(store.client)
    out = tmp_path / "out"
    _pull(gateway, [store.child], out)
    gateway.downloaded.clear()

    (outcome,) = _pull(gateway, [store.child], out)

    assert gateway.downloaded == [] and isinstance(outcome, UpToDate) and outcome.files == 1


def test_an_unknown_size_file_that_changed_on_disk_is_fetched_again(store: Store, tmp_path: Path) -> None:
    gateway = UnsizedGateway(store.client)
    out = tmp_path / "out"
    _pull(gateway, [store.child], out)
    target = out / f"child-{store.child[:8]}" / "c.txt"
    target.write_text("y" * 7)
    os.utime(target, ns=(0, target.stat().st_mtime_ns + 10**9))
    gateway.downloaded.clear()

    _pull(gateway, [store.child], out)

    assert gateway.downloaded == ["c.txt"]


def test_an_unknown_size_is_not_trusted_for_a_run_that_is_not_finished(store: Store, tmp_path: Path) -> None:
    live = RunTarget(store.client.get_run(store.live), tmp_path / "live")
    finished = RunTarget(store.client.get_run(store.child), tmp_path / "child")
    listing = [ArtifactFile(RelPath("f"), UNKNOWN_SIZE)]

    def plan(target: RunTarget):
        recorded = Manifest(target.run.info.run_id, RunStatus.FINISHED, {RelPath("f"): OnDiskFile(3, 9)})
        return plan_run(target, listing, recorded, PathFilter(), Content.parse(["artifacts"]).reserved)

    assert [f.path.value for f in plan(finished).up_to_date] == ["f"]
    assert [f.path.value for f in plan(live).files] == ["f"]


def test_max_size_still_refuses_unknown_sizes_that_must_be_fetched(
    store: Store, tmp_path: Path
) -> None:
    gateway = UnsizedGateway(store.client)
    with pytest.raises(UnknownSizeError):
        _pull(gateway, [store.child], tmp_path / "out", _options(max_bytes=10**6))


# --- a run directory that is a symlink -------------------------------------------------------------------------------


@POSIX_ONLY
def test_a_symlinked_run_directory_is_not_up_to_date(store: Store, counting: CountingGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(counting, [store.child], out)
    run_dir = out / f"child-{store.child[:8]}"
    elsewhere = tmp_path / "elsewhere"
    run_dir.rename(elsewhere)
    run_dir.symlink_to(elsewhere)  # the old manifest and files are reachable through the link
    counting.downloaded.clear()
    resolution = resolve_runs(counting, [store.child], out, FINISHED)

    plan = plan_pull(counting, resolution)
    assert [f.path.value for f in plan.runs[0].files] == ["c.txt"] and plan.runs[0].up_to_date == ()

    with pytest.raises(SymlinkError):
        execute_plan(counting, plan)

    assert counting.downloaded == []


# --- a crashed write leaves a temporary file behind ------------------------------------------------------------------


def test_a_pull_removes_old_leftovers_of_its_own_atomic_writes(store: Store, gateway: RunGateway, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _pull(gateway, [store.child], out)
    run_dir = out / f"child-{store.child[:8]}"
    stale = run_dir / temp_name(MANIFEST_NAME)
    stale.write_text("{")
    old = time.time() - 2 * STALE_TEMP_SECONDS
    os.utime(stale, (old, old))
    (run_dir / "c.txt").unlink()  # something to fetch, so the run executes

    _pull(gateway, [store.child], out)

    assert not stale.exists() and (run_dir / "c.txt").exists()
