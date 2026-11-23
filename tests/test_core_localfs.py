"""The guarded local file system: file modes, directory syncs, stale temporary files, symlinked run directories."""

from __future__ import annotations

import errno
import os
import stat
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mlflow_pull.localfs import (
    STALE_TEMP_SECONDS,
    is_plain_directory,
    remove_stale_temp_files,
    temp_name,
    write_atomically,
)

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX modes, umask and symlinks")


@pytest.fixture
def umask() -> Iterator[Callable[[int], None]]:
    """Set the process umask for one test and restore it after."""
    previous = os.umask(0o022)

    def set_mask(mask: int) -> None:
        os.umask(mask)

    yield set_mask
    os.umask(previous)


# --- modes follow the umask, as downloaded artifacts do ------------------------------------------------------


@POSIX_ONLY
@pytest.mark.parametrize(("mask", "mode"), [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_an_atomic_write_honours_the_umask(tmp_path: Path, umask, mask: int, mode: int) -> None:
    umask(mask)
    write_atomically(tmp_path / "run.json", "x")
    assert stat.S_IMODE((tmp_path / "run.json").stat().st_mode) == mode


# --- the directory sync tolerates mounts that cannot do it ---------------------------------------------------


def _fsync_failing_on_directories(code: int):
    real = os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        real(fd)

    return fsync


@POSIX_ONLY
@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.ENOSYS])
def test_a_directory_that_cannot_be_synced_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    monkeypatch.setattr(os, "fsync", _fsync_failing_on_directories(code))
    write_atomically(tmp_path / "m.json", "x")
    assert (tmp_path / "m.json").read_text() == "x"


@POSIX_ONLY
def test_any_other_directory_sync_error_is_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "fsync", _fsync_failing_on_directories(errno.EIO))
    with pytest.raises(OSError, match="Input/output error"):
        write_atomically(tmp_path / "m.json", "x")


# --- leftovers of a crashed write ------------------------------------------------------------------------


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then), follow_symlinks=False)


@POSIX_ONLY
def test_only_our_old_regular_temporary_files_are_removed(tmp_path: Path) -> None:
    old = STALE_TEMP_SECONDS * 2
    ours = tmp_path / temp_name(".mlpull.json")
    fresh = tmp_path / temp_name("run.json")
    foreign = tmp_path / f".other-{temp_name('run.json')[len('.mlpull-'):]}"
    wrong_end = tmp_path / temp_name("run.json").removesuffix(".tmp")
    precious = tmp_path / "precious"
    link = tmp_path / ".mlpull-link.tmp"
    directory = tmp_path / ".mlpull-dir.tmp"
    for path in (ours, fresh, foreign, wrong_end, precious):
        path.write_text("x")
    link.symlink_to(precious)
    directory.mkdir()
    for path in (ours, foreign, wrong_end, precious, link, directory):
        _age(path, old)

    remove_stale_temp_files(tmp_path)

    assert not ours.exists()
    for kept in (fresh, foreign, wrong_end, precious, directory):
        assert kept.exists()
    assert link.is_symlink() and precious.read_text() == "x"


@POSIX_ONLY
@pytest.mark.parametrize(
    "lookalike", [".mlpull-notes.tmp", ".mlpull-x.ab12cd.tmp", ".mlpull-x.0123456789ABCDEF.tmp", ".mlpull-.0123456789abcdef.tmp"]
)
def test_a_user_artifact_that_only_looks_like_our_temporary_file_survives(tmp_path: Path, lookalike: str) -> None:
    artifact = tmp_path / lookalike
    artifact.write_text("mine")
    _age(artifact, STALE_TEMP_SECONDS * 2)  # copy2 keeps the old mtime of a downloaded file

    remove_stale_temp_files(tmp_path)

    assert artifact.read_text() == "mine"


@POSIX_ONLY
def test_every_name_our_writes_use_is_recognised_as_ours(tmp_path: Path) -> None:
    leftovers = [tmp_path / temp_name(name) for name in (".mlpull.json", "run.json")]
    for path in leftovers:
        path.write_text("{")
        _age(path, STALE_TEMP_SECONDS * 2)

    remove_stale_temp_files(tmp_path)

    assert not any(path.exists() for path in leftovers)


def test_removing_stale_files_from_a_missing_directory_is_nothing(tmp_path: Path) -> None:
    remove_stale_temp_files(tmp_path / "missing")


# --- the run directory itself must not be a link -------------------------------------------------------------


@POSIX_ONLY
def test_a_plain_directory_chain_below_the_root_is_plain(tmp_path: Path) -> None:
    (tmp_path / "root" / "a" / "b").mkdir(parents=True)
    assert is_plain_directory(tmp_path / "root", tmp_path / "root" / "a" / "b")
    assert is_plain_directory(tmp_path / "root", tmp_path / "root")


@POSIX_ONLY
def test_a_symlink_or_a_gap_in_the_chain_is_not_plain(tmp_path: Path) -> None:
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    (root / "a").mkdir(parents=True)
    elsewhere.mkdir()
    (root / "link").symlink_to(elsewhere)
    (root / "a" / "inner").symlink_to(elsewhere)
    (root / "file").write_text("x")

    assert not is_plain_directory(root, root / "link")
    assert not is_plain_directory(root, root / "a" / "inner")
    assert not is_plain_directory(root, root / "missing")
    assert not is_plain_directory(root, root / "file")


@POSIX_ONLY
def test_the_root_itself_may_be_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / "run").mkdir(parents=True)
    (tmp_path / "root").symlink_to(real)
    assert is_plain_directory(tmp_path / "root", tmp_path / "root" / "run")
