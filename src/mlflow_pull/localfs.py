"""The output directory, guarded: no symlinks followed, nothing trusted. Effects: lstat, mkdir, read, write.

The output directory may hold files from earlier or hostile pulls, so every directory and file
below the pull root is ``lstat``-ed before mlpull creates it, downloads into it or believes it.
The root itself and everything above it are the user's choice and may be symlinks.
A process that can swap a directory for a symlink between the check and the download is not stopped.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import secrets
import stat
import time
from pathlib import Path

from mlflow_pull.paths import RelPath, UnsafePathError, require_within
from mlflow_pull.plan import OnDiskFile
from mlflow_pull.ports import ArtifactFile, KnownSize

MAX_MANIFEST_BYTES = 16 * 1024**2
# Windows has neither flag; without them a symlink or FIFO is still caught by the ``fstat`` that follows.
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_ENCODING = "utf-8"
_TEMP_PREFIX = ".mlpull-"
_TEMP_SUFFIX = ".tmp"
_TEMP_TOKEN_BYTES = 8
# Exactly the names ``temp_name`` makes: a user artifact such as ``.mlpull-notes.tmp`` (``copy2`` keeps old mtimes) must survive.
_TEMP_NAME = re.compile(
    rf"{re.escape(_TEMP_PREFIX)}.+\.[0-9a-f]{{{2 * _TEMP_TOKEN_BYTES}}}{re.escape(_TEMP_SUFFIX)}", re.DOTALL
)
_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_NEW_FILE_MODE = 0o666  # the umask trims it, as it does for downloaded artifacts
STALE_TEMP_SECONDS = 3600  # a write takes milliseconds: an hour-old temporary file is from a crash
# Some NFS, SMB and FUSE mounts cannot fsync a directory; nothing is wrong with the write then.
_UNSYNCABLE_DIRECTORY = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.ENOSYS})


class SymlinkError(UnsafePathError):
    """A destination is, or lies below, a symlink; mlpull never writes through one."""


class NotRegularFileError(UnsafePathError):
    """A path that must hold a regular file holds something else."""


class SizeMismatchError(ValueError):
    """A downloaded file is not the size the server listed."""


def _kind(path: Path) -> int | None:
    """The ``st_mode`` of ``path`` itself (a symlink is not followed), or ``None`` when nothing is there."""
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return None


def _refuse_symlink(path: Path, mode: int) -> None:
    if stat.S_ISLNK(mode):
        raise SymlinkError(f"{path} is a symlink; mlpull does not write through symlinks")


def is_plain_directory(root: Path, directory: Path) -> bool:
    """Whether every component of ``directory`` below ``root`` is a real directory: no symlink, nothing missing.

    The root and what lies above it are the user's choice and may be links.
    """
    base = Path(os.path.abspath(root))
    current = base
    try:
        parts = Path(os.path.abspath(directory)).relative_to(base).parts
    except ValueError:
        return False
    for part in parts:
        current = current / part
        mode = _kind(current)
        if mode is None or not stat.S_ISDIR(mode):  # lstat: a link to a directory is S_ISLNK, not S_ISDIR
            return False
    return True


def ensure_directory(root: Path, directory: Path) -> None:
    """Create ``directory`` below ``root``; every component below ``root`` must be a real directory."""
    require_within(root, directory)
    base = Path(os.path.abspath(root))
    base.mkdir(parents=True, exist_ok=True)
    current = base
    for part in Path(os.path.abspath(directory)).relative_to(base).parts:
        current = current / part
        mode = _kind(current)
        if mode is None:
            try:
                os.mkdir(current)
                continue
            except FileExistsError:  # another worker made it first: judge what is there now
                mode = _kind(current)
        if mode is None or stat.S_ISLNK(mode):
            raise SymlinkError(f"{current} is a symlink; mlpull does not write through symlinks")
        if not stat.S_ISDIR(mode):
            raise UnsafePathError(f"{current} is not a directory")


def prepare_destination(root: Path, destination: Path) -> None:
    """Make the directories above ``destination`` and refuse a destination that is not a plain file or absent."""
    ensure_directory(root, destination.parent)
    mode = _kind(destination)
    if mode is None:
        return
    _refuse_symlink(destination, mode)
    if not stat.S_ISREG(mode):
        raise NotRegularFileError(f"{destination} exists and is not a regular file")


def stat_file(root: Path, path: RelPath) -> OnDiskFile | None:
    """Size and mtime of the regular file at ``root/path``; ``None`` when absent, odd or behind a symlink."""
    current = root
    *directories, name = path.parts
    try:
        for part in directories:
            current = current / part
            if not stat.S_ISDIR(os.lstat(current).st_mode):
                return None
        found = os.lstat(current / name)
    except OSError:
        return None
    if not stat.S_ISREG(found.st_mode):
        return None
    return OnDiskFile(found.st_size, found.st_mtime_ns)


def verified_download(root: Path, file: ArtifactFile) -> OnDiskFile:
    """What landed at ``root/file.path``: a regular file of the listed size, else an error."""
    found = stat_file(root, file.path)
    if found is None:
        raise NotRegularFileError(
            f"{file.path.under(root)} is not a regular file after its download "
            "(the server listed a file, MLflow wrote something else)"
        )
    if isinstance(file.size, KnownSize) and file.size.bytes != found.size:
        raise SizeMismatchError(
            f"{file.path}: listed as {file.size.bytes} bytes but {found.size} bytes were written"
        )
    return found


def read_small_text(path: Path) -> str | None:
    """The UTF-8 text of a regular file of at most ``MAX_MANIFEST_BYTES``; ``None`` for anything else.

    Never follows a symlink, never blocks on a FIFO, never reads more than the limit.
    """
    try:
        descriptor = os.open(path, _READ_FLAGS)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    if len(data) > MAX_MANIFEST_BYTES:
        return None
    try:
        return data.decode(_ENCODING)
    except UnicodeDecodeError:
        return None


def _sync_directory(directory: Path) -> None:
    """Make the rename durable; Windows cannot open a directory, so there it is skipped."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as err:
        if err.errno not in _UNSYNCABLE_DIRECTORY:
            raise
    finally:
        os.close(descriptor)


def _is_stale_temp_file(entry: os.DirEntry[str], now: float) -> bool:
    """One of our own leftovers: our name pattern, a regular file (not a link), untouched for a long time."""
    if _TEMP_NAME.fullmatch(entry.name) is None:
        return False
    try:
        found = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(found.st_mode) and now - found.st_mtime > STALE_TEMP_SECONDS


def remove_stale_temp_files(directory: Path) -> None:
    """Delete the temporary files an earlier crashed write left directly in ``directory``; best effort."""
    now = time.time()
    try:
        with os.scandir(directory) as entries:
            stale = [Path(entry.path) for entry in entries if _is_stale_temp_file(entry, now)]
    except OSError:
        return
    for path in stale:
        with contextlib.suppress(OSError):
            path.unlink()


def temp_name(final_name: str) -> str:
    return f"{_TEMP_PREFIX}{final_name}.{secrets.token_hex(_TEMP_TOKEN_BYTES)}{_TEMP_SUFFIX}"


def write_atomically(path: Path, text: str) -> None:
    """Never leave a half-written file under the final name, and never write through a planted name.

    The temporary file has a random name opened with ``O_EXCL`` and ``O_NOFOLLOW``, so a symlink at
    a guessable name is ignored and two processes writing the same file cannot collide. Its mode
    follows the umask like every downloaded file, so readers of the output directory can read it.
    """
    name = path.parent / temp_name(path.name)
    descriptor = os.open(name, _TEMP_FLAGS, _NEW_FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding=_ENCODING) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise
    _sync_directory(path.parent)
