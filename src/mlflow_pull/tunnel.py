"""SSH tunnel to a tracking server. Effects: subprocess, sockets, time.

``SshHost`` and ``RemoteAddr`` are validated on construction, so a value that
ssh would read as an option (``-oProxyCommand=...``) never reaches its argv.
"""

from __future__ import annotations

import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO

from mlflow_pull.render import one_line

SIGTERM_EXIT_CODE = 128 + signal.SIGTERM
TUNNEL_TIMEOUT_S = 15.0
MAX_ERROR_CHARS = 2000
_POLL_S = 0.2
_LOOPBACK = "127.0.0.1"
_MAX_PORT = 65535
_HOST_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*")  # alias, host or user@host
_REMOTE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_.-]*|\[[0-9A-Fa-f:.]+\]):(\d{1,5})")


@dataclass(frozen=True, slots=True)
class SshHost:
    """What ``ssh`` accepts as destination: an ``~/.ssh/config`` alias, host or ``user@host``."""

    name: str

    def __post_init__(self) -> None:
        if not _HOST_NAME.fullmatch(self.name):
            raise ValueError(f"not a valid ssh host: {self.name!r}")


@dataclass(frozen=True, slots=True)
class RemoteAddr:
    """Server address as seen from the SSH host."""

    host: str
    port: int

    def __post_init__(self) -> None:
        if not _REMOTE.fullmatch(f"{self.host}:{self.port}") or not 0 < self.port <= _MAX_PORT:
            raise ValueError(f"not a valid HOST:PORT: {self}")

    @classmethod
    def parse(cls, text: str) -> RemoteAddr:
        match = _REMOTE.fullmatch(text)
        if not match:
            raise ValueError(f"expected HOST:PORT, got {text!r}")
        return cls(match[1], int(match[2]))

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


DEFAULT_REMOTE = RemoteAddr("localhost", 5000)


def _free_port() -> int:
    # shortcut: the port may be taken between close and ssh's bind; ssh then fails
    # fast (ExitOnForwardFailure) and the error is reported. Retry loop if it bites.
    with socket.socket() as sock:
        sock.bind((_LOOPBACK, 0))
        return sock.getsockname()[1]


def ssh_command(host: SshHost, remote: RemoteAddr, local_port: int, compress: bool = False) -> list[str]:
    """``ssh`` argv forwarding ``local_port`` to ``remote`` as reached from ``host``."""
    return [
        "ssh", "-N",
        *(["-C"] if compress else []),
        "-o", "ExitOnForwardFailure=yes",
        # Fail instead of waiting on a password prompt nobody sees.
        "-o", "BatchMode=yes",
        # Never join a ControlMaster from ~/.ssh/config: the master would keep the
        # forward open after this command exits.
        "-o", "ControlMaster=no",
        "-o", "ControlPath=none",
        "-L", f"{_LOOPBACK}:{local_port}:{remote}",
        "--", host.name,
    ]


def _failure(host: SshHost, errors: IO[str]) -> SystemExit:
    errors.seek(0)
    # ssh relays what the remote end says before authentication, so it is as untrusted as a server name.
    return SystemExit(f"ssh to {host.name} failed: {one_line(errors.read().strip()[-MAX_ERROR_CHARS:])}")


@contextmanager
def _sigterm_closes_the_tunnel() -> Iterator[None]:
    """Turn SIGTERM into ``SystemExit`` so the ``finally`` that stops ssh runs.

    Without it a plain ``kill`` ends Python at once and leaves ``ssh -N`` forwarding an
    unauthenticated loopback port to the server. SIGKILL still cannot be caught.
    Signal handlers can only be set from the main thread, and Windows has no SIGTERM to catch.
    """
    if sys.platform == "win32" or threading.current_thread() is not threading.main_thread():
        yield
        return

    def exit_now(_signum: int, _frame: object) -> None:
        raise SystemExit(SIGTERM_EXIT_CODE)

    previous = signal.signal(signal.SIGTERM, exit_now)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def ssh_tunnel(
    host: SshHost,
    remote: RemoteAddr = DEFAULT_REMOTE,
    compress: bool = False,
    timeout: float = TUNNEL_TIMEOUT_S,
) -> Iterator[str]:
    """Open an SSH tunnel to ``remote`` via ``host``; yield its local HTTP URI."""
    port = _free_port()
    # A file, not a pipe: nothing reads ssh's stderr while the tunnel runs, and a full
    # pipe would block ssh mid-download.
    with _sigterm_closes_the_tunnel(), tempfile.TemporaryFile("w+", errors="replace") as errors:  # localized ssh messages may not be UTF-8
        proc = subprocess.Popen(
            ssh_command(host, remote, port, compress),
            stdin=subprocess.DEVNULL,
            stderr=errors,
            text=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                if proc.poll() is not None:
                    raise _failure(host, errors)
                try:
                    socket.create_connection((_LOOPBACK, port), _POLL_S).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        errors.seek(0)
                        detail = one_line(errors.read().strip()[-MAX_ERROR_CHARS:])
                        raise SystemExit(
                            f"ssh tunnel to {host.name} not ready after {timeout:g} s"
                            + (f": {detail}" if detail else "")
                        )
                    time.sleep(_POLL_S)
            yield f"http://{_LOOPBACK}:{port}"
        finally:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
