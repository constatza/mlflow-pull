"""Download the artifacts of an MLflow run, its nested runs, or a whole experiment.

The tracking server comes from ``MLFLOW_TRACKING_URI``, like the mlflow CLI.
With ``--ssh HOST`` (or ``MLPULL_SSH``) an SSH tunnel to a server that listens
only on HOST is opened for the duration of the download.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

PARENT_RUN_TAG = "mlflow.parentRunId"
DEFAULT_OUT = Path("mlflow-downloads")
SEARCH_PAGE_SIZE = 50_000  # MLflow's server-side maximum per search call
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')
DEFAULT_REMOTE = "localhost:5000"  # server address as seen from the SSH host
TUNNEL_TIMEOUT_S = 15.0
_POLL_S = 0.2
_LOOPBACK = "127.0.0.1"


def _dir_name(run: Run) -> str:
    name = _UNSAFE_PATH_CHARS.sub("_", run.info.run_name or "run")
    return f"{name}-{run.info.run_id[:8]}"


def _children_by_parent(client: MlflowClient, experiment_id: str) -> dict[str | None, list[Run]]:
    """One search for the whole experiment; run -> children map built in memory."""
    runs = client.search_runs([experiment_id], max_results=SEARCH_PAGE_SIZE)
    children: dict[str | None, list[Run]] = defaultdict(list)
    for run in runs:
        children[run.data.tags.get(PARENT_RUN_TAG)].append(run)
    return children


def _download(
    client: MlflowClient, run: Run, children: dict[str | None, list[Run]], parent_dir: Path
) -> int:
    """Download ``run`` and its descendants under ``parent_dir``; return the run count."""
    run_dir = parent_dir / _dir_name(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(run.info.run_id, "", str(run_dir))
    return 1 + sum(
        _download(client, child, children, run_dir) for child in children[run.info.run_id]
    )


def download_run_tree(client: MlflowClient, run_id: str, out: Path) -> int:
    """Download a run and all its nested runs; return how many runs were downloaded."""
    run = client.get_run(run_id)
    children = _children_by_parent(client, run.info.experiment_id)
    return _download(client, run, children, out)


def download_experiment(client: MlflowClient, name_or_id: str, out: Path) -> int:
    """Download every top-level run of an experiment with its nested runs."""
    experiment = client.get_experiment_by_name(name_or_id) or client.get_experiment(name_or_id)
    children = _children_by_parent(client, experiment.experiment_id)
    experiment_dir = out / _UNSAFE_PATH_CHARS.sub("_", experiment.name)
    return sum(_download(client, run, children, experiment_dir) for run in children[None])


def _free_port() -> int:
    # shortcut: the port may be taken between close and ssh's bind; ssh then fails
    # fast (ExitOnForwardFailure) and the error is reported. Retry loop if it bites.
    with socket.socket() as sock:
        sock.bind((_LOOPBACK, 0))
        return sock.getsockname()[1]


def ssh_command(host: str, remote: str, local_port: int) -> list[str]:
    """``ssh`` argv forwarding ``local_port`` to ``remote`` as reached from ``host``."""
    return [
        "ssh", "-N", "-C",
        "-o", "ExitOnForwardFailure=yes",
        # Fail instead of waiting on a password prompt nobody sees.
        "-o", "BatchMode=yes",
        "-L", f"{_LOOPBACK}:{local_port}:{remote}",
        host,
    ]


@contextmanager
def ssh_tunnel(
    host: str, remote: str = DEFAULT_REMOTE, timeout: float = TUNNEL_TIMEOUT_S
) -> Iterator[str]:
    """Open an SSH tunnel to ``remote`` via ``host``; yield its local HTTP URI."""
    port = _free_port()
    proc = subprocess.Popen(
        ssh_command(host, remote, port), stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"ssh to {host} failed: {proc.stderr.read().strip()}")
            try:
                socket.create_connection((_LOOPBACK, port), _POLL_S).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"ssh tunnel to {host} not ready after {timeout:g} s")
                time.sleep(_POLL_S)
        yield f"http://{_LOOPBACK}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlpull", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="destination directory")
    # Accepted before or after the subcommand; SUPPRESS keeps the subcommand's
    # parser from overwriting a value given before it.
    for p, defaults in (
        (parser, {"ssh": os.environ.get("MLPULL_SSH"),
                  "remote": os.environ.get("MLPULL_REMOTE", DEFAULT_REMOTE)}),
        (common, {"ssh": argparse.SUPPRESS, "remote": argparse.SUPPRESS}),
    ):
        p.add_argument("--ssh", metavar="HOST", default=defaults["ssh"],
                       help="tunnel to the server through this SSH host (env MLPULL_SSH)")
        p.add_argument("--remote", metavar="HOST:PORT", default=defaults["remote"],
                       help=f"server address seen from the SSH host (default {DEFAULT_REMOTE}, env MLPULL_REMOTE)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", parents=[common], help="a run and all its nested runs").add_argument(
        "run_id"
    )
    sub.add_parser("experiment", parents=[common], help="every run of an experiment").add_argument(
        "experiment"
    )
    args = parser.parse_args(argv)

    with ExitStack() as stack:
        if args.ssh:
            # Env, not just the client argument: proxied ``mlflow-artifacts:`` stores
            # resolve the server from MLFLOW_TRACKING_URI.
            os.environ["MLFLOW_TRACKING_URI"] = stack.enter_context(ssh_tunnel(args.ssh, args.remote))
        client = MlflowClient()
        if args.command == "run":
            count = download_run_tree(client, args.run_id, args.out)
        else:
            count = download_experiment(client, args.experiment, args.out)
    print(f"Downloaded {count} run(s) to {args.out.resolve()}")
