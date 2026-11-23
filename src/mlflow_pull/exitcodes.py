"""The process exit codes of ``mlpull``, in one place so scripts and the docs read the same list."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    REFUSED = 1  # status gate, --max-size, unsafe or reserved artifact path, unknown experiment, MLflow errors
    USAGE = 2  # raised by typer itself for a bad option or argument
    NOTHING_MATCHED = 3  # the selection was valid but picked no run
