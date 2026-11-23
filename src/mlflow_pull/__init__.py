"""Download the artifacts of MLflow runs, their nested runs, or whole experiments.

Layout: ``selection``, ``paths``, ``tree``, ``resolve`` decide which runs to fetch (pure,
or reads through the ``RunGateway`` port); ``pathfilter`` and ``plan`` decide which files
of each run (globs, sizes, manifests, plans; pure); ``summary``, ``render`` and ``report``
compute and format run summaries and pull reports (pure), ``exitcodes`` names the exit codes; ``gateway``, ``execute``, ``localfs`` and
``tunnel`` perform effects (listings, downloads, guarded local files, ssh); ``cli`` wires them together.
"""

from mlflow_pull.cli import app
from mlflow_pull.execute import (
    download,
    download_experiments,
    download_runs,
    execute_plan,
    plan_pull,
    pull,
    summarise_runs,
)
from mlflow_pull.exitcodes import ExitCode
from mlflow_pull.gateway import MlflowGateway
from mlflow_pull.pathfilter import PathFilter
from mlflow_pull.paths import RelPath
from mlflow_pull.plan import (
    Content,
    Downloaded,
    Manifest,
    PullOptions,
    PullPlan,
    RunOutcome,
    RunPlan,
    SizeLimitError,
    UpToDate,
    What,
    parse_size,
    plan_run,
)
from mlflow_pull.ports import ArtifactDir, ArtifactEntry, ArtifactFile, KnownSize, RunGateway, UnknownSize
from mlflow_pull.render import (
    Column,
    FieldColumn,
    Format,
    MetricColumn,
    RunField,
    Stat,
    default_columns,
    parse_columns,
    render,
)
from mlflow_pull.resolve import resolve_experiments, resolve_run_list, resolve_runs
from mlflow_pull.selection import (
    FAILED,
    FINISHED,
    UNFINISHED,
    RunStatusError,
    allowed_statuses,
    build_filter,
    parse_when,
)
from mlflow_pull.summary import (
    MetricStats,
    Point,
    RunSummary,
    downsample,
    matching_keys,
    summarise_metric,
)
from mlflow_pull.tree import PARENT_RUN_TAG
from mlflow_pull.tunnel import RemoteAddr, SshHost, ssh_command, ssh_tunnel

__all__ = [
    "ArtifactDir",
    "ArtifactEntry",
    "ArtifactFile",
    "Content",
    "Downloaded",
    "ExitCode",
    "KnownSize",
    "Manifest",
    "PathFilter",
    "PullOptions",
    "PullPlan",
    "RelPath",
    "RunOutcome",
    "RunPlan",
    "SizeLimitError",
    "UnknownSize",
    "UpToDate",
    "What",
    "execute_plan",
    "parse_size",
    "plan_pull",
    "plan_run",
    "pull",
    "FAILED",
    "FINISHED",
    "PARENT_RUN_TAG",
    "UNFINISHED",
    "Column",
    "FieldColumn",
    "Format",
    "MetricColumn",
    "MetricStats",
    "MlflowGateway",
    "Point",
    "RemoteAddr",
    "RunField",
    "RunGateway",
    "RunStatusError",
    "RunSummary",
    "SshHost",
    "Stat",
    "allowed_statuses",
    "app",
    "build_filter",
    "default_columns",
    "download",
    "download_experiments",
    "download_runs",
    "downsample",
    "matching_keys",
    "parse_columns",
    "parse_when",
    "render",
    "resolve_experiments",
    "resolve_run_list",
    "resolve_runs",
    "ssh_command",
    "ssh_tunnel",
    "summarise_metric",
    "summarise_runs",
]
