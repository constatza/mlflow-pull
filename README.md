# mlflow-pull

Download MLflow artifacts in bulk: a single run, a run together with all of its nested runs, or every run of an experiment.

The MLflow GUI has no bulk download, and `mlflow artifacts download` fetches one run at a time. Nested (child) runs are separate runs in MLflow, so their artifacts are not included when you download the parent. `mlpull` walks the run tree for you and mirrors it on disk.

## Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

Install once, then use `mlpull` from any directory:

```sh
uv tool install git+https://github.com/constatza/mlflow-pull
```

Or run it without installing:

```sh
uvx --from git+https://github.com/constatza/mlflow-pull mlpull run <run_id>
```

Update or remove an installed copy:

```sh
uv tool upgrade mlflow-pull
uv tool uninstall mlflow-pull
```

If the command is not found after installing, run `uv tool update-shell` and open a new terminal.

## Configure

Point it at your tracking server with the standard MLflow environment variable. There is no URI flag.

```sh
export MLFLOW_TRACKING_URI=http://localhost:5000
```

PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"   # current session
setx MLFLOW_TRACKING_URI "http://localhost:5000"     # persistent, applies to new shells
```

### Server behind SSH

If the server listens only on a remote machine, `--ssh HOST` opens an SSH tunnel for the duration of the download and points MLflow at it. `MLFLOW_TRACKING_URI` is not needed then.

```sh
mlpull --ssh myhost run <run_id>
export MLPULL_SSH=myhost        # or set it once; PowerShell: setx MLPULL_SSH myhost
```

The system `ssh` client is used, so `~/.ssh/config` (aliases, `User`, `Port`, `IdentityFile`, `ProxyJump`), ssh-agent and `known_hosts` apply as for `ssh myhost`. It runs non-interactively (`BatchMode`): key authentication must work without a prompt, through ssh-agent or an unencrypted key. The server is assumed at `localhost:5000` as seen from HOST; change it with `--remote HOST:PORT` or `MLPULL_REMOTE`.

## Use

```sh
mlpull run <run_id> [-o DIR]               # a run and all its nested runs
mlpull experiment <name-or-id> [-o DIR]    # every run of an experiment, nested runs included
```

`-o` / `--out` defaults to `./mlflow-downloads`.

Child runs are placed inside their parent's directory, so the output mirrors the run tree:

```
mlflow-downloads/
└── <run_name>-<id8>/              artifacts of the run
    ├── figures/...
    └── <child_name>-<id8>/        artifacts of a nested run
        └── <grandchild_name>-<id8>/...
```

Directory names are `<run name>-<first 8 characters of the run id>`. Characters that are invalid in Windows file names are replaced with `_`. Experiment mode adds one more level, `<experiment name>/`, on top.

## Notes

- Artifact stores that need their own client library (for example S3 needs `boto3`) require it in the tool's environment: `uv tool install --with boto3 git+https://github.com/constatza/mlflow-pull`.
- Nested runs are found through the standard `mlflow.parentRunId` tag and are assumed to live in the same experiment as their parent.

## Development

```sh
git clone https://github.com/constatza/mlflow-pull
cd mlflow-pull
uv run pytest
```
