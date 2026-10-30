# mlflow-pull

Download MLflow artifacts in bulk: a run, all its nested runs, or a whole experiment.
`mlflow artifacts download` handles one run; nested runs are separate runs and the GUI has no bulk download.

```
uv tool install ~/projects/mlflow-pull        # once; `mlflow-pull` then works in any directory
# or, without installing:
uvx --from ~/projects/mlflow-pull mlflow-pull run <run_id>
# after pushing: uv tool install git+https://github.com/<user>/mlflow-pull

export MLFLOW_TRACKING_URI=http://localhost:5000   # PowerShell: $env:MLFLOW_TRACKING_URI="..."

mlflow-pull run <run_id> [-o DIR]
mlflow-pull experiment <name-or-id> [-o DIR]
```

Output (`-o` defaults to `./mlflow-downloads`), children inside their parent:

```
<out>/<run_name>-<id8>/<artifacts...>
<out>/<run_name>-<id8>/<child_name>-<id8>/<artifacts...>
```

Artifact stores such as S3 need their client library (e.g. `boto3`) in the same environment.
