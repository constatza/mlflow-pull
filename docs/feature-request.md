# mlpull feature requests

Open and deferred items only. Each is generic MLflow client behaviour; none names a project, cluster or run layout. What mlpull does today is in the README.

## Deferred

- `mlpull compare <run A> <run B>`: a caller can derive it from `mlpull runs --format json`, so a separate command adds no capability yet.
- `~/.config/mlpull/config.toml` with named servers: the environment variables `MLPULL_SSH`, `MLPULL_REMOTE` and `MLPULL_OUT` already cover the common case.
- Selection for `run` by tag, parameter or start time: `run` takes ids only, and those selectors exist on `experiment` alone.
- A way around a name that MLflow's HTTP listing refuses: over `mlflow-artifacts:` a `#` in an artifact name makes the server reject the listing of its whole directory, so the run fails with exit code 1 before anything is written, and the error names neither the directory nor the name. Skipping only that directory would need both.

## Tunnel: ServerAliveInterval and ServerAliveCountMax

Deferred. They detect a dead peer, not a dead parent, so they do not stop an orphaned `ssh -N` after `mlpull` is killed with SIGKILL, which is the one case that still leaves `ssh -N` running. Another mechanism would be needed, and it is not worth the complexity yet.

Not planned: a `ui` subcommand. Opening a server UI in a browser is generic transport and belongs to a managed forward. mlpull stays MLflow-specific.

## Client and server version matrix

The client floor is `mlflow-skinny>=3.7`, the first release in which `MlflowClient` hands its own tracking URI to the artifact repository. Older clients resolve `mlflow-artifacts:` stores against the global URI and would ignore the SSH tunnel, so the floor is set by that behaviour and not by what the sqlite-backed suite tolerates (it passes back to 2.10 with `sqlalchemy<2.1` and `setuptools<81`). The slow tests (`pytest -m slow`) start a real `mlflow server --app-name basic-auth` and exercise the HTTP read paths, but only with the MLflow version the lock file pins. Running them against other client and server versions is not automated.

## Authentication without a password in the environment

Not needed today: the servers used with mlpull have no passwords, so nothing is passed. If one is added, `mlpull` should not make users export `MLFLOW_TRACKING_PASSWORD` (it is visible to every child process and to anyone who can read `/proc/<pid>/environ` as that user, and `setx` stores it in the Windows registry in plaintext). Options, in order of preference:

- Rely on the SSH tunnel as the only authentication. The tunnel already requires an SSH key, and with the server bound to `127.0.0.1` no one else can reach it, so basic auth adds little. This is the simplest fix and needs no code.
- If the server keeps basic auth, use a token (`MLFLOW_TRACKING_TOKEN`) instead of a user password, so that it can be revoked and scoped, and read it from a `0600` file or the system keyring at start. The read belongs in `cli.py` only; the pure modules never see it.
- Do not add a `--password` option: command-line arguments show up in `ps` and in shell history.
