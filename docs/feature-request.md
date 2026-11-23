# mlpull feature requests

Status: proposed, not implemented, except where an item says otherwise. Each item is generic MLflow client behaviour; none names a project, cluster or run layout.

## Today

`mlpull run <ids>` and `mlpull experiment <names|ids|globs>` download artifacts of FINISHED runs, mirroring the nested-run tree (`tree.py` builds the tree; `plan.py` decides the files and `execute.py` downloads them). `--path`, `--max-size`, `--dry-run` and `--what` make downloads selective, and a per-run manifest makes a repeated pull fetch only what is missing. Selection on `experiment`: `--since`, `--until`, `--tag K=V`, `--param K=V`, `--filter`, `--key`. `mlpull runs` summarises metrics, params and tags of runs of any status as json, jsonl, a Markdown table or text. `run` and `experiment` take `--format jsonl` for one record per run, and the exit codes tell a refusal (1), a usage error (2) and an empty selection (3) apart; see the README. `--ssh HOST` opens a `-L` tunnel for the duration of the command (`tunnel.py`). `cli.py` wires these together.

Gap seen when the tool is used by scripts: `run` takes ids only; `experiment` can select by tag.

## Deferred

- `mlpull compare <run A> <run B>`: a caller can derive it from `mlpull runs --json`, so a separate command adds no capability yet.
- `~/.config/mlpull/config.toml` with named servers: reading it needs `tomllib`, which is in the standard library only from Python 3.11, and the environment variables `MLPULL_SSH`, `MLPULL_REMOTE` and `MLPULL_OUT` already cover the common case.

## 1. Tunnel hardening

Done: `ssh_command` passes `-o ControlMaster=no -o ControlPath=none`, so an existing ControlPersist master cannot adopt the forward. `--remote` accepts `host:port` and `MLPULL_REMOTE` provides a default, for tracking servers that listen on a port other than 5000.

Deferred: `ServerAliveInterval` and `ServerAliveCountMax`. They detect a dead peer, not a dead parent, so they do not stop an orphaned `ssh -N` after `mlpull` is killed with SIGKILL. Another mechanism would be needed, and it is not worth the complexity yet.

Not planned: a `ui` subcommand. Opening a server UI in a browser is generic transport and belongs to a managed forward (`rr fwd <name> --open`, see the rr request). mlpull stays MLflow-specific.

## 2. Client and server version matrix

The client floor is `mlflow-skinny>=3.7`, the first release in which `MlflowClient` hands its own tracking URI to the artifact repository. Older clients resolve `mlflow-artifacts:` stores against the global URI and would ignore the SSH tunnel, so the floor is set by that behaviour and not by what the sqlite-backed suite tolerates (it passes back to 2.10 with `sqlalchemy<2.1` and `setuptools<81`). The existing tests use a sqlite-backed local store, which bypasses `HttpStore`, run-search paging and artifact proxying, so they cannot show that the client works against a real server. Add one slow test (or a documented manual check) that starts a real `mlflow server --app-name basic-auth` and runs the read paths against it.

## 3. TODO: authentication without a password in the environment

Not needed today: the servers used with mlpull have no passwords, so nothing is passed. If one is added, `mlpull` should not make users export `MLFLOW_TRACKING_PASSWORD` (it is visible to every child process and to anyone who can read `/proc/<pid>/environ` as that user, and `setx` stores it in the Windows registry in plaintext). Options, in order of preference:

- Rely on the SSH tunnel as the only authentication. The tunnel already requires an SSH key, and with the server bound to `127.0.0.1` no one else can reach it, so basic auth adds little. This is the simplest fix and needs no code.
- If the server keeps basic auth, use a token (`MLFLOW_TRACKING_TOKEN`) instead of a user password, so that it can be revoked and scoped, and read it from a `0600` file or the system keyring at start. The read belongs in `cli.py` only; the pure modules never see it.
- Do not add a `--password` option: command-line arguments show up in `ps` and in shell history.

## Breaking changes

Exit code 3 for "nothing matched" replaced exit 0 when `run`, `experiment` or `runs` found no run, and an `experiment` or `runs` glob that matches no experiment now exits 3 instead of 1. All other additions are new options or commands; defaults of `run` and `experiment` stay.

## Suggested order

Floor and compatibility (item 2) first. Tunnel hardening (item 1) and authentication (item 3) are independent of the rest.
