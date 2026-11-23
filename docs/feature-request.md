# mlpull feature requests

Status: proposed, not implemented. Each item is generic MLflow client behaviour; none names a project, cluster or run layout.

## Today

`mlpull run <ids>` and `mlpull experiment <names|ids|globs>` download artifacts of FINISHED runs, mirroring the nested-run tree (`__init__.py:94-120`). Selection on `experiment`: `--since`, `--until`, `--tag K=V`, `--param K=V`, `--filter`. `--ssh HOST` opens a `-L` tunnel for the duration of the command (`:216-263`).

Gaps seen when the tool is used by people and scripts that read finished or running experiments:

- It never reads metrics, params or tags.
- It writes no JSON and has no list or dry-run mode.
- `experiment` with no selection pulls every artifact of every run, including model weights, and has no size guard.
- A re-run downloads everything again.
- `run` takes ids only; `experiment` can select by tag.
- The default remote is `localhost:5000`.

## 1. `mlpull summary` and `mlpull runs`

Read-only, one MLflow API round trip per run, nothing downloaded.

```
mlpull summary --key <tag value> [--tag-name run_key] --metric 'val/*' --metric 'train/loss/*' --json
mlpull runs --experiment <name|glob> --tag K=V --since 2d --format md-table --columns name,status,val/loss:min,val/loss:last
```

- Per run: id, name, status, start and end, params, selected tags, and for each metric glob the minimum, the last value, the step of the minimum and the number of points. Optional downsampled history (`--history N`).
- Formats: `--json`, JSON lines, a Markdown table (`--format md-table --columns ...`), plain text.
- Metrics, params and tags are safe to read for RUNNING runs, so the FINISHED gate applies to artifacts only.
- `--key K` is sugar for a tag filter (default tag name is configurable) and is accepted by `run` as well as `experiment`.
- Why it matters: callers currently write the same `MlflowClient.search_runs` plus `get_metric_history` snippet over and over to get a minimum and a last value.

Tests: a local MLflow store fixture with nested runs, a RUNNING run, and a metric with several steps; assert the min, last and argmin step.

## 2. Selective, cheaper downloads

- `--what metrics,params,tags,artifacts` (default `artifacts`, unchanged).
- `--path GLOB` (repeatable, `!GLOB` excludes): `--path 'config/**' --path 'plots/*.png' --path '!model/**'`.
- `--max-size 1G`: refuse before downloading, as `rr get` does; measured from artifact listings.
- `--dry-run` and `mlpull ls`: list what would be fetched with sizes.
- A per-run manifest (`.mlpull.json`: run id, status at pull time, files, sizes) so a finished run already pulled is skipped and a run that was RUNNING is refreshed.

Tests: path glob include and exclude; size refusal writes nothing; second run is a no-op; a run that finished since the last pull is re-pulled.

## 3. `mlpull compare` and structured output

- `mlpull compare <run A> <run B> [--json]`: params that differ and a side-by-side metric table (min, last).
- JSON-lines output for every command (`{run_id, name, status, dir, files, bytes}`) and distinct exit codes: refused (status gate), usage error, nothing matched.
- Optional `~/.config/mlpull/config.toml` with named servers (`ssh`, `remote`, `out`), so `--ssh` and the environment variables are not needed for the common case.

## 4. Tunnel hardening (small, non-breaking)

- `ssh_command` (`__init__.py:216`) adds `-o ControlMaster=no -o ControlPath=none`, so an existing ControlPersist master cannot adopt the forward (test at `tests/test_download.py:290`).
- Default `--remote` accepts `host:port` and an environment default (`MLPULL_REMOTE`); document that a tracking server may listen on a port other than 5000.
- Add `ServerAliveInterval=15 -o ServerAliveCountMax=3` so an orphaned child ends by itself if the parent is killed with SIGKILL.
- Do not add a `ui` subcommand. Opening a server UI in a browser is generic transport and belongs to a managed forward (`rr fwd <name> --open`, see the rr request). mlpull stays MLflow-specific.

## 5. Client and server version matrix

The dependency is `mlflow-skinny>=2.10` while servers run 3.x. Add one slow test (or a documented manual check) that runs the read paths against a 3.x server, and raise the floor if 2.10 no longer works.

## Breaking changes

None. All additions are new options or commands; defaults of `run` and `experiment` stay.

## Suggested order

1, then 2, then 4 (a few lines), then 3.
