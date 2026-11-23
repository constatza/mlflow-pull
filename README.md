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
mlpull run --ssh myhost <run_id>
mlpull run --ssh myhost -z <run_id>   # -z: compress data in transit (ssh -C), like rsync -z
export MLPULL_SSH=myhost              # or set it once; PowerShell: setx MLPULL_SSH myhost
```

The system `ssh` client is used, so `~/.ssh/config` (aliases, `User`, `Port`, `IdentityFile`, `ProxyJump`), ssh-agent and `known_hosts` apply as for `ssh myhost`. It runs non-interactively (`BatchMode`): key authentication must work without a prompt, through ssh-agent or an unencrypted key. The server is assumed at `localhost:5000` as seen from HOST; change it with `--remote HOST:PORT` or `MLPULL_REMOTE`.

`--ssh` takes an ssh alias, a host name or `user@host`, and may not start with `-`, so a value can never be read as an ssh option. `--remote` takes `HOST:PORT`, where HOST is a host name or an `[ipv6]` address and PORT is 1 to 65535. Both are checked before ssh is started.

## Security and credentials

`mlpull` never reads, stores, logs or accepts credentials. It has no password or token option. Authentication is left to the MLflow client, which reads `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` and `MLFLOW_TRACKING_TOKEN` from the environment and a credentials file at `~/.mlflow/credentials`. That file is plaintext ini, and MLflow does not check its permissions, so keep it private with `chmod 600 ~/.mlflow/credentials`. On Windows, `setx` stores the value in the registry in plaintext; do not use it for passwords. Use `$env:` for the current session instead.

The tool does not write `MLFLOW_TRACKING_URI` into its own environment. It passes the tunnel address to the MLflow client directly, so nothing leaks into child processes.

Things to know about the tunnel:

- It listens on `127.0.0.1` on a random port. Every local user of that machine can connect to that port. On a shared machine, only use `--ssh` with servers that require authentication.
- The SSH connection protects the path to the SSH host only. The last hop, from the SSH host to the server, is plain HTTP unless `--remote` points at a loopback address on that host.
- A server that binds `0.0.0.0` on a shared login node sends basic-auth passwords in cleartext over the cluster network. Bind the server to `127.0.0.1` where possible and let `--ssh` reach it there.
- `--ssh` ignores any existing `ControlMaster` or `ControlPath` setting (it passes `-o ControlMaster=no -o ControlPath=none`). Otherwise an already-running master connection would adopt the forward and keep it open after `mlpull` exits; with these options the forward closes when `mlpull` does. That holds when `mlpull` ends normally, on an error, on Ctrl-C and on SIGTERM (a plain `kill`, exit code 143). SIGKILL (`kill -9`) cannot be caught and leaves `ssh -N` running; find it with `pgrep -af 'ssh -N'` and stop it.

## Use

```sh
mlpull run <run_id>... [options]            # runs and all their nested runs
mlpull experiment <experiment>... [options] # runs of experiments (names, ids or * globs)
mlpull runs [options]                       # read-only summary of runs: metrics, params, tags
mlpull run --help                           # every option
```

Options go after the command. `-o` / `--out` sets the destination (env `MLPULL_OUT`, default `./mlflow-downloads`); `-j N` downloads N run trees in parallel (default 4).

### Run status

Only `FINISHED` runs have their artifacts downloaded, so a run that is still going or that crashed never lands as a partial export. A run named on the command line with another status stops the command before anything is written; nested and experiment runs with another status are skipped with a note. The gate protects artifacts only: with `--what` set to metrics, params or tags alone (see below), runs of any status are read.

| flag | also downloads |
|---|---|
| `-u`, `--unfinished` | `RUNNING`, `SCHEDULED` |
| `-f`, `--failed` | `FAILED`, `KILLED` |
| `-a`, `--all` | every status |

### Selecting runs of an experiment

`experiment` takes selection options; all given options must match (AND):

| option | selects runs |
|---|---|
| `--since WHEN`, `--until WHEN` | started at/after, or before, WHEN: ISO date or datetime (local time if no zone), or `30m`, `6h`, `2d`, `1w` ago |
| `--tag KEY=VALUE` | with this tag value; repeatable (`--tag mlflow.user=alice`) |
| `--param KEY=VALUE` | with this parameter value; repeatable |
| `--filter EXPR` | matching a raw [MLflow filter](https://mlflow.org/docs/latest/search-runs.html), e.g. `"metrics.loss < 0.1"` |

Without selection options every top-level run is downloaded. With them, every matching run at any depth is downloaded with its nested runs (a match inside another match's subtree is not downloaded twice), directly under `<experiment name>/`.

```sh
mlpull experiment --ssh myhost --tag phase=report --since 2d 'my-project/study-*'
```

Child runs are placed inside their parent's directory, so the output mirrors the run tree:

```
mlflow-downloads/
└── <run_name>-<id8>/              artifacts of the run
    ├── figures/...
    └── <child_name>-<id8>/        artifacts of a nested run
        └── <grandchild_name>-<id8>/...
```

Directory names are `<run name>-<first 8 characters of the run id>`. Names are made safe as a single path component, because MLflow accepts an experiment named `..` and a run or experiment name must never move the output outside the destination. A name that is empty, `.` or `..` is replaced with safe text. Control characters and characters invalid in Windows file names are replaced with `_`. Windows device names such as `CON` and `NUL` are replaced too, and trailing dots and spaces are stripped, since Windows silently drops them. Experiment mode adds one more level, `<experiment name>/`, on top.

### Selective downloads

By default every artifact of every selected run is fetched. These options on `run` and `experiment` fetch less, and a repeated pull fetches only what is missing. Each run is first planned (its artifact directories are listed, one call per directory) and only then downloaded, so the options below are decided before anything is written.

| option | effect |
|---|---|
| `--path GLOB` | Only artifacts whose path matches GLOB. Repeatable; `!GLOB` excludes, and an exclude beats every include. Without `--path` every artifact is taken. |
| `--max-size SIZE` | Refuse, before writing anything, when the files to fetch add up to more than SIZE. Exit code 1. |
| `--dry-run` | Print, per run, the files that would be fetched with their sizes and the totals, and write nothing, not even directories. Applies `--max-size` too: exit code 0, or 1 when the download would be refused. |
| `--what LIST` | What to pull per run: any of `metrics`, `params`, `tags`, `artifacts`, as a comma list or repeated. Default `artifacts`. |
| `--format text\|jsonl` | `text` (default) prints for people. `jsonl` prints one JSON record per run on stdout and every note on stderr; see Output and exit codes. |

```sh
mlpull run <run_id> --path 'config/**' --path 'plots/*.png' --path '!plots/draft*' --dry-run
mlpull experiment my-project --max-size 2G
mlpull run <run_id> --what metrics,params           # run.json only, any run status
```

Quote the globs: the shell would expand `*` and read `!` as history expansion.

**Globs** match the path of an artifact relative to the run's artifact root (`config/a.yaml`, not the local directory). `*` matches any characters except `/`, `?` matches one character except `/`, and `**` as a whole path component matches any number of directories, including none. So `plots/*.png` does not match `plots/old/a.png`, while `plots/**` and `**/a.png` do match. `[` is an ordinary character, not a class: `a[1].txt` matches only a file of that name. A glob that ends in `/` (`plots/`) matches nothing and is refused; write `plots/**` for everything below a directory. `.` and `..` components are refused. An include glob that cannot match anything below a directory (for example `config/**` and the directory `plots/`) keeps that directory from being listed at all, and so does an exclude ending in `/**`; the result is the same as filtering the full listing, with fewer calls.

**Sizes** are in units of 1024: `K`, `M`, `G`, `T` (so `1G` is 1073741824 bytes), as in `500M`, `1.5G`, `2GiB`, or a bare number of bytes. The size checked is that of the files that would be fetched, after `--path` and after skipping files an earlier pull already holds. A second pull of the same runs therefore passes any limit.

**Manifest.** `mlpull` writes `.mlpull.json` in the run's directory (nested runs have their own, in their own directories) through a temporary file and a rename, so it never holds a half-written record. A file is recorded only once it has been downloaded and verified, and when a run's pull stops on an error or an interrupt the manifest of the files verified so far is still written, so a repeated pull fetches only the rest. It records the run id, the run's status at that pull and, for each fetched file, its path and the size and modification time (`mtime_ns`) the file system reported right after the download:

```json
{"version": 2, "run_id": "...", "status": "FINISHED", "files": {"config/a.yaml": {"size": 120, "mtime_ns": 1763900000000000000}}}
```

On the next pull a file is skipped when the manifest records it, the server still lists the recorded size, and the file on disk still has that size and that modification time. A store that lists no size for a file is trusted only for a finished run: there a recorded file that is intact on disk is skipped too. A different `--path` set therefore fetches only the files not fetched before, and files outside the current `--path` stay recorded. If the run was not finished when it was last pulled (`RUNNING` or `SCHEDULED`, pulled with `-u`), everything selected is fetched again, because its files may have been changing. A manifest that is missing, unreadable, corrupt, of another run or of an unknown version counts as empty: the files are fetched again and the manifest is rewritten. Delete `.mlpull.json` to force a full fetch. The manifest is untrusted input: it is read only as a regular file (never through a symlink, never from a FIFO), at most 16 MiB of it, and its `status` can only make mlpull fetch more (an unrecognised status counts as not finished), never skip more.

There is no content hash. A file that was replaced by one of the same size and the same modification time (for example restored with `cp -p`) is taken for the recorded one; delete it or the manifest to refetch.

The manifest and `run.json` are written at the root of the run directory. An artifact that is one of those names, or lies below a directory of that name (`.mlpull.json/x`), would collide, and so would one below the directory of a nested run of the same run (`child-1a2b3c4d/...`), so the pull stops with an error before writing anything; leave the artifact out with `--path '!.mlpull.json'` (or `'!run.json'`, or the artifact's own path). Artifacts of that name in subdirectories are fine. The same check stops two artifacts that are one file on the target (`é` composed and decomposed everywhere; on Windows and macOS, which fold case, also `A.txt` and `a.txt`, and the reserved names match in any case there, while on Linux `Model.pkl` and `model.pkl` are two files), and a file that is also the directory of another artifact (`a` and `a/b`). Each of these errors names the `--path '!NAME'` that leaves one of the artifacts out.

### Safety of downloads

The tracking server and the artifact store are not trusted, and neither is what an earlier pull left in the output directory. Before anything is written mlpull checks every artifact path. A hostile path (`..`, an absolute path, an empty component, control characters, invalid UTF-8) stops the pull with exit code 1. A harmless name the machine cannot create is skipped like the `#` names below: one over 255 bytes per component or 1024 per path, and on Windows also a backslash, a drive letter, `:` and the other characters Windows refuses, trailing dots or spaces, and device names. A directory with such a name is skipped with everything below it and counted once; `--path '!NAME'` removes either from the count. Each run's listing is walked once per directory and stops with exit code 1 beyond 64 levels or 200000 entries. A listed size must be a non-negative integer. When a size is not listed, `--max-size` refuses the pull (exit code 1) rather than count the file as empty; without `--max-size` the file is fetched. After each download the file must be a regular file of the listed size, and the manifest records what is on disk.

Every directory and file below the output directory is checked with `lstat` before it is created or written, and a symlink anywhere on the way is an error (exit code 1), so a link planted in an earlier or hostile pull cannot redirect a download. The output directory itself and its parents may be symlinks. A process that can swap a directory for a link between the check and the download is not stopped. The manifest and `run.json` are written through a fresh, exclusively created temporary file that is synced and renamed over the target, so two pulls into one directory do not collide (the last one wins) and a planted `.mlpull.json.tmp` is never followed; they get the permissions the umask gives every downloaded file, so group readers of a shared output directory can read them, and a temporary file left by a crash is removed by the next pull of that run once it is an hour old. A run directory that is a symlink is never read for a manifest (nothing counts as downloaded) and the pull of that run stops with an error. MLflow decides itself whether a listed file is a directory and then downloads it recursively; mlpull cannot stop that and reports it as an error afterwards.

MLflow cannot download an artifact whose name contains `#`, `?` or a percent escape such as `%41`: it builds a `runs:/` URI from the name, and these names fetch nothing or another file. Such artifacts are skipped, counted (`unsupported` in the jsonl record, `N unsupported name(s) skipped` in the summary and dry-run lines), named on stderr, and left out of `--max-size` and the manifest. The rest of the run is downloaded. Names with `;`, a lone `%`, spaces, `+` or non-ASCII letters are fine.

A run still `RUNNING` (pulled with `-u`) can append to an artifact between the listing and the download. For such a run a file that is not the size listed is not an error: it is counted (`N file(s) changed while pulling` in the final line), left out of the manifest and fetched again by the next pull. For a finished run a size mismatch stays an error.

**`--what`.** Anything other than `artifacts` writes `run.json` in each run directory: the run's record as `mlpull runs --format json` prints it for one run (status, start and end times, params, tags and the latest value of each metric; see below). The record is the same whichever of `metrics`, `params` and `tags` you name. It is written whatever the run's status. `--what metrics,artifacts` downloads artifacts too, and then the status gate applies to the run.

The final line reads `Downloaded N run(s), M file(s), SIZE to <dir> (K run(s) already up to date)`; SIZE is in B, KiB, MiB, GiB or TiB; the part in parentheses is left out when K is 0.

## Output and exit codes

`run` and `experiment` print for people by default: skipped runs and refusals on stderr, and a last line on stdout. `--format jsonl` is for scripts: stdout carries only JSON lines, one per run, and every human note (skipped runs, refusals, the summary line, and the file listing of `--dry-run`) goes to stderr.

```sh
mlpull run <run_id> --format jsonl 2>/dev/null | jq -r 'select(.outcome == "downloaded") | .dir'
```

Each line is one strict JSON object (no NaN, UTF-8-safe ASCII, no blank lines):

```json
{"run_id": "...", "name": "parent", "status": "FINISHED", "dir": "/abs/out/parent-1a2b3c4d", "outcome": "downloaded", "files": 12, "bytes": 3407, "unsupported": 0}
```

| `outcome` | meaning | `files`, `bytes` |
|---|---|---|
| `downloaded` | something was written for the run | files fetched and their size |
| `up_to_date` | nothing to write; an earlier pull already holds everything selected | files already in place and their size |
| `skipped_status` | the run's status is not allowed (see Run status); its nested runs are left out too | 0, 0 |
| `would_download` | `--dry-run` only: what a pull would fetch | files and size that would be fetched |

`unsupported` counts the artifacts skipped because MLflow cannot download their names (see Safety of downloads); it is 0 for `skipped_status`. `dir` is absolute: where the run's files are, or would go. The records come in the order the runs were resolved (each run before its nested runs); runs skipped by status come after the others. With `--dry-run`, `--format jsonl` still lists the records and still stops with exit code 1 when `--max-size` would refuse the download.

Exit codes:

| code | meaning |
|---|---|
| 0 | Done. This includes a pull where every run was already up to date. |
| 1 | Refused or failed: a named run outside the allowed statuses, `--max-size` exceeded or not checkable, an unsafe artifact path, a reserved or conflicting name, a symlink in the way, an artifact listing beyond the limits, a download that is not the listed file, an unknown experiment name or id, an MLflow or file system error, a failed SSH tunnel (`--ssh`: no `ssh` binary, authentication refused, no free local port, not ready in time), Ctrl-C (click prints `Aborted!`), a closed stdout such as `mlpull runs | head` (silent). Nothing is written for a refusal. |
| 2 | Usage error: an unknown option or value, a bad glob or size, `--history` without `json` or `jsonl`. |
| 3 | Nothing matched: the selection was valid but no run is left to download, because no run matched, or every candidate was skipped by status, or an experiment `*` glob matched no experiment. For `runs`: no run was found (`json` still prints `[]`, `jsonl` prints nothing). |

A pull or `runs` that finds nothing exits 3, so a script can tell an empty result from a download. A glob is a selection, so a glob that matches no experiment exits 3; a plain name or id that does not exist is a mistake and exits 1. SIGTERM ends the process with 143 (128 + 15) after closing the tunnel.

Every failure above prints one line on stderr and no traceback. The one residual: an artifact store whose client raises its own exception type. boto3 errors (`botocore` `BotoCoreError`, `ClientError`) are mapped to one line when boto3 is installed; any other store library's own error types are not and show a traceback with exit code 1.

## Reading runs

`mlpull runs` only reads: it never downloads or writes anything, and it works on runs of any status, including `RUNNING`. It lists the runs of the chosen experiments, nested runs included, ordered by start time. It exits 3 when no run is found (see Output and exit codes), including when an `-e` glob matches no experiment: the empty result is still printed first (`[]` for `json`, nothing for `jsonl`, the header for the table formats).

With `MLFLOW_TRACKING_URI` unset the MLflow client falls back to a local store in the current directory and creates it: `./mlflow.db` with the client floor of 3.7 and later (older clients used `./mlruns/`), so "never writes" holds only with a server URI (or `--ssh`).

```sh
mlpull runs                                               # every run of every experiment, latest metric values
mlpull runs -e my-project --since 2d --tag phase=report   # selected runs of one experiment
mlpull runs -e 'my-project/study-*' --key abc --metric 'val/*' --format md-table \
    --columns name,status,val/loss:min,val/loss:last
mlpull runs -e my-project --metric 'val/loss' --history 20 --format json
```

| option | meaning |
|---|---|
| `-e`, `--experiment NAME\|ID\|GLOB` | Experiment to read; repeatable. Default: all experiments. Names, ids or `*` globs over names, as for `experiment`. |
| `--since`, `--until`, `--tag`, `--param`, `--filter` | Select runs, as for `experiment`. |
| `--key VALUE` | Runs whose tag `--tag-name` equals VALUE: shorthand for `--tag run_key=VALUE`. `--tag-name` (env `MLPULL_KEY_TAG`) defaults to `run_key`. Not available on `run`, which takes run ids. |
| `--metric GLOB` | Summarise the full history of the metric keys matching GLOB (repeatable): minimum, step of the minimum, last value and number of points. `*` matches any characters, including `/`, so `val/*` also matches `val/a/b`. |
| `--history N` | With `--metric`: also give at most N points per key (at least 2), spread evenly, always keeping the first and the last. |
| `--format` | `text` (default, aligned columns), `md-table`, `json` (an array of records) or `jsonl` (one record per line). |
| `--columns COL,...` | Columns of the table formats; see below. |
| `--ssh`, `--remote`, `-z`, `-j` | As for downloads; `-j` runs history reads in parallel. |

One run search returns only the latest value of each metric. Without `--metric` that is all you get, and no history is read. With `--metric`, one history call per run and matched key is made, in parallel.

**Columns.** The table formats take `--columns` as a comma list of run fields `run_id`, `name`, `status`, `start`, `end`, and metric statistics written `KEY:STAT`, with STAT one of `min`, `last`, `min_step`, `count`. The metric key is literal (no glob characters) and is split from the statistic at the last colon, so `val/loss:min` works for keys that contain `/`. Every statistic but `last` needs a key matched by a `--metric` glob, because it comes from the history; `val/loss:last` without `--metric` shows the latest value. Without `--columns`: `run_id`, `name`, `status`, then `KEY:min` and `KEY:last` for each key the `--metric` globs matched, or `KEY:last` (the latest value) for every metric when there is no `--metric`. Table cells are plain text: floats print in full precision, a missing value is empty, and characters that would drive a terminal are shown as escapes.

**JSON.** Each run is one record:

```json
{"run_id": "...", "name": "...", "status": "FINISHED",
 "start": "2026-11-23T10:00:00+00:00", "end": null,
 "params": {"lr": "0.1"}, "tags": {"run_key": "abc"},
 "latest": {"val/loss": 0.3},
 "metrics": {"val/loss": {"min": 0.2, "min_step": 2, "last": 0.3, "last_step": 4, "count": 5}},
 "history": {"val/loss": [[0, 1.0], [4, 0.3]]}}
```

`end` is `null` while the run has not ended. `metrics` and `history` only hold the keys matched by `--metric` (`history` only with `--history`). JSON has no NaN or infinity, so those values are `null`.

## Notes

- Artifact stores that need their own client library (for example S3 needs `boto3`) require it in the tool's environment: `uv tool install --with boto3 git+https://github.com/constatza/mlflow-pull`.
- Nested runs are found through the standard `mlflow.parentRunId` tag and are assumed to live in the same experiment as their parent.

## Development

```sh
git clone https://github.com/constatza/mlflow-pull
cd mlflow-pull
uv run pytest
```

The modules under `src/mlflow_pull/` split into decision and effect code. `selection`, `paths`, `tree` and `resolve` decide which runs to fetch, and `pathfilter` and `plan` decide which files of each run (globs, size limit, manifest diff), as pure functions or by reading through the `RunGateway` port, so they test without a server. `summary`, `render` and `report` compute and format run summaries and pull reports, and `exitcodes` names the process exit codes. `gateway`, `execute`, `localfs` and `tunnel` perform the effects (MLflow calls, guarded file reads and writes, ssh), and `cli` wires them together; see the docstring in `src/mlflow_pull/__init__.py`. A pull is two steps: `plan_pull` lists and diffs without writing, `execute_plan` fetches exactly the planned files.
