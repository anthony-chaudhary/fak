# GPU-Server / Slack Boundary

This is the source of truth for the recurring GPU-server/Slack confusion.

> **Just want to reach the channel?** See [`private-comms-channel.md`](private-comms-channel.md)
> — the public stub that points to the live Slack control-bridge in `fak-private`. This doc
> explains *what is public vs private and why*; that stub is the entry point.

## Public tree

The public `fak` tree may keep scrubbed benchmark evidence, runbooks, and result
summaries for the GPU-server work. Those artifacts must use generic public language
(`GPU server`, hardware class, no lab host/IP/path/token) and must pass the scrub and
file-admission gates.

Examples that can be public:

- `docs/benchmarks/*GPU-SERVER*.md`
- scrubbed result artifacts under `experiments/qwen36/...`
- GPU acceptance scripts that run local commands and do not implement the lab control
  channel

## Private tree

The live control plane for the lab GPU server is private operational plumbing. It
belongs in `fak-private`, not here.

Private-only paths and concepts:

- `cmd/*dgx*/`, `internal/*dgx*/`
- `cmd/slackgc/`
- `cmd/*slack*bridge*/`, `internal/*slack*control*/`, and similar Slack control-bridge
  packages
- the sunset Python bridge paths `tools/bench_slack.py` and `tools/bench_slack_test.py`
- GPU-server machine catalog runs under `experiments/benchmark/runs/by-machine/dgx*/`
- raw Slack-control state, transcripts, tokens, workspace IDs, lab hostnames, and
  operator paths

## Go vs Python

New public tooling is Go. Add a `fak` subcommand or a small `cmd/<name>/` binary, with
pure logic under `internal/<name>/` where appropriate. Do not add a new `tools/*.py`.

Existing Python tools are grandfathered only. The allowlist in `internal/pythongate` can
shrink when a Python tool is ported or sunset, but it must not grow. Restoring
`tools/bench_slack.py` would violate both rules: it is a new Python path after deletion
and it is private Slack/GPU-server control-plane code.

## Enforced by

- `internal/pythongate`: refuses new tracked `tools/*.py`
- `tools/check_committed_files.py`: refuses private-only Slack/GPU-server control paths
- `.gitignore`: keeps private GPU-server run outputs and bridge working copies out of status
- `tools/scrub_public_copy.py`: strips private GPU-server machine runs and lab identifiers from
  exported copies
