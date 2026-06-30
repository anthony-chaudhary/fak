---
title: "GPU-server private boundary: public vs private"
description: "The source of truth for what is public versus private in fak's GPU-server work, and the scrub and file-admission gates that enforce the boundary."
---

# GPU-Server Private Boundary

This is the source of truth for the recurring GPU-server public/private boundary.

> **Just want to reach the channel?** See [`private-comms-channel.md`](private-comms-channel.md)
> — the public stub that points to the live private control bridge in `fak-private`. This doc
> explains *what is public vs private and why*; that stub is the entry point.
>
> **Operating the box fleet?** See [`fleet.md`](fleet.md) — the public, transport-agnostic
> Go core (`fleetctl`: roster + fold + readiness score + render). It folds the per-box report
> JSON the private bridge writes; the boundary below is the rule it lives inside.

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

- private bridge commands and support packages
- private notification cleanup helpers
- private bridge/control packages
- the sunset Python bridge paths `tools/bench_slack.py` and `tools/bench_slack_test.py`
- GPU-server machine catalog runs under private machine IDs
- raw control-plane state, transcripts, tokens, workspace IDs, lab hostnames, and
  operator paths

## Confirming a feeder actually posted

The feeders fail OPEN by design (a secret-less run renders to the step summary and exits 0),
so a misconfigured feeder is silent. `fak slack health` is the public watchdog that CONFIRMS
a post landed: per surface it folds resolution + `auth.test` + a real `conversations.history`
read into an `OK | INCOMPLETE | AUTH_FAIL | STALE` verdict and exits non-zero on any non-OK.
The unattended arm is `.github/workflows/slack-watchdog.yml`, which files one deduped issue on
a non-OK verdict. Like every public Slack surface here, it carries no token, channel id, or
lab identifier — it reads them from env/`vars` at run time. See
[`cli-reference.md`](cli-reference.md).

## Go vs Python

New public tooling is Go. Add a `fak` subcommand or a small `cmd/<name>/` binary, with
pure logic under `internal/<name>/` where appropriate. Do not add a new `tools/*.py`.

The public, transport-agnostic fleet core now exists in Go: `cmd/fleetctl/` (`fleetctl`)
is the Go home the scattered `tools/fleet_*.py` helpers port into — a typed roster, a
deterministic fold + readiness score, and a render that stays readable at 100+ boxes. It
reads the per-box report JSON the private bridge writes (the seam is a data contract,
not a code import), so the live control plane stays private while the core stays public.
See [`fleet.md`](fleet.md).

Existing Python tools are grandfathered only. The allowlist in `internal/pythongate` can
shrink when a Python tool is ported or sunset, but it must not grow. Restoring
`tools/bench_slack.py` would violate both rules: it is a new Python path after deletion
and it is private GPU-server control-plane code.

## Enforced by

- `internal/pythongate`: refuses new tracked `tools/*.py`
- `tools/check_committed_files.py`: refuses private-only GPU-server control paths
- `.gitignore`: keeps private GPU-server run outputs and bridge working copies out of status
- `tools/scrub_public_copy.py`: strips private GPU-server machine runs and lab identifiers from
  exported copies
