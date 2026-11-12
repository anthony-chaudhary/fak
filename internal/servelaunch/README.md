# internal/servelaunch

The readiness / teardown / remote-inspection half of launching a third-party
server for a benchmark: you start a process on a host you are not sitting in
front of, you wait for it to answer, you look at it over ssh while it runs, and
you stop it cleanly afterwards. Every step is honest about what it actually
observed.

The package is a clean-room conceptual adaptation of the pinned Benchmark
Python launcher's SSH/`screen` subsystem (study:
`docs/notes/CONCEPT-STUDY-BENCH-SGLANG-DEPLOYMENT-2026-11-09.md`, §A). No source
is copied; the behavior contracts come from the study's pinned citations.

## What lives here

| File | Concern |
| --- | --- |
| `readiness.go` | The readiness wait: poll a health URL with a deadline, emit heartbeats, and race a crash side-channel so a fast crash returns in seconds instead of riding the timeout. Warm/cold budget from a cache-state probe. |
| `teardown.go` | The escalating, idempotent stop ladder: SIGTERM → bounded exit poll → SIGKILL → port-kill safety net. Rollback-in-reverse for partial launches. |
| `port.go` | Fail-closed port claiming: a port is accepted only on a positively confirmed free probe; an inconclusive probe is busy. |
| `sshargv.go` | The hardened ssh/scp argv builder. BatchMode, no host-key mutation, forwarding escapes closed; a password stance that **never puts a secret in argv**. |
| `screen.go` | GNU-screen session builders and parsers: detached launch with log capture, the keep-alive wrapper and its `[Exited $?]` crash witness, exact-name `screen -ls` resolution, the bounded crash-log tail. |
| `receipt.go` | The launch receipt: the serializable record of one launch (screen name, log path, command, lifecycle, crash log) plus the operator banner with the `screen -r` reattach line and an atomic JSON `Save`. |
| `remote.go` | The transport that runs the built commands over ssh: probe availability, list/resolve sessions, tail a log, launch a session, and `Inspect` — the one call that produces a receipt. |

Everything time-ordered and every process goes through an injected seam
(`Clock`, `Ticker`, `ExecRunner`), so the whole package is tested without a
clock, a network, or a spawned process.

## Operator use

`fak-dev remote-launch` drives the transport directly:

```
fak-dev remote-launch inspect --host gpu-box-a --server s1 --save ./receipts/
fak-dev remote-launch start   --host gpu-box-a --server s1 --command 'python -m sglang.launch_server --port 30000'
```

`inspect` prints the reattach banner and writes the receipt JSON; `start` first
launches the command (and optionally the paired hardware monitor) under a
detached screen session, then inspects it.
