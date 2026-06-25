---
title: "Live session status & control"
description: "Read what a served session is doing right now, and cancel or update it in flight — the /v1/fak/session(s) routes and the `fak session` operator CLI over the per-session DRIVE-state table."
---

# Live session status & control

A long agent session has a *drive* state that changes while it runs: a run-state
(running / throttled / paused / draining / stopped), a remaining work **budget**, a
scheduling **priority**, and a per-turn **pace**. fak keeps that state as a
first-class, TraceID-keyed value (`internal/session.Table`, the structural twin of the
IFC taint ledger), so the current state is a *lookup*, not a thing reconstructed after
the fact from git commits and a process scan. This page is how you read it and change
it from outside the session.

Design background: [`docs/notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md`](../notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md).

## The routes

| Method · path | What it does |
|---|---|
| `GET /v1/fak/session/{id}` | observe one session's drive state |
| `GET /v1/fak/sessions` | snapshot **every** live session (priority order) |
| `POST /v1/fak/session/{id}/run` | set the run-state (`{"run":"paused"}`, `"stopped"`, …) |
| `POST /v1/fak/session/{id}/budget` | re-set the allotment (`{"budget":{"turns_left":3,"tokens_left":-1}}`) |
| `POST /v1/fak/session/{id}/pace` | re-set the per-turn throttle (`{"pace":{"max_tokens_per_turn":256}}`) |
| `POST /v1/fak/session/{id}/priority` | re-set the scheduling rank (`{"priority":7}`) |

Every write echoes back the new state, including a monotonic `rev`. A write may carry
`"if_rev": N` — an optimistic-concurrency guard so a stale controller cannot clobber a
newer change; a lost race returns **409** (re-read and retry). A stopped session is
terminal: every control verb on it returns 409 (you start a new session, you do not
un-stop one). The routes are **fail-closed**: a gateway with no session table returns
**404**, never a silent clean reading.

## The CLI

`fak session` is the operator front end to those routes — read status, and cancel or
update a session in flight:

```sh
fak session ls                              # every live session
fak session status   <id>                   # one session's drive state
fak session stop     <id> --reason oom      # request a clean stop (taken at the next boundary)
fak session pause    <id>                   # hold at the next turn boundary
fak session resume   <id>                   # un-pause (a live state flip)
fak session throttle <id> --reason "yield to urgent"
fak session run      <id> draining          # any run-state
fak session budget   <id> --turns 3         # cut the allotment to let an urgent one pass
fak session pace     <id> --max-tokens 256 --gap-ms 200
fak session priority <id> 7
```

Connection: `--addr` (default `$FAK_ADDR` or `http://127.0.0.1:8080`) and `--key`
(`$FAK_KEY`) — a loopback gateway with no `--require-key` needs neither. `--json`
emits the raw record; `--if-rev N` fences a write. A partial `budget`/`pace` update
(naming only one axis) reads the current state first, preserves the axes you did not
name, and fences the read-modify-write with the observed rev so a concurrent change is
caught rather than clobbered.

## What "cancel in flight" means today

The control surface writes the per-session DRIVE table; **fak's own agent turn loop**
(`agent.RunArm`, via `WithSessionTable`) reads it at each turn boundary, so a `stop` is
taken cleanly at the next boundary — never mid-decode, so it cannot tear a half-emitted
tool call. The stop carries its reason as a value (`StoppedBySession`), so "why did it
stop" is a field, not an inference from an exit code.

Honest fences (the follow-on epic): the gating is wired into the agent harness loop
first, not yet the flagship proxied serve path; `priority` is recorded but not yet
consumed by a contended scheduler; and there is no cross-restart persistence of the
drive state yet. See the design note's status section for the full list.
