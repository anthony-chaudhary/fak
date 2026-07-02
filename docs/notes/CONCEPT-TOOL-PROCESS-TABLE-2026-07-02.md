---
title: "The tool process table: mastering long-running tool use"
description: "fak adjudicates a tool call at admission and its payload at re-entry — but a kernel also owns the process table. This note maps the long-running-tool gap (background shells, monitors, polled jobs crossing the floor as uncorrelated events), ships the decision spine (internal/toolproc + fak toolproc), and names each enforcement seam as a labeled next step."
date: 2026-07-02
---

# The tool process table

Status: concept note **plus a shipped decision spine**. `internal/toolproc` (the
pure fold) and `fak toolproc` (the CLI) land with this note; every enforcement
seam below is a **labeled next step** — nothing kills, cancels, or quarantines
anything today. The spine is offline-provable: `fak toolproc sample` folds a
deterministic built-in journal, no key, no model, no GPU.

## The problem: the syscall got a lifetime

fak's thesis is "treat the tool call like a syscall: the model proposes, the
kernel disposes." Today the kernel disposes at exactly two instants:

1. **admission** — `abi.Adjudicator` folds the capability floor over the
   proposed call before dispatch (`internal/kernel.Submit`);
2. **result re-entry** — the `abi.ResultAdmitter` chain screens the payload
   before it enters model context (`internal/kernel.Reap`, the gateway's
   inbound-result floor).

Between those two instants a tool call is invisible. That was the right model
when every call was a sub-second request/response. It is the wrong model now:
harnesses run background shells (`Bash(run_in_background)`), file/process
monitors, subagents, remote jobs — tool calls that live minutes to hours, emit
streams, and outlive their turn, sometimes their session.

Concretely, on today's floor:

- `run_in_background`, `BashOutput`, and `KillShell` are just **allow-listed
  tool names** (see `cmd/fak/guard-default-policy.json`,
  `examples/presets/coding-agent-safe.json`). Each crosses the floor once, as
  an **independent, uncorrelated event**. Nothing links the adjudicated launch
  to the job it spawned or to the later polls that deliver its output.
- Nothing models "this call is still running," so nothing can enforce a
  runtime deadline, notice a stall, or reap an orphan at tool-call
  granularity. All existing watchdog machinery sits one or more levels up:
  `internal/procguard` (OS processes), `internal/taskmgr` (fak's own process
  tasks), `internal/timeoutphase` (worker post-mortems), `fak guard
  --max-duration` and the resume watchdog (sessions/runs), `fleetmon` (fleet).
- The no-babysitting doctrine
  ([CONCEPT-NO-BABYSITTING-2026-07-01](CONCEPT-NO-BABYSITTING-2026-07-01.md))
  frames its liveness watch at run/session/fleet granularity and explicitly
  observes that harnesses are absorbing background tasks; it proposes no
  mechanism at the individual-tool-call level. This note is that missing row.

A kernel that gates entry but does not own the process table is a doorman, not
a kernel. After `exec`, a real kernel owns PIDs: state, signals, rlimits,
wait/reap, orphan handling. The tool-call analogue is what this note ships the
spine of.

## The failure classes (why this is a security AND a cost surface)

1. **Runaway call** — a tool call that never returns burns wall-clock, tokens,
   and (for local tools) the host. Today's only backstops are session-level
   (`--max-duration`) or host-level (`procguard`), both far coarser than the
   call that misbehaved.
2. **Silent stall** — a background job that stops producing is
   indistinguishable from one that is working. Silence looks like progress
   (the same failure class the Monitor/watchdog docs warn about one level up).
3. **Orphan leak** — the owning session dies; its spawned jobs keep running.
   This class is real here: guarded-session children crashing when the parent
   exits, and the 2026-06-21 runaway-`find` incident that held 98.8% of system
   handles (AGENTS.md hard rules) are both orphan-shaped.
4. **Post-kill admission** — a call is revoked mid-flight, but its completion
   arrives later and its payload enters context as if the call were live. The
   result floor screens *content* (secrets, poison); nothing screens
   *provenance-in-time* ("this payload belongs to a call the kernel already
   revoked").
5. **Launch↔poll uncorrelation** — a launch adjudicated as harmless returns
   its real output through later polls that are screened as fresh, unrelated
   results, with no linkage back to the envelope the launch was granted.

## The shipped spine: `internal/toolproc`

A tool call that goes long becomes a **tool process**: a kernel object with an
owner (session), a declared runtime envelope granted at admission (deadline +
heartbeat cadence), observed liveness signals, and a terminal transition.

- **Journal events** (append-only JSONL, closed vocabulary, fail-closed
  parse): `spawn` (call id, tool, owner, envelope), `pulse` (any liveness
  signal — heartbeat, output chunk, progress, poll; optionally correlated to
  the polling call via `via`), `exit` (ok|error), `kill` (citing a closed
  reason token), `session_end` (the orphan boundary).
- **`Fold(events, now, config)`** is a pure function to the process table at
  one instant: per-proc state (`RUNNING`/`DONE`/`KILLED`), liveness vs the
  declared cadence (`LIVE`/`QUIET`/`STALLED` — undeclared cadence folds to
  QUIET, honest-not-green), deadline overdue-ness, orphan-ness. Benign races
  are tolerated and counted (a late pulse, an exit after a kill, a kill that
  lost to completion); impossible transitions refuse the fold (duplicate
  spawn, events for a never-spawned call, a spawn owned by an ended session —
  the orphan-leak class at its source).
- **Closed verdict vocabulary**, reserved in the registered out-of-tree
  reason range (1040–1043, the `egressfloor` pattern; the consumer registers
  the names):

  | token | advice | failure class |
  |---|---|---|
  | `TOOL_DEADLINE_EXCEEDED` | `kill` | runaway call |
  | `TOOL_HEARTBEAT_STALLED` | `probe` | silent stall |
  | `TOOL_ORPHANED` | `reap` | orphan leak |
  | `TOOL_RESULT_AFTER_KILL` | `quarantine_result` | post-kill admission |

- **CLI**: `fak toolproc ps --events <jsonl> [--now-unix-ms N]
  [--default-deadline-ms N] [--stall-mult F] [--json]` renders the table and
  exits 0 (clean) / 3 (attention advised — gate-able) / 1 (refusal) / 2
  (usage). `fak toolproc sample [--json|--journal]` is the deterministic
  offline proof, one row per verdict class.

Same events + same instant + same config ⇒ byte-identical table (tested), so a
supervisor tick, a CI gate, and a forensic replay all read the same truth.

## Enforcement map: the deeper integration points (all labeled next steps)

Each seam below already exists; toolproc gives it a shared clock, vocabulary,
and advice stream. None of this is wired yet.

1. **Gateway supervisor** (`internal/gateway`). The proxy already sees every
   proposed `tool_use` and every inbound `tool_result`. Emitting `spawn` when
   an adjudicated call is dispatched, `pulse` when a poll references the job
   (`BashOutput`'s shell id argument is the correlation key), and `exit` when
   its terminal result crosses — then folding on a tick and acting on advice
   (cancel the upstream request, refuse the next poll, annotate the turn) —
   makes the table live. The decision journal (`FAK_AUDIT_JOURNAL`) gains
   lifecycle rows, not just admission rows.
2. **Result-admission rung** (`abi.ResultAdmitter`). A completion whose call
   the table shows as `KILLED` is `TOOL_RESULT_AFTER_KILL`: quarantine the
   payload (`VerdictQuarantine{PageOut}`), never admit it as live. This closes
   failure class 4 with machinery the kernel already has.
3. **MCP wire** (`fak serve --stdio` / `/mcp`). MCP has native
   progress-notification and cancellation semantics; a brokered MCP tool call
   maps 1:1 onto the event vocabulary (progress → `pulse`, cancellation →
   `kill`). fak, fronting the wire, can enforce a deadline the MCP client
   never has to know about.
4. **Harness hooks** (Claude Code). `fak guard` already installs PreCompact
   and Stop hooks; a PreToolUse/PostToolUse pair emitting
   `spawn`/`pulse`/`exit` extends the same install path to harnesses where
   fak is a hook, not a proxy.
5. **Policy envelope** (`internal/policy`). Deadline and heartbeat cadence
   belong in the capability manifest, per tool — the runtime envelope granted
   at admission alongside the capability itself. "You may run this tool" and
   "you may run it for this long, reporting at this cadence" are one grant.
6. **procguard bridge**. `reap` advice has teeth only if the tool process is
   bound to its OS process tree (job objects / process groups) at spawn.
   `internal/procguard` already knows how to classify and reap at the OS
   level; the missing piece is the tool-call ↔ process-tree binding.

## Honesty fences

- **Witnessed today**: the pure fold, its closed vocabulary, its determinism,
  and the CLI — by `go test ./internal/toolproc` and the byte-stable sample.
- **Not witnessed**: any live effect. No process is killed, no request
  cancelled, no payload quarantined by this commit. The sample is a fixture,
  not a measurement. Per the net-true-value standard, no efficiency or safety
  number is claimed for the spine alone.
- The event stream is only as honest as its producer. Once the gateway emits
  events (seam 1), the journal inherits the audit chain's tamper-evidence;
  a hand-authored journal proves nothing beyond the fold's behavior.

## Relationship to existing surfaces (one line each)

| surface | granularity | what it watches |
|---|---|---|
| `internal/procguard` | OS process | runaway CPU/threads/handles, orphan sprawl |
| `internal/taskmgr` | fak's own process | task/step progress, ETA, liveness |
| `internal/timeoutphase` | dispatch worker | post-mortem phase of a timeout |
| `fak guard --max-duration`, resume watchdog | session/run | wall-clock budget, stalled runs |
| `internal/fleetmon` | fleet | janitor over worker process trees |
| **`internal/toolproc` (this note)** | **tool call** | **deadline, stall, orphan, post-kill admission** |
