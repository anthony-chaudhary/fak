---
title: "The tool process table: mastering long-running tool use"
description: "fak adjudicates a tool call at admission and its payload at re-entry — but a kernel also owns the process table. This note maps the long-running-tool gap (background shells, monitors, polled jobs crossing the floor as uncorrelated events), ships the decision spine (internal/toolproc + fak toolproc), and names each enforcement seam as a labeled next step."
date: 2026-07-02
---

# The tool process table

Status: concept note **plus shipped mechanism at three depths**.
`internal/toolproc` (the pure fold) and `fak toolproc` (the CLI) landed with
this note. Seam 2 is **wired**: `internal/toolprocgate` registers a rank-2
`abi.ResultAdmitter` (defconfig-enabled, inert until a kill) that quarantines
any completion whose call the kernel revoked — witnessed through the real
`kernel.AdmitResult` fold. Seam 1's **engine is shipped**:
`toolprocgate.Supervisor` runs the live journal and Tick loop, and its kill
and reap advice CANCELS the in-flight work via the lever registered at Spawn
and arms the revocation table — the full pipeline (spawn → deadline/orphan →
cancel → post-kill quarantine) is witnessed end-to-end in-process. What
remains of seam 1 is the wire adapter (the gateway/guard observation
plumbing). Seam 4 is **shipped and auto-installed**: `fak toolproc hook
(pre|post|stop)` turns any hook-capable harness's firings into journal
events, and `fak guard` now installs the PreToolUse/PostToolUse/SessionEnd
hooks for Claude children by default (observe mode, fail-open,
`--toolproc-hooks off` to disable) — a default guarded session carries a
live `fak toolproc ps` table with zero setup, and SessionEnd (never Stop,
which fires every turn) marks the orphan boundary. Seam 6 is **wired**:
`Supervisor.BindPID` binds a spawned call to the OS process tree it
launched, and a kill/reap tick terminates the bound tree through
`procguard.KillPID` (`NewOSSupervisor`) in the same act that cancels and
revokes — reap advice has OS teeth once the embedder binds the pid. Seam 5
is **wired**: the manifest's `tool_runtime` block grants each tool its
runtime envelope (deadline + heartbeat cadence, validated fail-loud at
load), and the hook adapter's `--policy` resolves the grant per spawn —
exact row over `*` catch-all, flags filling when no row matches. Seam
3 remains a **labeled next step**. The spine is
offline-provable: `fak toolproc sample` folds a deterministic built-in
journal, no key, no model, no GPU.

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

1. **Gateway supervisor** (`internal/gateway`). **ENGINE SHIPPED** as
   `toolprocgate.Supervisor`: the live journal + Tick loop that folds the same
   pure `toolproc.Fold` and acts — kill/reap advice cancels the in-flight
   work via the cancel lever registered at Spawn, enters the call into the
   revocation table (arming the Gate against its late completion), and
   records the kill in the journal; probe stays advisory. Clock-free,
   goroutine-free, witnessed end-to-end in-process (spawn → deadline → cancel
   → post-kill quarantine through `kernel.AdmitResult`). **Remaining**: the
   wire adapter — the proxy already sees every proposed `tool_use` and
   inbound `tool_result`; it needs to map dispatch onto `Spawn` (registering
   the upstream request's cancel), polls onto `Pulse` (`BashOutput`'s shell
   id is the correlation key), terminal results onto `Exit`, and tick on its
   cadence. The decision journal (`FAK_AUDIT_JOURNAL`) then gains lifecycle
   rows, not just admission rows.
2. **Result-admission rung** (`abi.ResultAdmitter`) — **SHIPPED** as
   `internal/toolprocgate`: a rank-2 admitter (in front of the content screens)
   quarantines a completion whose call id is in the in-process revocation
   table (`toolprocgate.Kill`), citing `TOOL_RESULT_AFTER_KILL`, payload
   stubbed in place and the original bytes dropped (a post-kill payload has no
   legitimate re-entry path). Registered-but-inert by default: with an empty
   table it Defers on every result. Closes failure class 4 in-process; the
   cross-process kill feed rides on seam 1.
3. **MCP wire** (`fak serve --stdio` / `/mcp`). MCP has native
   progress-notification and cancellation semantics; a brokered MCP tool call
   maps 1:1 onto the event vocabulary (progress → `pulse`, cancellation →
   `kill`). fak, fronting the wire, can enforce a deadline the MCP client
   never has to know about.
4. **Harness hooks** (Claude Code). **SHIPPED** in CLI form: `fak toolproc
   hook (pre|post|stop)` reads the hook stdin envelope and appends one
   journal event (pre → `spawn` with the granted envelope, post → `exit`,
   stop → `session_end`; identity = `tool_use_id` with respawn generations
   for repeated identical calls; fail-open — observation never wedges the
   harness). The journal is the same one `fak toolproc ps --events` folds,
   so a hooked session has a live table today: a call that never posts stays
   RUNNING, and session end flags survivors `TOOL_ORPHANED`. **Remaining**:
   `fak guard` auto-installing the three hook lines (it already installs
   PreCompact and Stop), and a pulse source for streamed output.
5. **Policy envelope** (`internal/policy`) — **wired**. Deadline and
   heartbeat cadence live in the capability manifest, per tool: the
   `tool_runtime` block declares `deadline_ms` / `heartbeat_every_ms` rows
   (exact tool name or `*` catch-all; empty, negative, all-zero, and
   duplicate rows refuse at load), and `Runtime.ToolRuntime.EnvelopeFor`
   resolves exact-over-wildcard, nil-safe — absent config grants nothing, so
   an undeclared envelope still folds QUIET, never STALLED. "You may run
   this tool" and "you may run it for this long, reporting at this cadence"
   are one grant: `fak toolproc hook --policy FILE` stamps the resolved
   grant on each spawn event. What remains is the seam-1 wire adapter doing
   the same at gateway dispatch.
6. **procguard bridge** — **wired**. `Supervisor.BindPID(callID, pid)` is
   the tool-call ↔ process-tree binding (self-pid refused; retired on exit,
   prune, or kill), and `NewOSSupervisor` arms `procguard.KillPID` (taskkill
   /T /F on Windows, SIGKILL on POSIX) as the tick's OS lever — invoked once
   per kill/reap, outside the lock, its verdict recorded on the TickAction
   (`reaped`/`reap_detail`; a failed reap never claims success). What remains
   is embedder adoption: the spawn plumbing that knows the pid (the seam-1
   wire adapter, the hook adapter) calls BindPID.

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
