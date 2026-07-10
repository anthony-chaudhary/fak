---
title: "Triage: Windows shell/TUI/console fault boundary (2170)"
description: "Triage of #2170: the Windows shell/TUI/console fault-boundary mechanism already exists in-tree; only four missing regression witnesses remain to split out."
---

# Triage: Windows shell/TUI/console fault boundary (#2170)

Status: **triage / classification only — #2170 stays OPEN.** This note records
the generation-horizon decision and the mechanism-vs-witness gap so the next
dispatcher (Claude or opencode) starts warm instead of re-deriving the map. It
does not resolve the epic.

Date: 2026-07-04. Lane routed: `policy`. Labels: bug, priority/P1, security,
testing, operator.

## The ask (verbatim invariant)

> A terminal, shell, PTY, or TUI render surface is a fault boundary. If that
> surface crashes, loses its console pipe, or exits unexpectedly, fak records a
> structured child/session failure and keeps the kernel/supervisor plus
> unrelated agents alive.

Concrete crash class: a `pwsh.exe` FailFast (`System.Management.Automation.Host.HostException`,
Win32 `0xE9` "No process is on the other end of the pipe", stack through
`Microsoft.PowerShell.ConsoleHost.Start`) correlated with a batch of sessions
dying mid-tool.

## Generation horizon: `gen/now`

Classified from issue evidence, not guessed (the dispatch frame's proof bar):

- It improves **current-product reliability / operator loop** (fleet stability
  on the Windows host), which is the `gen/now` definition in `docs/generation.md`.
- It has a **clear witness path that already exists in-tree** (see mechanism
  below) — no dependency on a future-architecture bet, which would push it to
  `gen/next`/`second-next`.
- Priority (P1) is orthogonal and unchanged; horizon ≠ priority.

Recommended intake repair if operators want it generation-tracked: add
`generation` + `gen/now` and bind milestone *Generation G0 - Now / Immediate*.
Left as a recommendation (not applied) to avoid over-labeling a plain bug and to
keep this session's action reversible; a `needs-triage` hold is **not** warranted
because the horizon is not ambiguous.

### Generation evidence (contract requirement)

- **Promotion evidence** (what would move this closer to done): landing any of
  the four missing witnesses below as a red→green regression test. The mechanism
  is already present, so each witness is a bounded, dispatchable leaf.
- **Demotion / retirement evidence** (what would push it back or close it): if a
  survey shows all four acceptance witnesses already exist under different names
  (e.g. the console-pipe case is already covered by an EOF test in
  `internal/toolprocgate`), the epic retires to "witnessed" and this note is the
  demotion record. That survey has **not** been completed — see the assumption.
- **Invalidating assumption**: this triage assumes the existing
  `toolprocgate`/`windowgate`/`procguard` mechanism is *sufficient* and only the
  *witnesses* are missing. If a witness attempt shows the parent actually dies on
  a lost console handle (rather than folding it to a structured event), the ask
  becomes a mechanism fix, not just a test — and the horizon claim holds but the
  scope estimate below is wrong.

## Mechanism already in tree (evidence)

The fault-boundary machinery mostly exists; this is a *witness + searchability*
epic, not a *build-the-mechanism* epic.

- `internal/toolprocgate/supervisor.go` — `Supervisor` folds child lifecycle
  events (Spawn/Pulse/Exit/SessionEnd) into structured kill/reap actions and
  **holds no goroutine and reads no clock**, so a child death cannot take the
  parent down. Directly serves "parent must keep draining/reaping children."
- `internal/toolprocgate/output.go` — `AdmitChildOutput` routes child
  stdout/stderr/structured bytes through result admission and **quarantines**
  refused/broken output into a bounded stub rather than letting raw bytes crash a
  parent-visible surface. Directly serves "broken pipes/EOF become structured
  events."
- `internal/windowgate/*.go` — `ConfigureBackgroundCommand` (CREATE_NO_WINDOW),
  `conhost --headless` recognition, and the Go/PS/Python spawn-suppression
  ratchet + live process/window/task classifiers. Serves "prefer hidden/headless
  background launch primitives."
- `internal/procguard/procguard.go` — process-tree descendant reaping (commit
  `c156d11c`). Serves "child-process reaping."
- `cmd/fak/guard_child.go` (peer, **uncommitted** in the shared tree as of this
  note) — `finishGuardChildAndReport` surfaces a child crash as a structured exit
  report, runs a terminal-restore pulse, tears the gateway down cleanly, and
  prints `guardToolprocSummary`. Do **not** duplicate this; coordinate.
- `cmd/fak/guard_toolproc_summary.go` + `fak toolproc ps --events` — a durable
  status surface that folds stalled tool processes to `TOOL_HEARTBEAT_STALLED`.

## Acceptance criteria vs. state (residual)

| # | Acceptance witness | State | Owning lane |
|---|---|---|---|
| 1 | Child shell/console **pipe failure** → parent survives + records child failure | Mechanism present (`AdmitChildOutput`, `Supervisor`); **no witness** confirmed simulating a broken console pipe / PTY EOF | `toolprocgate` |
| 2 | **TUI/render-pane failure** → supervisor + siblings survive | Partial: `TestWatchdogAutohealKeepsAgentPaneClean` + terminal-restore pulse exist; **no witness** that a render-pane crash spares the supervisor and sibling sessions | `cmd` (tui) |
| 3 | Windows **hidden/headless launch + child reaping** tests | Partial: `windowgate/exec_windows_test.go`, `procguard` tree-kill; confirm CREATE_NO_WINDOW background-launch + reaping are covered together on Windows | `windowgate`, `procguard` |
| 4 | Crash class **searchable in a durable status surface** | Partial: `fak toolproc ps --events` surfaces stalled tool procs; the specific `pwsh` HostException / `0xE9` console-pipe FailFast class is **not** mapped into a fak surface (still Event Viewer only) | `toolproc` / `windowgate` + status surface |

Also architectural (larger than one leaf): "the TUI should be an attachable
client/view over state, not the owner of the long-lived supervisor/agent process
tree." `cmd/fak/tui.go` is under active peer edit; this is a refactor, not a
test, and should be its own child issue.

## Smallest next step

Split #2170 into four dispatchable child witnesses (one per row above), each a
red→green regression test against the *existing* mechanism, routed to the owning
lane in the table — plus one architecture child for the TUI-ownership refactor.
Row 1 (toolprocgate console-pipe/EOF witness) is the highest-value first leaf:
it is the exact crash class in the report and the mechanism to prove is already
there.

## Why this session did not land a code fix

- The acceptance witnesses live in `toolprocgate`, `cmd` (tui), `windowgate`,
  and `procguard` — **outside** the leased `policy` lane; a witnessed commit
  there needs the matching `(fak <leaf>)` trailer, not `(fak policy)`.
- The dispatch frame scoped this to **triage only** (classify the horizon before
  implementation).
- Native `go test` is blocked on this Windows host (AGENTS.md — OS
  Application-Control policy); a witness would have to be gated under WSL/CI.
- Peer work is already in flight in `cmd/fak/guard_child.go` and
  `cmd/fak/toolproc.go`; duplicating it risks a `PATHSPEC_RACE` on the shared
  trunk.
