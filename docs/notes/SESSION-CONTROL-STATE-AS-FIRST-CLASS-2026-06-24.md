---
title: "fak note: session control state as a first-class, queryable value (2026-06-24)"
description: "A design for making a served session's DRIVE state — run-state, planner budget, priority, pace — a first-class value that is read each turn and written live, instead of a thing re-derived or guessed. Grounded in the existing TraceID + ifc.Ledger + /v1/fak/trace seam, which already proves the per-session, live-mutable, queryable pattern for exactly one bit."
---

# Session control state as a first-class value

> Date: 2026-06-24.
> Scope: a design + the in-repo seam it generalizes. Nothing here is shipped; §7 is the
> honest fence. It treats the *drive* state of a served session (how hard, how fast, how far
> to keep going) the way [`O1-TURN-CONTEXT-PLANNER`](O1-TURN-CONTEXT-PLANNER-2026-06-23.md)
> treats the *context* of a turn: a value you read, not a thing you reconstruct.

## 0. The idea in one paragraph

A long agent session has a control state that changes while it runs: a planner budget, a
priority, a pace, and a run-state (running / throttled / paused / draining / stopped). Today
that state is **implicit** — `maxTurns` is fixed at entry, the budget is resolved once at
init, and "is this session still going?" is reconstructed after the fact from git commits, a
process scan, and a 0-byte log. An operator who wants to *dial a session down* mid-flight —
drop its budget, lower its priority, pause it, let an urgent one pass — has nowhere to write
that intent, and nothing reads it. The fix is to make the session's drive a **first-class,
TraceID-keyed value**: one small record per live session, written live through a route, read
by the turn loop at each turn boundary. The current state is then a *lookup*, never a
re-derivation — "smooth to know at any moment," because the moment is a field, not a guess.

This is the same move the repo already made for *taint*. `ifc.Ledger` is a TraceID-keyed,
bounded-LRU, concurrent, live-mutable per-session store with a `GET /v1/fak/trace/{id}` read
and a `POST /v1/fak/trace/reset` write. It carries exactly **one** value (the taint
high-water mark). Session control state is that exact seam, widened from one bit to a small
struct.

## 1. Why "re-derive" is the bug, not a missing feature

The dispatch loop ([`dispatch-loop.md`](../dispatch-loop.md)) is honest that it has no live
session state: live-worker count is `MAX(kernel lease count, OS process scan)`, progress is
git commits, and the operator view is a 0-byte-log scan. That is a *reconstruction* layer,
and it is load-bearing precisely because the running session exposes no state of its own. The
doc even names the failure mode: a `claude -p` worker buffers stdout, so its log is 0 bytes
until it exits — "don't read 0-byte log as did-nothing." The robust signal had to be git,
*because the session itself answers nothing while live*.

Reconstruction has three costs the goal is reacting to:

- **It is lossy.** A 0-byte log is ambiguous across {still running, hung, killed, produced
  nothing}. Git commits tell you a session *shipped*, never how hard it is currently trying.
- **It is racy.** Two readers (the status doc, the watchdog) re-derive independently and can
  disagree — the `dos_status` digest exists specifically because peers were each guessing.
- **It is read-only.** You can *observe* a reconstructed state; you cannot *set* it. There is
  no way to say "this session, from now on, plans under half the budget" — the thing you'd
  write to does not exist.

A first-class value fixes all three at once: it is unambiguous (the field says what it is),
single-source (everyone reads the same record), and writable (setting it IS the control).

## 2. The state, named

One record per live session, keyed by TraceID. Four orthogonal axes — deliberately small,
because a control surface nobody can hold in their head gets ignored:

| Axis | Type | Meaning | Who writes it |
|---|---|---|---|
| **RunState** | enum | `RUNNING` · `THROTTLED` · `PAUSED` · `DRAINING` · `STOPPED` | operator / supervisor / the session itself (on drain) |
| **Budget** | `{TurnsLeft, TokensLeft int}` | remaining work allotment, **decremented each turn** | the loop debits; operator can *re-set* (raise or cut) live |
| **Priority** | int | scheduling rank; lower = yields first under contention | operator / supervisor |
| **Pace** | `{MaxTokensPerTurn, MinTurnGapMs int}` | per-turn throttle (slow a session without pausing it) | operator |

The run-state is a small, total state machine, and the transitions are the verbs the goal
asked for:

```text
            ┌──────────── speed-up: raise Budget / Pace / Priority ───────────┐
            ▼                                                                  │
  RUNNING ──throttle──► THROTTLED ──pause──► PAUSED ──resume──► RUNNING        │
     │  │                   │                   │                              │
     │  └──── resume / clear-throttle ──────────┘                              │
     │                                                                         │
     ├── budget exhausted / operator stop ──► DRAINING ──(turn boundary)──► STOPPED
     └─────────────────────────────────────────────────────────────────────────┘
```

`DRAINING` is the load-bearing nuance: a stop is requested at any instant but **taken at a
turn boundary**, never mid-decode — so a stop never corrupts a half-emitted tool call or a
mid-flight KV mutation. The session keeps the `STOPPED` reason as a value (a closed token, the
same discipline as the [refusal vocabulary](../../POLICY.md)), so "why did it stop" is a
field, not an inference from an exit code. This mirrors how the kernel takes a poison
quarantine on the `<|im_end|>` boundary in `EvictPoisoned`, never mid-token.

The **priority-queue** framing the goal raised falls straight out: `Priority` + `Budget` are
exactly the key a supervisor scheduler needs. "Reduce a session's budget part-way through to
let an urgent one pass" is `POST /v1/fak/session/{id}/budget {turns_left: 3}` plus a
`priority` cut — no restart, no re-derivation, and the very next turn honors it.

## 3. The seam it generalizes (already on the live path)

The mechanism is not speculative — it is `ifc.Ledger` with a wider value. The proven pieces:

| Proven today (taint, one bit) | Generalized (drive state, a struct) |
|---|---|
| `ifc.Ledger`: TraceID-keyed, `sync.RWMutex`, bounded-LRU (`DefaultLedgerLimit=8192`) | `SessionTable`: same keying, same bound, same concurrency posture |
| `Ledger.Level(trace)` read · `Raise` / `Reset` write | `Table.Get(trace)` read · `Set*` / `Debit` / `Transition` writes |
| `GET /v1/fak/trace/{id}` → `TraceObserveResponse{TraceID,Taint,Dangerous}` | `GET /v1/fak/session/{id}` → `SessionStateResponse{TraceID,RunState,Budget,Priority,Pace}` |
| `POST /v1/fak/trace/reset` → `TraceResetResponse{Reset,TraceID}` | `POST /v1/fak/session/{id}/{verb}` → `SessionStateResponse` (the new state) |
| `traceFor()` mints `gw-<n>` per session (`gateway.go:707`) | the SAME id is the session key — no new identity invented |
| route disabled ⇒ 404 (never a silent clean reading) | identical fail-closed posture: no table ⇒ 404 |

The cap matters and it is already solved: `DefaultLedgerLimit` exists *because* "gateways mint
a non-empty TraceID per served session, so a long-running process must not retain every
historical trace forever." The session table inherits that exact bound — the most-recently
touched N sessions stay resident; an evicted one is `RUNNING` with a full default budget on
next touch (the safe default, never a phantom `STOPPED`).

A Go shape, deliberately small and stdlib-only, sitting beside `ifc.Ledger` on the gateway
`Server` (which today holds no session map — `gateway.go:213`):

```go
// internal/session — the per-session DRIVE state, keyed by TraceID. Same posture as
// ifc.Ledger: bounded-LRU, RWMutex, the empty key is the single-session default.
type RunState uint8
const (
    Running   RunState = iota // the default for an unseen trace
    Throttled                 // pace-limited, still advancing
    Paused                    // holds at the next turn boundary
    Draining                  // stop requested; taken at the next boundary
    Stopped                   // terminal; carries a closed StopReason token
)

type Budget struct{ TurnsLeft, TokensLeft int } // -1 == unbounded (the v0.1 default)
type Pace struct{ MaxTokensPerTurn, MinTurnGapMs int }

type State struct {
    Run      RunState
    Budget   Budget
    Priority int
    Pace     Pace
    Reason   string // closed token on Stopped/Throttled; "" otherwise
    Rev      uint64 // monotonic; bumped on every write (the optimistic-concurrency guard)
}

// Decide is the ONE call the turn loop makes at each boundary. Pure given the state:
// it debits the turn, applies pace, and returns whether to proceed — no re-derivation.
func (t *Table) Decide(trace string) (proceed bool, st State)
```

`Rev` is what makes a write a control and not a race: a `POST` may carry an `If-Rev` so a
stale operator UI cannot clobber a newer transition, and `/v1/fak/changes` (the existing
cursor feed, `coherence.go:16`) can stream drive-state revisions the same way it streams cache
changes — a live "what is every session doing right now" tail for free.

## 4. Where the loop reads it (one call, one boundary)

The whole point is that the loop asks *once per turn* and the answer is a lookup. In
`internal/agent/loop.go` the loop is `for turn := 0; turn < maxTurns; turn++` with `maxTurns`
frozen at entry. The change is a single guard at the boundary — the budget/pace/run-state
become the loop condition instead of a fixed integer:

```go
for {
    proceed, st := sessions.Decide(trace) // O(1) map read + debit, under the table lock
    if !proceed {                          // PAUSED holds; DRAINING/STOPPED exits cleanly;
        return finalize(st)                //   budget-exhausted exits with the reason token
    }
    if st.Pace.MinTurnGapMs > 0 { throttle(st.Pace) }
    comp, err := p.Complete(ctx, messages, tools, maxTokens(st.Pace)) // pace caps THIS turn
    // ... existing turn body, unchanged ...
}
```

Three properties this buys, each answering a clause of the goal:

- **Stop / halt is clean and instant-to-request, boundary-to-take.** An operator `POST`s
  `DRAINING` at any millisecond; the loop sees it at the next `Decide` and exits without a
  torn turn. No `kill -9`, no half-written commit.
- **Speed-up / slow-down need no restart.** Raise `Budget`/`Pace`/`Priority` (speed up) or cut
  them (slow down); the *next* turn honors it. The session never leaves the loop.
- **Resume is a state flip, not a cold re-attach.** For a live session, `PAUSED → RUNNING` is
  a write — distinct from the *cold* resume the repo already has (re-attaching a persisted
  ctxplan core image at process start, `ctxplan/image.go`). Naming the two apart removes a
  real ambiguity: "resume" today only ever means cold-start re-attach.

The `Decide` call also unifies the two budgets that exist but never compose: the matmul-cores
`FAK_BUDGET` (`internal/model/budget.go`, static, process-wide) and the per-turn ctxplan
window (`SessionPlanner.Budget`, per-session but set once). `Pace.MaxTokensPerTurn` can drive
the ctxplan budget *down* for a throttled session, so "slow this one session" finally has a
single knob instead of a process-global one and a set-once one.

## 5. Persistence (so a resumed session re-attaches its drive, too)

A cold resume already reloads the context (the ctxplan index next to the recall core image,
`recall.PersistIndex`). The drive state rides the same image: one more small sibling
(`session.json` beside `manifest.json` / `index.json`), so a process restart re-attaches a
session at the budget/priority/run-state it held, not a default. A `STOPPED` session reloads
as `STOPPED` (with its reason) — it is not silently resurrected as `RUNNING`. This is the
honesty rung: the persisted drive is a fact, re-checkable, the same way
[`dos_recall`](../../README.md) re-verifies a memory against ground truth instead of trusting
the body.

## 6. What this is NOT (so it stays small)

- **Not a scheduler.** The table holds `Priority`; it does not *act* on it. A supervisor reads
  the table and decides who yields — the same split as the dispatch loop, where the kernel
  holds loop state and the always-on task acts. Keeping policy out of the table is what keeps
  the table a *value*.
- **Not the audit plane.** [`hosted-control-plane.md`](../fak/hosted-control-plane.md) is a
  read-side aggregator over what *already happened* (deny/allow verdicts, reason tokens). This
  is a read-**write** surface over what happens *next*. They meet at the TraceID join key but
  never overlap: the audit plane reports decisions; the session table sets drive.
- **Not a second source of truth for taint.** Taint stays in `ifc.Ledger` (it has its own
  monotonic-restrictiveness semantics). The session table is the *drive* axes only; it links
  to the taint mark by shared TraceID, it does not duplicate it.

## 7. Honest fences (what is unbuilt)

- **Nothing here is shipped.** `internal/session` does not exist; the gateway holds no session
  map (`gateway.go:213`); `loop.go` still runs to a fixed `maxTurns`. This note is the design
  + the proof that the seam to build it on (`ifc.Ledger` + `/v1/fak/trace` + `traceFor`) is
  already live and load-bearing — so the build is *widen an existing pattern*, not invent one.
- **The loop the design wires is the A/B harness loop, not the flagship serve path.**
  `loop.go` is the benchmark harness; the served gateway turn is the `req.Raw` passthrough that
  the ctxplan seam is also still gated behind (`#555`). The session guard lands on the harness
  loop first (where it is testable end-to-end), and threads the gateway turn second — the same
  "seam first, gateway second" sequencing the ctxplan note took.
- **Priority is a field, not yet a contended scheduler.** Until a supervisor reads it, lowering
  a session's priority does nothing on its own — there is one process, one loop. The value is
  real (it persists, it streams on `/v1/fak/changes`); the *yield* it implies needs the
  multi-session supervisor that does not exist in Go yet (it is DOS-kernel today, external).
- **The two real budgets compose only on paper here.** Wiring `Pace.MaxTokensPerTurn` into
  `SessionPlanner.Budget` and `FAK_BUDGET` is described in §4 but not implemented; the static
  `model/budget.go` doc explicitly rejected live-load sensing, so making it per-session live is
  a deliberate, measured change, not a free one.
- **Boundary-only stop assumes a turn is short.** A `DRAINING` session with a long decode still
  finishes that decode before exiting. That is the *correct* trade (no torn turn) but it means
  "stop" is not instantaneous — it is "stop at the next boundary," and the design owns that
  word honestly rather than promising a mid-token kill.

## Related

- [`O1-TURN-CONTEXT-PLANNER-2026-06-23.md`](O1-TURN-CONTEXT-PLANNER-2026-06-23.md) — the same
  "read a value, don't re-derive it" move, applied to a turn's context instead of a session's
  drive. The `SessionPlanner.Budget` this composes with lives there.
- [`dispatch-loop.md`](../dispatch-loop.md) — the reconstruction layer this would replace for
  live sessions (git-commit progress, 0-byte-log scan, `MAX(lease, process)` liveness).
- [`hosted-control-plane.md`](../fak/hosted-control-plane.md) — the read-side audit plane; the
  write-side peer this note describes meets it at the `X-Trace-Id` join key.
- `internal/ifc/ifc.go` (`Ledger`) · `internal/gateway/gateway.go` (`traceFor`,
  `TraceResetResponse`) — the live seam this generalizes.
