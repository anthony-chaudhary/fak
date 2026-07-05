---
title: "Out-of-band operator control: a closed control vocabulary for a running session"
description: "Design note + SOTA survey for the out-of-band operator control plane — replacing freeform prose mid-run with a closed, typed, witnessed set of control operations on a live session's drive state. Benchmarks fak's shipped surface (fak session / fak signal / the a2achan bus) against the ~15 canonical control ops the field converged on, and sequences the gap-closing epic."
---

# Out-of-band operator control — a closed control vocabulary for a running session

Once a session has started, how does a human change what it is doing? Today the
usual answer is: type more prose into the same channel the agent reads as its task.
That is *in-band* steering, and it has three structural problems. It is **slow**
(you wait for a turn boundary and hope the model re-reads the new instruction). It
is **ambiguous** (prose is interpreted, not applied — "actually, stop touching the
auth module" may or may not land). And it is **injection-shaped**: a steering
instruction delivered as task text is indistinguishable from task *content*, which
is precisely the confusion fak's kernel exists to prevent.

The alternative the rest of computing settled on decades ago is **out-of-band
control**: a separate, addressable channel carrying a closed set of *typed
operations* to a specific live execution, distinct from the data the execution is
working on. TCP's urgent pointer, Unix signals (`SIGSTOP`/`SIGCONT`/`SIGTERM`,
`nice`, `setrlimit`), and Temporal's signals/queries/updates are all the same shape:
control that is not injected as more input.

This note expands that concept for agent loops, surveys where the field is (2025–26),
benchmarks what fak already ships against it, and sequences the work to close the
gap. The one-line thesis: **fak already owns the hard parts of an out-of-band control
plane; the frontier is a closed, structured control *vocabulary* to replace the
freeform `steer` escape hatch, plus the operator surfaces to drive it and the
witness that each op was actually applied.**

## Why fak is unusually well-positioned

The control-plane / data-plane split is not a new subsystem fak has to bolt on — it
is fak's founding thesis (the kernel sits *between* the agent and its tools and
adjudicates every call). fak already ships every structural precondition the field
says out-of-band control requires:

- **A durable, addressable session drive-state** — `internal/session.Table`, an
  OS-process-control-block by analogy: keyed by `X-Trace-Id`, LRU-bounded, with a
  closed run-state machine (`Running / Throttled / Paused / Draining / Stopped`), a
  `Priority`, and a `Budget{TurnsLeft, TokensLeft, ClarificationQueriesLeft}`. This is
  the "durable checkpointed state keyed by a session id" every control plane is built
  on.
- **A separate, capability-gated control channel** — `internal/a2achan`, a
  taint-aware message bus keyed by `(Locale, ID)`. Operator steers ride the
  `Session`-locale mailbox keyed by the run's trace, fail-closed without `CapA2ASend`,
  and are screened on ingress. Control never travels as task tokens.
- **Turn-boundary application** — the owned loop (`RunArm` + `SessionGate` /
  `WithSessionTable`) re-`Decide`s at each turn boundary, parks on a hold, and
  splices a drained steer into the *same* turn. Pause is turn-granular (it resumes
  the held turn, it does not restart the run — witnessed, #1321).
- **A closed refusal vocabulary + witness discipline** — the exact culture a control
  plane needs: an op should refuse with a structured reason (you cannot cancel a
  `Stopped` session), and "enqueued" must be distinguished from "applied" (the steer
  test already asserts *spliced-into-the-turn*, not merely *queued*).

## The state of the art (2025–26)

The field is converging on the same architecture from many directions. The survey
below is condensed; each system expresses the same operations in its own dialect.

- **LangGraph** — the most complete pause/inspect/edit/resume/redirect model:
  `interrupt()` (park at a node, surface a value), `Command(resume=…)` (resume with a
  structured value), `Command(goto=…)` (redirect to another node / `__end__` to abort
  a branch), `Command(update=…)` and `graph.update_state()` (edit working state out of
  band), static/dynamic breakpoints. Built on a checkpointer so control survives a
  pause across process boundaries.
- **AG-UI** — an event protocol for the agent↔UI channel: ~16 typed events including
  `INTERRUPT` (pause for approval/input) and bidirectional `STATE_SNAPSHOT` /
  `STATE_DELTA` (RFC-6902 JSON-Patch state sync in both directions).
- **A2A** — the task lifecycle *is* the control surface: states
  `submitted / working / input-required / auth-required / completed / failed /
  canceled / rejected`; `tasks/cancel`, `tasks/get` (query), `tasks/resubscribe`
  (re-attach). `input-required` is a non-terminal interrupt: the caller resumes the
  *same* `taskId`.
- **Claude Agent SDK** — layered mid-session control: the `canUseTool` callback
  (per-call approve / deny / **modify-args**), permission modes (`plan`,
  `acceptEdits`, `bypassPermissions` — an autonomy dial), hooks (`PreToolUse`,
  `Stop`, …) as the universal-invariant layer, and a streaming `interrupt()`.
- **OpenAI Agents SDK / Assistants** — run lifecycle
  `queued → in_progress → requires_action → completed`; `requires_action` +
  `submit_tool_outputs` (approval), cancel-run, tool-approval interruptions surfaced
  as a serializable `RunState` (a paused run, not a new turn), and guardrail
  *tripwires* that halt a run before spend.
- **Durable-execution engines** — the canonical control/data split. **Temporal**:
  Signal (async one-way write), Query (sync read-only), Update (sync read/write with a
  validator), and graceful *cancel* vs forceful *terminate*. **Restate**: awakeables
  (suspend until an external `completeAwakeable`), signals, recursive cancel with
  compensation vs `kill`. **Inngest**: `cancelOn` events, `waitForEvent`, pause APIs.
- **Supervisory-control theory** — human-*in*-the-loop (a checkpoint the system cannot
  pass) versus human-*on*-the-loop (autonomous run, human intervenes at their own
  discretion via high-level indicators), and *adjustable / sliding autonomy* (the
  level of autonomy is a runtime-variable dial, not a fixed setting).
- **Origins** — TCP urgent/`MSG_OOB` (the term itself: "a conceptually independent
  channel"), MCP `notifications/cancelled` + `notifications/progress` (a request-scoped
  control channel parallel to the call), and cancellation tokens (`context.Context`,
  `AbortSignal`, `CancellationToken`) — cooperative stop propagated at await points.

### The ~15 canonical operations the field converged on

1. **Pause / interrupt** — suspend at a safe boundary without ending.
2. **Resume** — continue, optionally injecting a value the agent awaited.
3. **Cancel / abort** — graceful stop with cleanup / compensation.
4. **Kill / terminate** — forceful stop, no cleanup.
5. **Redirect-goal / reroute** — change what the agent does next.
6. **Edit-state** — mutate the agent's working state out of band.
7. **Approve / deny (+ modify)** — gate a specific pending action, optionally
   rewriting its arguments.
8. **Query-status** — read live state / progress without mutating.
9. **Halt-on-policy (guardrail / tripwire)** — automatic stop on a rule violation.
10. **Adjust-budget / limits** — change resource or step ceilings mid-run.
11. **Adjust-autonomy-level** — slide the oversight dial (approve-everything ↔ run-free).
12. **Prioritize / expedite** — act on an urgent message ahead of the in-band queue.
13. **Re-attach / subscribe** — reconnect an operator/peer to a running session's stream.
14. **Signal / send-event** — deliver a named, fire-and-forget message that triggers a handler.
15. **Fork / snapshot-and-branch** — checkpoint durable state and spawn a divergent continuation.

Two structural constants underlie all of them: durable checkpointed state, and a
separate addressable channel keyed by the execution id. fak has both.

## What fak ships today (honest inventory)

| Canonical op | fak surface | Status |
|---|---|---|
| Pause / Resume | `fak session pause/resume`, `fak signal pause/resume` (turn-granular park, warm-KV resume seam) | **shipped** (#1321) |
| Cancel / drain | `fak session stop` → `Draining` at the next boundary | **shipped** |
| Throttle | `fak session throttle` | **shipped** |
| Query-status | `fak session ls/status/context`, `GET /v1/fak/session/{id}`, `/session/changes` stream | **shipped** |
| Adjust-budget | `fak session budget --turns/--tokens/--context-tokens` (rev-fenced RMW) | **shipped** |
| Adjust-pace | `fak session pace --max-tokens/--gap-ms` (`SessionPlanner.ApplyPace`) | **shipped** |
| Prioritize | `fak session priority <N>` (scheduler re-sorts next boundary) | **shipped** |
| Halt-on-policy | adjudicator deny + `loopmgr` governor (`WITNESS_COLLAPSE`, `REFUSAL_STORM`, …) | **shipped** |
| Signal / send-event | `fak signal steer` → `a2achan` Session bus, spliced at boundary | **shipped, wiring flagged partial** (#760) |
| Set-envelope (wall/spend/throughput) | `fak session envelope <spec>` parses all axes | **parsed, only budget+pace applied** |
| Kill / terminate | `fak fleet janitor --apply` (process-tree kill); A2A `tasks/cancel` | **coarse / act-path simulated** |
| Re-attach / subscribe | `/v1/fak/session/changes` revision stream; resume is cold | **partial** |
| **Redirect-goal** | only freeform `steer` prose | **gap** |
| **Edit-state** | — | **gap** |
| **Approve / deny (operator)** | reversibility gate is agent *self-confirm* (`_fak_confirm`), no operator inbox | **gap** |
| **Adjust-autonomy (mid-run)** | permission floor is fixed per session | **gap** |
| **Fork / live checkpoint** | `sessionimage` rehydrate is cold; no live snapshot verb | **gap** |

Surfaces: the control plane is **CLI + HTTP** today. The TUI (`fak tui`) is
render-only (no mutation keybindings). The inbound Slack chatops door is epic #2259
(unshipped). `trajctl` — the *goal-curve* steering plane — has its ledger shipped but
its live-nudge half unshipped and its CLI parked.

## The core design move: a closed control vocabulary

The premise of this work is the goal's own framing — **replace prose mid-run with a
structured set of controls**. The eight-plus lifecycle/resource verbs above already
do that for *how much* and *how fast* a session runs. The freeform `steer` is the one
hole: to change *what* the agent does, or to tighten *what it may touch*, you still
inject prose.

The move is to make the operator control surface a **first-class closed vocabulary**,
exactly analogous to fak's closed *refusal* vocabulary (`dos.toml [reasons.*]`). Each
control op is a typed record with four fixed properties:

- **who may send it** — a capability on the `a2achan` send (operator principal), so
  control authority is checked, not assumed;
- **when it applies** — the boundary it lands on (immediately / next turn / next safe
  quiesce point), stated per op, never mid-stream;
- **the witness of applied** — the loop-side proof the op was *consumed*, not merely
  enqueued (the steer splice-vs-enqueue witness, generalized to every op);
- **the structured refusal** — a closed reason when the op is illegal for the current
  state (cannot cancel a `Stopped` session; cannot redirect a session with no
  objective declared), from the same closed-vocabulary discipline as every other
  fak refusal.

`steer` (freeform prose) stays — but demoted to the **escape hatch**. The structured
verbs are the default, so routine control is auditable, capability-checked, refusable,
and injection-resistant by construction; prose is the deliberate, logged exception.

### The proposed vocabulary (grouped)

- **Lifecycle**: `hold` (pause), `resume`, `drain` (graceful cancel), `throttle`,
  `terminate` (forceful, with the graceful/forceful distinction the field draws and
  fak currently blurs).
- **Resource**: `set-budget`, `set-pace`, `set-priority`, `set-envelope`
  (finish wall/spend/throughput).
- **Semantic** (the gap): `redirect` / `set-objective` (structured goal change),
  `add-constraint` (tighten the live policy — forbid a tool, add a deny rule, narrow a
  file-tree lane), `set-autonomy` (slide the mid-run permission dial), `edit-state`
  (bounded working-state mutation).
- **Oversight**: `approve` / `deny` (+ modify) of a specific pending gated action via
  an operator inbox; `query` (status); `subscribe` (re-attach).
- **Durability**: `checkpoint` (on-demand live snapshot), `fork` (snapshot-and-branch).
- **Fleet**: broadcast a lifecycle/semantic op to every session on a lane or wave (the
  missing "steer the whole wave" verb).
- **Escape hatch**: `steer` (freeform prose), logged and de-prioritized.

## Honest fences

- **Not yet, by construction.** Nothing new here is shipped. The lifecycle/resource
  half *is* shipped and is the platform; every semantic/oversight/durability op is a
  named gap with its own witnessed done-condition in the child issues. The `steer`
  splice is itself flagged `partial` (#760) and its end-to-end consumption should be
  re-witnessed before anything is built on top.
- **Regime-gate every intervention.** From the trajectory-control doctrine
  (#2533): interventions *harm* high-scoring sessions. A control op — especially an
  automatic nudge — must be able to gate on curve health, and the default posture for a
  healthy session is hands-off. Out-of-band control is a scalpel, not a steering wheel
  held the whole drive.
- **Control is a typed verb, never prose-executed.** The injection fence is absolute:
  an operator message either parses as a closed control op (capability-checked, applied
  at a boundary, witnessed) or it is the explicit `steer` escape hatch delivered as
  *data* to an already-authorized loop. There is no path where channel text becomes an
  instruction the kernel executes.
- **Next turn boundary, not mid-stream.** Every op lands at a loop boundary; an
  already-open upstream round-trip finishes first. `checkpoint`/`fork` need a durable
  snapshot point, so they are gated on the session-image seam being live, not cold.

## The spine and the fan-out

**Spine (S):** this note + a first-class control-op vocabulary type (the closed enum,
the four per-op properties, and the *existing* verbs registered into it) with the
generalized witness-of-applied contract. It unifies the fragmented surface
(`fak session` drive-state, `fak signal` job-control + steer, `fak steering` — which
is a code-quality scorecard, an unrelated name collision — and parked `trajctl`) under
one named plane and one grammar.

**Children (the gaps, worst-first):**

1. Structured `redirect` / `set-objective` — replace freeform steer for the common
   "change the goal" case.
2. `add-constraint` — tighten the live policy on a running session (forbid-tool /
   add-deny / narrow-lane).
3. Operator `approve` / `deny` inbox — resolve a pending reversibility/ESCALATE gate
   out of band (operator-driven HITL, not agent self-confirm).
4. Graceful `drain` vs forceful `terminate` on the owned loop, with compensation; wire
   the A2A `tasks/cancel` act-path (de-simulate).
5. `set-autonomy` — a mid-run permission-mode dial (approve-everything ↔ run-free).
6. Live `checkpoint` / snapshot verb on a running session.
7. `fork` — snapshot-and-branch a divergent continuation from a live checkpoint.
8. Apply the envelope `wall` / `spend` / `throughput` axes live (finish parsed-not-applied).
9. TUI control keybindings — mutate a running session from `fak tui` over the existing
   routes (turn the render-only console into a control pane).
10. Fleet / lane-scoped broadcast — one control op to every session on a lane or wave.
11. Wire the `trajctl` steering ladder (regime-gated auto-nudge over the control bus)
    and unpark its CLI — connect goal-curve control to this plane.
12. Witness-of-applied + structured refusals for *every* control op (generalize the
    steer splice witness; add the closed reason set for illegal ops).
13. `subscribe` / re-attach — an operator or peer re-attaches to a running session's
    control + event stream (the A2A `resubscribe` equivalent), addressable by an
    external trace.
14. Unify + document the operator control plane — one doctrine page, one front-door
    verb listing the closed vocabulary, and a fix for the naming fragmentation, so an
    agent or human can discover the whole surface in one place.

## Related

- Trajectory-control epic #2533 — the *goal-curve* steering sibling; its live-nudge
  ladder is child 11 above.
- Slack chatops door epic #2259 — one inbound *transport* for these ops (the door);
  this plane owns the *vocabulary* the door would carry.
- Reversibility preview-confirm gate (`internal/adjudicator/reversibility.go`) — the
  agent-self-confirm HITL that child 3 lifts into an operator inbox.
- Session drive-state control (#620), operator steer (#760/#850), session control on
  the owned loop (#1321) — the shipped platform this builds on.
