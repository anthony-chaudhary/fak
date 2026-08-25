---
title: "Out-of-band operator control plane"
description: "The closed steer, pause, resume, cancel, and throttle vocabulary for a running fak session, with each op's capability, boundary, and witness."
---

# The out-of-band operator control plane

**The doctrine page for epic [#2753](https://github.com/anthony-chaudhary/fak/issues/2753)
([#2768](https://github.com/anthony-chaudhary/fak/issues/2768)).**
Vocabulary spine: [`internal/sessionctl`](../internal/sessionctl/vocab.go) · CLI:
[`fak session`](../cmd/fak/session_cmd.go) · [`fak signal`](../cmd/fak/signal.go) ·
read-only table: [`fak ps` / `fak top`](../cmd/fak/ps.go) · design record:
[OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05](notes/OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05.md).

Once a session has started, how does a human change what it is doing? The usual answer
is to type more prose into the same channel the agent reads as its task — *in-band*
steering, which is slow (wait a turn boundary), ambiguous (prose is interpreted, not
applied), and injection-shaped (a steering instruction delivered as task text is
indistinguishable from task content).

fak's answer is the split the rest of computing settled on decades ago (TCP OOB, Unix
signals, Temporal signals/queries/updates): a **separate, typed, capability-checked,
witnessed control plane** over a live session's drive state. The model proposes on the
data plane; the operator disposes on the control plane.

**This page is the one place that names the whole plane.** If you are looking for "what
can I do to a running session, and how do I know it took", you are in the right place.

## The control op is a typed record, not a sentence

Every op in the plane carries the same four fixed properties. They are declared once, as
data, in [`sessionctl.Vocabulary()`](../internal/sessionctl/vocab.go) — not re-invented
per verb:

1. **Capability** — the send-right required to submit the op.
2. **Boundary** — when it takes effect relative to the loop. Never mid-decode.
3. **Witness** — the shape of the loop-side proof it was *consumed*. The enqueue or the
   table write is **never** the witness; the running arm observing it at a boundary is.
4. **Refusal** — the closed reason token it surfaces when it is illegal for the state.

The behavior each row describes is proven loop-side by
[`internal/agent/loop_control_witness_test.go`](../internal/agent/loop_control_witness_test.go),
which cross-binds its op set and per-op refusal tokens back to the spine. A new op is not
in the plane until it registers a row **and** the witness table proves it.

## The closed vocabulary

Nine ops at HEAD. This table is generated from the same registry the code reads;
[`cmd/fak/control_plane_doc_test.go`](../cmd/fak/control_plane_doc_test.go) fails if it
drifts.

| Op | CLI spelling | Boundary | Witness | Refuses with |
|---|---|---|---|---|
| `steer` | `fak signal <id> steer --text "…"` | next-turn | splice | `DEFAULT_DENY`, `TRUST_VIOLATION` |
| `redirect` | *no CLI spelling yet* | next-turn | directive | `REDIRECT_MALFORMED`, `REDIRECT_NO_REDIRECTABLE_STATE` |
| `pause` | `fak session pause <id>` · `fak signal <id> pause` | next-turn | boundary-stop | `CONTROL_SESSION_TERMINAL` |
| `resume` | `fak session resume <id>` · `fak signal <id> resume` | immediate | same-turn-wake | `CONTROL_SESSION_TERMINAL` |
| `cancel` | `fak session stop <id>` · `fak signal <id> stop` | quiesce | boundary-stop | `CONTROL_SESSION_TERMINAL`, `CONTROL_REV_STALE` |
| `terminate` | `fak session terminate <id>` | immediate | safe-point-stop | `CONTROL_SESSION_TERMINAL`, `CONTROL_REV_STALE` |
| `throttle` | `fak session throttle <id>` | next-turn | sampling-cap | `CONTROL_SESSION_TERMINAL` |
| `budget` | `fak session budget <id> [--turns N] […]` | next-turn | boundary-stop | `CONTROL_SESSION_TERMINAL` |
| `priority` | `fak session priority <id> <N>` | immediate | scheduler-read | `CONTROL_SESSION_TERMINAL` |

**Boundaries** are a closed three-value grammar. `next-turn`: taken at the loop's next
clean turn boundary. `quiesce`: the session drains to a safe stop at the boundary —
`cancel` lets the turn finish. `immediate`: consumed without waiting for a fresh
next-turn gate — `resume` wakes the held turn in place, `terminate` cancels the in-flight
model call at the next safe point, `priority` is read by the scheduler at its next pick.

`cancel` and `terminate` are the deliberate pair: **cancel finishes the turn, terminate
does not.**

**Witness shapes** name how "applied" is proven. `splice`: the payload lands in the
turn's user input and the mailbox drains. `directive`: it is carried in as a standing
system directive and the live objective reflects it. `boundary-stop` /
`safe-point-stop`: the arm halts with the op's closed stop-reason recorded on
`ArmMetrics.StoppedBySession` (`DRAINING` for cancel, `TERMINATED` for terminate).
`same-turn-wake`: a final answer from exactly the resumed turn. `sampling-cap`: the
turn's effective `SampleParams` cap reflects the write. `scheduler-read`: the next pick's
rank order reflects it.

## The front door

**Two CLI verbs write, one reads.** They are not two control planes — `fak signal` is the
OS-process-model framing over the same `/v1/fak/session/{id}/…` routes `fak session`
uses.

| Surface | What it is | Use it when |
|---|---|---|
| [`fak session`](../cmd/fak/session_cmd.go) | The operator control surface for one served session's live **drive state**: read it, then cancel or update it in flight. | You want the full write vocabulary, including `terminate`, `budget`, and `priority`. |
| [`fak signal`](../cmd/fak/signal.go) | **Job control** in Unix names — `pause` (SIGSTOP), `resume` (SIGCONT), `stop` (SIGTERM), plus `steer` to send *input* to a running agent. | You think in process terms, or you want `steer`. |
| [`fak ps`](../cmd/fak/ps.go) / `fak top` | The **read-only process table**: one aligned row per live session. Issues no control verb. | You need to find the session id first, or watch the fleet. |

Read-back after any write is `fak session status <id>` (or `fak ps`). Every write verb
accepts `--if-rev N`, the optimistic-concurrency guard: a stale operator racing a second
controller loses with `CONTROL_REV_STALE` instead of clobbering.

Transport is `POST /v1/fak/session/{id}/{verb}` (and `/steer`); `GET
/v1/fak/session/changes` is the drive-state revision stream. Connection flags are shared:
`--addr` (`$FAK_ADDR`, default `http://127.0.0.1:8080`) and `--key` (`$FAK_KEY`).

## What is *not* a control op — the `steering` name collision

Three shipped verbs share the "steer" root and **none of the first two is an out-of-band
control op**. This is a real collision. Per epic scope it is documented here rather than
renamed — renaming a shipped verb would break users for a naming win.

| Verb | What it actually is | Relation to this plane |
|---|---|---|
| `fak steering` | The **steerability Slack surface** for `#steering-guard` (`status`/`report`/`alert`/`pin`) — a CI/CD reporting feeder. See [STEERABILITY-SCORECARD](STEERABILITY-SCORECARD.md). | **None.** Reports *about* steerability; controls nothing. |
| `fak steer` | **Pull-request steering** (`fak steer prs`): folds landed trunk commits into PR-sized operator-attention units. See [operator-steerability PRs](operator-steerability-prs.md). | **None.** Operates on commits, not on a running session. |
| `fak signal <id> steer` | The control op in the table above. | **This is the one.** |

`fak trajctl` is a fourth near-miss worth naming: it is the trajectory-control
**objective-ledger lifecycle** (`declare`/`close`/`list`/`curve`/`score`/`scorers`) from
epic [#2533](https://github.com/anthony-chaudhary/fak/issues/2533). Its CLI is live
(unparked under [#2765](https://github.com/anthony-chaudhary/fak/issues/2765)), but it
declares and scores objectives out of band of any *running* session — it is not a
registered control op, and the steering-ladder / regime-gated nudge half of #2765 is
still open.

**Rule of thumb.** A control op takes a **session id** and applies at a **loop boundary**.
If the verb does not take a session id, it is not on this plane.

## Honest fences

The plane is real, and it is smaller than the epic. What is *declared* here is not
uniformly *enforced*:

- **Capability is named, not yet gated.** Every row carries a required send-right
  (`operator-send` for `steer`/`redirect`, `operator-control` for the drive-state ops),
  but only `steer`'s is wired — it rides the taint-aware `a2achan` bus and fails closed
  (`DEFAULT_DENY` capless, `TRUST_VIOLATION` tainted/over-scoped). The
  `POST /v1/fak/session/{id}/{verb}` route today applies **run-state and CAS legality
  only**, with no capability floor. The rows record the name a future gate adopts.
- **The spine has no production consumer yet.** `sessionctl.Vocabulary()` is declarative
  data; wiring each control route, audit, and help surface to read it is per-op follow-on
  (tracked in [#3559](https://github.com/anthony-chaudhary/fak/issues/3559)).
- **`redirect` has no CLI spelling.** It is registered and library-reachable
  (`sessionctl.EnqueueRedirect`, appending to a per-session mailbox), with no HTTP route
  and no verb. Structured goal change is not yet an operator affordance.
- **Some shipped `fak session` writes are not registered ops.** `pace`, `envelope`, and
  `run <id> <state>` write drive state but carry no vocabulary row, so they have no
  declared boundary or witness shape. Registering them is open work, not a claim.
- **`steer`'s splice is flagged partial** ([#760](https://github.com/anthony-chaudhary/fak/issues/760)) —
  re-witness end to end before building on it.
- **Not yet shipped at all:** `add-constraint`, `set-autonomy`, operator approve/deny
  inbox, live checkpoint/fork, fleet or lane-scoped broadcast, and TUI control
  keybindings. They are named children of #2753, not capabilities.

The rest of `fak session` — `audit`, `compact-audit`, `gate-fatigue`, `reset-diff` — is
**offline transcript analysis**, not control. It touches no running session.

## Reach it from here

- Every verb's flags and one-line synopsis: [CLI reference](cli-reference.md)
- `fak help session` · `fak help signal` · `fak help ps` each point back at this page.
- The concept expansion, SOTA survey, and coverage table:
  [OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05](notes/OUT-OF-BAND-OPERATOR-CONTROL-2026-07-05.md)

### Lifecycle refusal recovery schema

Ordinary `409 session_paused`, `session_draining`, and `session_stopped` responses include a stable `recovery` object with `state`, `terminal`, `retryable`, `session_id`, and a typed `next_action`. A paused session is held, not killed: continuity remains available, retries stay disabled while paused, and clients resume it through the control API using `next_action: "resume"` and the returned session id. Draining advertises `wait_for_drain`; stopped is terminal and advertises `start_new_session`. Clients must branch on these tokens rather than parse message prose. The schema adds fields compatibly and never exposes configuration paths or command strings.
