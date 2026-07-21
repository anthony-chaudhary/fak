---
title: "The supervisor seat: when the monitor loop is an agent, not a tick or a human"
description: "Progresses the 'monitor agents that run the meta-loop and spawn worker agents' concept. Maps the dense existing coverage (no-babysitting witnesses #2269, deterministic self-heal #2038, dispatch spawn, human-escalate #2271/#4347/#3517), isolates the one un-ticketed residual — an LLM agent that HOLDS the supervisor seat — and parks a doctrine-consistent epic + 3 leaves."
---

# The supervisor seat: when the monitor loop is an agent, not a tick or a human (2026-07-13)

**Operator ask (2026-07-13):** *"progress the concept of 'monitor agents' that do the
meta-loop monitor thing spawning for agent work — should be open tickets if not,
research and create."*

**Status:** research note. Nothing here changes runtime. It records a duplicate
sweep, isolates the residual concept, reconciles it with the no-babysitting
doctrine, and parks four contract-shaped issue bodies (one epic, three leaves)
for an authorized producer to file. Read-only survey; as of HEAD on `main`,
2026-07-13.

## The concept, stated precisely

A **monitor/supervisor agent** is an LLM that *holds the supervisor seat*: it is
woken when the fleet emits a typed event, it reads the fleet's **already-witnessed
state**, and it decides the meta-loop's next move — *spawn* a worker on the next
leaf, *replace* a dead one, *replan* a stuck lane, *widen* a lease, or *escalate*
to a human. It is the **Executor of the supervisor role** in the loop-family map's
four-role decomposition (Timer · Selector · Executor · Witness), applied one level
up: its "work item" is the fleet itself.

The load-bearing word is **agent**. Today the supervisor seat is held by either a
**deterministic tick** (`fak dispatch tick`, `fleetmon`, the fleet-status
self-heal verdict) or a **human** (an operator turn, the `HUMAN_RESIDUAL` bucket).
The concept to progress is the **middle tier**: the class of supervisor decisions
that are *judgment under uncertainty* — too situational for a fixed policy, too
cheap to be worth a person — and therefore fall to a human today only because no
agent is authorized to take them.

## What already exists (the honest map)

The supervision space in this repo is densely built. The dedup sweep found the
concept is covered on **four** of its five faces:

| Face | What it is | Where it lives (open) | Actor |
|---|---|---|---|
| **Witness** | make each condition decidable from evidence, not self-report | no-babysitting epic **#2269** + R-rungs (**#2271** packet, **#2274** ratchet); `dos_status`/`dos_verify`/`commit-audit`; `fleetmon` | deterministic |
| **Self-heal** | compute a fleet health verdict; heal transients; escalate only what a lifecycle can't | fleet-status epic **#2038** ("trends over snapshots, escalate only what the lifecycle can't heal") | deterministic |
| **Spawn** | admit a worker onto a lane under a lease | dispatch tick; scale-spawn **#1333**; lane arbitration (`dos_arbitrate`) | deterministic |
| **Escalate-to-human** | route the residual to a person as a typed packet | **#2271** (`fak.escalation.v1`), **#4347** (bridge escalate→`BLOCKED_BY_HUMAN`), **#3517** (route human-gates through `choicetriage`), **#4343** (→ Slack), **#4230** (phone approve/deny) | deterministic router → human |
| **Decide the middle** | consume the witnesses and *choose* spawn/replace/replan/widen for the non-obvious cases | **— not ticketed as an agent —** | **the gap** |

The three closest epics were read in full to be sure the residual is real:

- **#2038** is deterministic reporting plus a *computed* self-heal verdict (lifecycle
  ownership + a `rollup` fold). It escalates what it can't heal; it does not
  *decide* what to spawn next.
- **#3517** ("decenter the operator") is the closest in spirit — it routes work
  *away* from the human toward fresh-context agents — but the router is the
  **deterministic `choicetriage` disposition kernel** (`TAKE_OBVIOUS` /
  `FRESH_CONTEXT` / `FILE_TICKET` / `HUMAN_RESIDUAL`), not an LLM holding the seat.
- **#4347** is a pure read-only ledger→view fold (escalate-stop →
  `BLOCKED_BY_HUMAN`), no decision at all.

**Conclusion of the sweep:** the Witness, Self-heal, Spawn, and Escalate faces are
each a coherent open program. The **agent-held supervisor seat for the middle
tier** is named by none of them. That is the residual worth progressing.

## Reconciling with the doctrine (why this is not the thing the slope eats)

The no-babysitting doctrine (#2269, `CONCEPT-NO-BABYSITTING-2026-07-01.md`) is
emphatic about what **not** to build — and a naïve "monitor agent" is exactly on
its forbidden list:

> *No transcript-reading "is it stuck?" judge models — liveness is a rung, not a
> vibe (a judge is a recognizer in an arms race). No prompt-therapy layer that
> re-prompts a struggling agent "better." No smarter-than-the-model code grader.
> No load-bearing dashboard.*

The supervisor agent proposed here is **not** any of those, and the difference is
structural, not a matter of good intentions. Four fences make it doctrine-safe:

1. **It reads witnessed state, never transcripts.** Its input is the *closed set of
   typed witnesses the doctrine already emits* — `dos_status` liveness/progress,
   the `fleetmon` health verdict, the `fak.escalation.v1` packet, lane leases. It
   never opens a worker's transcript to *guess* health. Health stays a
   deterministic rung (W1); the agent consumes the rung's output, it does not
   re-adjudicate it. That is the line between a *recognizer* (forbidden) and a
   *router* (this).

2. **It is interrupt-driven, so nobody watches a healthy fleet.** It is woken by
   the same typed events the owned-turn / completion-watcher work already delivers
   (**#2388**, **#2932**) — a worker exits, a packet fires, a lease drains. A green
   fleet emits nothing, so the agent sleeps. This preserves B1 (poll→interrupt): it
   is an interrupt *handler*, not a poller, and it is not a dashboard.

3. **It acts only through the one spawn authority.** Every action it can take is an
   existing deterministic admission verb — `dos_arbitrate` for the lane, the
   dispatch admit path for the worker. It has **no private spawn path** and cannot
   invent a lane. This is loop-map rule R2 (one spawn authority) held even when the
   caller is an agent, so its blast radius is exactly a human operator's, no wider.

4. **Its envelope is earned and it fails closed.** Which interrupt *classes* it may
   handle unattended is gated by the autonomy ratchet (**#2274** / B5): it starts
   advisory (proposes, a human confirms), and widens only as its ledger witness
   rate earns it. Anything outside the envelope, or any witness it cannot obtain,
   escalates as a packet (**#2271**). The babysitting counter (touches per
   witnessed shipped unit) is its keep-bit: **if the agent in the seat does not
   lower that number, it is reverted.**

Put plainly: the doctrine forbids an agent that *manufactures a health signal*. It
does not forbid an agent that *consumes a witnessed signal and picks a
deterministic action*. The former is a recognizer in an arms race; the latter is
the meta-loop's executor. The loop note already calls the dispatch⇄replan cadence
"a supervisor concern, not a worker concern" — this epic is the first to ask
*whether an agent, not only a tick or a person, may be that supervisor*, and to
bound the answer so it stays on the safe side of every fence above.

## Why an agent, and not just more deterministic rules

The Self-heal face (#2038) already deterministically handles the cases a fixed
policy *can* express: reap a stale session, retry a transient, escalate an auth
wall. The residual is the cases where a fixed policy is brittle:

- **Selection under a tie.** Three leaves are ready and disjoint; one seat frees.
  Which one, given the fleet's current shape (what's mid-flight, what a landing
  will unblock, what's been starved)? A priority sort is a blunt instrument here.
- **Replan vs. widen vs. wait.** A lane is stuck behind a peer's half-landed WIP.
  Deterministic recovery can detect it; choosing *replan the leaf* vs. *widen the
  lease* vs. *wait for the sync merge* is judgment the memory log shows humans make
  case by case (e.g. the push-divergence and poison-tree diagnoses).
- **Composing a recovery from typed parts.** The witnesses say *what* is wrong; the
  order and combination of admitted verbs that fixes it without thrashing is the
  supervisor's call.

These are exactly "judgment, not taste" — decidable from witnessed state, but not
by a static rule. That is the seat an agent can hold and a `switch` statement
cannot, and it is below the B7 human residual (goal-fit, architecture, taste),
which stays human on purpose.

## The shape (proposed, for the epic to refine)

A bounded **supervisor-agent loop**, defined by four closed contracts so it can
never drift into the forbidden recognizer:

- **Input contract** — a fixed projection of typed witnesses only (`dos_status`
  digest, `fleetmon` verdict, escalation packet, lease table). No transcript, no
  raw log, no free-text. If a needed witness is missing, it escalates, it does not
  infer.
- **Action vocabulary** — a closed verb set (`spawn` / `replace` / `replan` /
  `widen` / `escalate` / `hold`), each lowering to an existing deterministic
  admission call. Every action is itself witnessed (a lease row, a dispatch admit).
- **Authorization envelope** — a per-interrupt-class allow-list gated by the #2274
  ratchet; `warn`→`enforce` soak like #3517's gate; fail-closed on
  `witness_refused`.
- **Keep-bit** — the babysitting counter (#2269's witness). The seat is an
  experiment that must *lower* touches-per-shipped-unit or be reverted; it earns
  its autonomy mechanically or loses it.

## Filed issues (contract-shaped; filed verbatim as #4477–#4480)

Filing note: this repo's convention is to keep a ready body in the note with a
stable dedup marker `<!-- fak-<slug>-key: <ns>/<seg> -->` (a rerun updates rather
than re-files), filed with `gh issue create`
(`.github/ISSUE_TEMPLATE/worker-ready-issue.yml` is the shape). Re-validate any
edit with `fak issue contract` before updating a filed issue. Tier is the
producer's call (`needs-triage` if unsure); the epic is frontier design.

**Dedup:** deduped against #2269 (witnesses/packets, not a decider), #2038
(deterministic self-heal verdict, not agent selection), #1333 (deterministic
spawn scaling), #3517 (deterministic `choicetriage` router toward fresh context,
not an LLM in the seat), #4347/#4343/#4230 (escalate-to-human plumbing). None
places an LLM agent in the supervisor seat; this epic does, bounded.

**Contract verdict (measured, `fak issue contract -from-issues`, HEAD 2026-07-13).**
The four bodies below were extracted verbatim and scored by the repo's own issue
contract. Result: the **three children each score 100 / `ready` / dispatchable**
(lanes `gateway`, `cmd`, `guardvars`); the **epic scores 100** but is
`triage_only` with the single reason `ISSUE_NOT_DISPATCH_LEAF` — which is inherent
and correct for a parent epic (epics group children, they are not dispatch leaves).
A producer re-running the check at file time should see the same verdicts.

---

### Epic — #4477 (measured contract verdict: score 100, `triage_only` — `ISSUE_NOT_DISPATCH_LEAF` is inherent to a parent epic)

Filing: `gh issue create` with labels `epic`, `dispatch`, `operator`,
`priority/P1` (raised from P2 on 2026-07-13 — foundational architecture; P1 is
"high priority — cheap/permanent/foundational", P0 reserved for load-bearing-now).

<!-- fak-supervisor-agent-key: supervisor-agent/epic -->

**Title:** `epic(dispatch): the supervisor seat as an agent — a bounded monitor/dispatch agent between deterministic self-heal and the human residual`

**Dispatch · operator** · the meta-loop's supervisor decisions are held by a tick or a human; the middle tier of judgment calls has no agent seat.

#### Parent context
Progresses the operator concept (2026-07-13): *"monitor agents that do the meta-loop monitor thing spawning for agent work."* Sits above the no-babysitting witness layer (#2269) and beside the deterministic self-heal report (#2038), the dispatch spawn path (#1333), and the escalate-to-human plumbing (#2271/#4347/#3517/#4343). Design note: `docs/notes/CONCEPT-SUPERVISOR-AGENT-SEAT-2026-07-13.md`.

#### Current state
The supervisor seat — the actor that consumes fleet witnesses and decides spawn/replace/replan/widen — is held by a **deterministic tick** (`fak dispatch tick`, `fleetmon`, #2038's self-heal verdict) or a **human** (an operator turn, the `HUMAN_RESIDUAL` bucket). The tick handles cases a fixed policy can express; the rest fall to a person. There is no agent authorized to take the middle tier of judgment calls (selection under a tie, replan-vs-widen-vs-wait, composing a recovery from typed parts), so they escalate to a human by default even when they need no human taste.

#### Why now
Fan-out already outruns constant human attention (the memory log records humans hand-adjudicating push-divergence, poison-tree, and starvation cases the tick can't). The witness layer (#2269) and the typed event wakeups (#2388/#2932) now exist, so an interrupt-driven agent can consume witnessed state without polling and act through the one spawn authority — the two preconditions that make an agent seat doctrine-safe rather than a forbidden recognizer.

#### Working spine
A bounded supervisor-agent loop with four closed contracts: (1) an **input contract** of typed witnesses only (`dos_status` digest, `fleetmon` verdict, `fak.escalation.v1` packet, lease table) — never a transcript; (2) a closed **action vocabulary** (`spawn`/`replace`/`replan`/`widen`/`escalate`/`hold`) each lowering to an existing deterministic admission verb; (3) a ratchet-gated **authorization envelope** (#2274), `warn`→`enforce` soak, fail-closed on `witness_refused`; (4) a **keep-bit** — the babysitting counter (#2269) must fall or the seat is reverted. Children below build the three contracts; the keep-bit rides #2269's counter.

#### In scope
The supervisor-agent loop and its four contracts; wiring it to the existing witness inputs and admission verbs; the ratchet gate; the advisory→enforce rollout; the keep-bit measurement hook.

#### Out of scope
Any transcript-reading or health *recognition* (health stays a deterministic rung — W1/`dos_status`/`fleetmon`). The B7 human residual (goal-fit, architecture, taste). New spawn paths or lane vocabulary (reuse `dos_arbitrate` + dispatch admit). New escalation transport (reuse #2271/#4343).

#### Done condition
An interrupt-driven agent, given only typed witnesses, proposes and (within its earned envelope) takes spawn/replace/replan/widen through existing admission verbs; everything outside the envelope or lacking a witness escalates as a packet; over a soak window the babysitting counter (touches per witnessed shipped unit) does not rise and the seat is either kept on evidence or reverted.

#### Witness
Per child. Epic-level: the babysitting counter time-series across the soak, plus the ratchet ledger showing envelope width tracking witness rate.

#### Acceptance gate
All three children merged and green; `warn`→`enforce` soak completed; keep-bit verdict recorded on the epic.

#### Work unit
Epic — three worker-ready children below.

#### Expected steps
3 (one per child).

#### Assumptions
- The typed witnesses (`dos_status`, `fleetmon` verdict, escalation packet, lease table) are sufficient to decide the middle-tier cases without a transcript. (Falsifiable: if a class of decision provably needs transcript content, it belongs in the human residual, not this seat.)
- The existing admission verbs (`dos_arbitrate`, dispatch admit) cover every action the vocabulary needs.

#### Confusion risks
- This is NOT a "judge model." It consumes witnessed state and picks a deterministic action; it never manufactures a health signal. If a change here starts reading transcripts to infer health, it has crossed into the forbidden recognizer — stop.
- "Supervisor agent" ≠ the deterministic self-heal verdict (#2038) and ≠ the `choicetriage` router (#3517). Those stay deterministic; this is the LLM seat for the judgment residual they hand off.

#### Coordination
- `internal/gateway`, `cmd/fak` dispatch, and `fleetmon` are contended lanes — arbitrate via `dos_arbitrate` before writing.

#### Lane
cmd

#### Likely files
- `internal/supervisoragent/`
- `cmd/fak/dispatch_tick_preflight.go`
- `internal/gateway/`

#### Closure binding
Closes when all three children (`supervisor-agent/input-contract`, `/action-vocab`, `/envelope-keepbit`) are merged and green and the keep-bit verdict — the babysitting counter across the `warn`→`enforce` soak — is recorded on this epic.

#### Trigger
Operator concept (2026-07-13); design note parked in-repo.

---

### Child 1 — input contract — #4478 (measured contract verdict: `ready`, dispatchable, score 100, lane `gateway`)

Filing: labels `enhancement`, `dispatch`, `priority/P1`.

<!-- fak-supervisor-agent-key: supervisor-agent/input-contract -->

**Title:** `feat(dispatch): supervisor-agent input contract — a closed projection of typed witnesses (dos_status · fleetmon verdict · escalation packet · leases), never a transcript`

**Dispatch** · the supervisor agent must be structurally unable to read a transcript; give it a fixed witness projection or it drifts into a recognizer.

#### Parent context
Child of the supervisor-seat epic (`supervisor-agent/epic`). Implements fence #1 from the design note: reads witnessed state, never transcripts.

#### Current state
Fleet witnesses exist but are scattered across surfaces (`dos_status` A2A digest, `fleetmon` health classes, `fak.escalation.v1`, the lease table via `dos_arbitrate`). Nothing assembles them into a single closed input a supervisor agent may consume, and nothing forbids that consumer from reaching for a transcript instead.

#### Why now
The input contract is the load-bearing doctrine fence: it is what makes the agent a router (safe) rather than a recognizer (forbidden). It must land before any decision logic, so the decision layer is physically incapable of seeing a transcript.

#### Working spine
Define a `SupervisorInput` projection: the `dos_status` liveness/progress digest, the `fleetmon` per-worker verdict (healthy/done/dead/stale/blocked), open `fak.escalation.v1` packets, and the live lease table — each a typed, payload-free field. Assemble it from the existing surfaces at wake time. No transcript, log body, or free-text field is included by construction; a missing witness surfaces as an explicit `absent` marker the decision layer must escalate on, not infer around.

#### In scope
The `SupervisorInput` type + its assembler from existing witness surfaces; a golden test pinning that it carries no transcript/payload field; the `absent`-witness marker.

#### Out of scope
The decision logic (child 2's vocabulary) and the envelope (child 3). Any new witness measurement — this only *projects* what #2269/`fleetmon`/`dos_status` already emit.

#### Done condition
A supervisor agent can be handed a `SupervisorInput` that fully describes fleet state from typed witnesses alone; a test proves the projection excludes transcript/payload content and marks missing witnesses `absent`.

#### Witness
`go test` green; a golden test asserts the projection's closed field set (no transcript/payload) and the `absent` behavior for a withheld witness.

#### Acceptance gate
Done condition + `make ci` green.

#### Work unit
One worker owns the projection type, its assembler, and the golden test.

#### Expected steps
4

#### Assumptions
- Every surface needed (`dos_status`, `fleetmon`, escalation, leases) is already queryable payload-free.

#### Confusion risks
- Do NOT add a "just in case" transcript or last-message field — that single field is what turns the agent into a recognizer. The projection's closedness is the whole point.

#### Coordination
- Touches the read side of `internal/gateway`/`fleetmon` — arbitrate the lane first.

#### Lane
gateway

#### Likely files
- `internal/supervisoragent/`
- `internal/gateway/`
- `internal/fleetmon/`

#### Closure binding
The `Witness` golden test above, green on a merged PR whose body carries `supervisor-agent/input-contract`, closes this leaf.

#### Trigger
Epic `supervisor-agent/epic`.

---

### Child 2 — action vocabulary through the one spawn authority — #4479 (measured contract verdict: `ready`, dispatchable, score 100, lane `cmd`)

Filing: labels `enhancement`, `dispatch`, `priority/P1`.

<!-- fak-supervisor-agent-key: supervisor-agent/action-vocab -->

**Title:** `feat(dispatch): supervisor-agent action vocabulary — closed verb set (spawn/replace/replan/widen/escalate/hold) lowered to existing admission verbs, no private spawn path`

**Dispatch** · the agent's every action must be an existing deterministic admission call, so its blast radius equals a human operator's — no wider.

#### Parent context
Child of `supervisor-agent/epic`. Implements fence #3: acts only through the one spawn authority (loop-map R2).

#### Current state
Spawning/arbitration is deterministic (`dos_arbitrate` for the lane, the dispatch admit path for the worker). There is no closed action surface an agent can be restricted to; a naïvely-wired agent could shell out and invent a lane, blowing past the lease discipline.

#### Why now
Without a closed vocabulary bound to existing verbs, an agent seat is unbounded and unauditable. This is the layer that keeps the agent's authority exactly a human operator's.

#### Working spine
Define a closed `SupervisorAction` union (`spawn`/`replace`/`replan`/`widen`/`escalate`/`hold`), each carrying only the typed args the corresponding deterministic verb needs, and each lowering to that verb (`dos_arbitrate` admission for `spawn`/`widen`; the dispatch admit path; the #2271 packet emit for `escalate`; a no-op for `hold`). No action reaches a raw shell or a lane the arbiter didn't grant. Every executed action is itself witnessed (a lease row / dispatch admit / packet).

#### In scope
The action union + its lowering to existing verbs; rejection of any action not expressible through them; a test that each verb produces its expected witnessed artifact.

#### Out of scope
The policy that *chooses* an action (that is the agent, driven by child 1's input). The envelope gating (child 3).

#### Done condition
An agent can only affect the fleet through the closed vocabulary; each verb is proven to lower to an existing admission call and leave a witnessed artifact; no path reaches a private spawn.

#### Witness
`go test` green; a test drives each verb and asserts the resulting lease/admit/packet artifact; a negative test proves an out-of-vocabulary action is rejected.

#### Acceptance gate
Done condition + `make ci` green.

#### Work unit
One worker owns the union, the lowering, and the tests.

#### Expected steps
4

#### Assumptions
- `dos_arbitrate` + dispatch admit + #2271 emit cover every action the middle tier needs; if a case needs a verb they lack, it escalates (child 3) rather than growing a private path.

#### Confusion risks
- `widen` still goes through `dos_arbitrate` — it is not a bypass of the lease rule, it is a re-arbitration. Do not add a "force" path.

#### Coordination
- `cmd/fak` dispatch + arbitration lane — arbitrate first.

#### Lane
cmd

#### Likely files
- `internal/supervisoragent/`
- `cmd/fak/dispatch_tick_preflight.go`

#### Closure binding
The per-verb witness test above, green on a merged PR whose body carries `supervisor-agent/action-vocab`, closes this leaf.

#### Trigger
Epic `supervisor-agent/epic`.

---

### Child 3 — ratchet-gated envelope + keep-bit — #4480 (measured contract verdict: `ready`, dispatchable, score 100, lane `guardvars`)

Filing: labels `enhancement`, `operator`, `rsi`, `priority/P1`.

<!-- fak-supervisor-agent-key: supervisor-agent/envelope-keepbit -->

**Title:** `feat(operator): supervisor-agent authorization envelope (ratchet-gated, warn→enforce, fail-closed) + keep-bit on the babysitting counter`

**Operator · rsi** · the agent may handle unattended only the interrupt classes it has earned, and it keeps the seat only if it lowers touches-per-shipped-unit.

#### Parent context
Child of `supervisor-agent/epic`. Implements fence #4: envelope earned via the #2274 ratchet; keep-bit is the #2269 babysitting counter.

#### Current state
The autonomy ratchet (#2274) exists advisory; the babysitting counter (#2269) exists as the doctrine's witness. Neither is wired to gate an agent's per-interrupt-class authority or to render a keep/revert verdict on an agent seat.

#### Why now
An agent seat without an earned, fail-closed envelope is either unsafe (too much unattended authority) or useless (everything escalates). The ratchet already models earned width; this binds it to the supervisor agent and makes the seat an experiment with an explicit keep-bit.

#### Working spine
Gate each `SupervisorAction` class by the #2274 ratchet: below the earned width, the action is *proposed* and a human confirms (advisory); at/above, it executes unattended. Fail closed on `witness_refused` (no witness → escalate, never act). Soak-switch `warn`→`enforce` (mirror #3517's `FAK_OPERATOR_TRIAGE_GATE`). Record the keep-bit: the babysitting counter (touches per witnessed shipped unit) over the soak; a rise reverts the seat.

#### In scope
The per-class ratchet gate; the advisory-confirm path; the fail-closed rule; the soak switch; the keep-bit measurement + verdict hook.

#### Out of scope
The input projection (child 1) and the action lowering (child 2). Changing the ratchet's earning formula (#2274 owns it) or the counter's definition (#2269 owns it).

#### Done condition
Each action class is gated by earned envelope width; sub-envelope actions require confirmation and super-envelope actions run unattended; a withheld witness forces escalation; a soak produces a keep-or-revert verdict from the babysitting counter.

#### Witness
`go test` green; a test drives an action across the ratchet threshold (advisory below, unattended above) and asserts fail-closed on `witness_refused`; a fixture computes the keep-bit verdict from a synthesized counter series.

#### Acceptance gate
Done condition + `make ci` green; `warn`→`enforce` soak recorded on the epic.

#### Work unit
One worker owns the gate, the soak switch, and the keep-bit hook.

#### Expected steps
5

#### Assumptions
- #2274's advisory ratchet exposes a per-class width this can read; if not, a thin adapter is in scope, the formula is not.

#### Confusion risks
- Fail-closed is non-negotiable: a missing witness must ESCALATE, never license an unattended action "because the fleet looked fine." A green *absence* is not a green *witness*.

#### Coordination
- `internal/guardvars`/operator-brief + the ratchet surface — arbitrate the lane.

#### Lane
guardvars

#### Likely files
- `internal/supervisoragent/`
- `internal/guardvars/`

#### Closure binding
The threshold, fail-closed, and keep-bit witnesses above — green on a merged PR whose body carries `supervisor-agent/envelope-keepbit` — close this leaf.

#### Trigger
Epic `supervisor-agent/epic`.

## Honesty fences

- **Filed 2026-07-13** as GitHub issues **#4477 (epic) · #4478 (input contract) ·
  #4479 (action vocabulary) · #4480 (envelope + keep-bit)** on
  `anthony-chaudhary/fak`, with the children linked under the epic's checklist.
  Their contract score was measured before filing (see "Contract verdict" above —
  three children `ready`/100, epic 100/parent). The bodies below are the verbatim
  filed text; the dedup markers keep any re-file idempotent.
- The dedup sweep read #2038/#3517/#4347 in full and search-swept the tracker for
  the agent-supervisor framing; it did not exhaustively read every open epic, so a
  prior art issue is *possible* though not found. The dedup markers make a
  re-file idempotent if one surfaces.
- The claim that the middle-tier decisions are "decidable from witnessed state
  without a transcript" is an **assumption stated to be falsified** (Child 1). If a
  decision class provably needs transcript content, it belongs in the human
  residual (B7), not this seat — that finding would *narrow* the epic, correctly.

## Next checkable step

Done: the epic + three children are filed and open (#4477–#4480), children linked
under the epic. Next is dispatch — a worker takes #4478 (input contract) first,
since #4479 and #4480 build on the typed projection it defines. The keep-bit (the
babysitting counter across a `warn`→`enforce` soak) is the falsifiable test of
whether the seat should exist at all; if it does not lower touches-per-shipped-unit,
the epic is reverted.
