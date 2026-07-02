---
title: "No babysitting: nobody watches a healthy fleet"
description: "The attention doctrine for unattended agents and fleets: babysitting is polling, and the kernel's job is to convert every poll into a rare, cheap, evidence-carrying interrupt. Decomposes supervision labor into seven watches, shows which ones better models retire and which they intensify, and binds the guard, witness, arbitration, recovery, budget, and notification programs to one falsifiable property with a ratchet."
date: 2026-07-01
---

# No babysitting: nobody watches a healthy fleet

Status: concept note + doctrine. Nothing new ships from this note; it binds
programs that already exist (guard/policy floor, DOS witnesses, lanes/leases,
resume watchdog, loop ledger [`LONG-RUNNING-AGENT-LOOPS`](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md),
perpetual sessions #1860, automatic context #2198, turntax #1147, operator
brief/heaviness) to one falsifiable product property, and names the rungs that
are missing between them. It is the umbrella over
[`CONCEPT-AUTOMATIC-CONTEXT`](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md): manual
context management was one *species* of babysitting; this note is the genus.

## The operator's ask

> fak should be good at helping operators, agents, users etc not have to
> "babysit" agents and fleets. Deeply think about this. Of course it can't do
> everything. Expect models, harnesses, our harness etc to keep getting better.
> But even on that slope, think about things.

## The thesis

Babysitting is **polling**. An operator who glances at a terminal every twenty
minutes, re-reads a transcript to see if the worker is stuck, spot-checks a
"done" message against the diff, or keeps a dashboard open "just in case" is a
human running a busy-wait loop over a system that has no interrupt controller.
Every mature computing layer killed its version of this: the OS turned device
polling into interrupts, memory overlays into page faults, process-watching
into SIGCHLD and supervisors. The agent era is still in the polling stage, and
the poll is running on the most expensive, least scalable component in the
system — human attention.

So the doctrine, in its falsifiable form:

> **Nobody watches a healthy fleet.** A human touch whose trigger the kernel
> could have decided from structure — a crash, a stale or unwitnessed claim, a
> collision, budget burn without evidence, a question the policy floor already
> answers — is a defect. Count the touches per witnessed unit of shipped work.
> The count is a ratchet that only goes down.

The general cure is the same three-step every time, and it is the move fak has
already made twice (security: structure over recognition; context: cells over
knobs):

1. **Name the condition** in a closed vocabulary (a token, not prose).
2. **Attach a witness** that decides it mechanically (evidence, not
   self-report).
3. **Route the interrupt** as a typed packet with a safe default (a decision,
   not a firefight).

Babysitting persists wherever one of those three is missing. It is a symptom
of a missing witness, never a duty.

## The seven watches (what a babysitter actually does)

"Babysitting" bundles seven distinguishable jobs. They have different
mechanics, different owners, and — the important part — different fates on the
model-quality slope.

| # | Watch | The human poll it replaces | The kernel mechanism |
|---|---|---|---|
| W1 | **Liveness** | "is it still moving, or hung/crashed/looping?" | heartbeats, watchdog, resume, loop ledger `stuck_age` |
| W2 | **Truth** | "did it really do what it said?" | witnesses: `dos verify`, `commit-audit`, ship stamps, claim-check |
| W3 | **Safety** | "will it do something terrible while I'm away?" | capability floor, gitgate, quarantine, decision journal |
| W4 | **Collision** | "are they stepping on each other?" | lanes/leases, `dos arbitrate`, `COLLISION_RISK`, sweep |
| W5 | **Budget** | "is it burning money without progress?" | budgets, turntax envelopes, burn-without-evidence |
| W6 | **Unblock** | "is it waiting on me and not saying so?" | closed refusal vocabulary, ESCALATE disposition, `fak notify` |
| W7 | **Quality/goal-fit** | "is the work actually good — and what I wanted?" | `dos review` residual bands; the rest is human on purpose |

Two symmetries worth naming. First, the goal says *operators, agents, users*:
an orchestrator babysitting its subagents runs the same seven polls one level
down, which is why every mechanism above must be agent-consumable (the MCP
`dos_*` verbs; `dos_status` returning **no `claimed` field by construction** is
the fail-closed A2A form). Second, each watch is the same shape: a condition
the operator polls for because nothing will interrupt them with evidence when
it becomes true.

## The slope: what better models retire, and what they intensify

The instruction says to assume models, harnesses, and our harness keep
improving. Take that seriously and the watches split cleanly.

**The slope retires (fak should not build here):**

- Most of **W7 quality**. Better models write better code, catch their own
  mistakes, self-review. Any fak investment in "recognize a confused agent by
  reading its transcript with a judge model," "nudge it with a better prompt,"
  or "grade the code smarter than the model that wrote it" is a recognizer in
  an arms race against the thing improving fastest. Structure over recognition
  was the security lesson; it is also the roadmap filter.
- Most of **W6 frequency**. Smarter agents need unblocking less often. (Note
  the inversion below: the escalations that *remain* are higher-stakes, so the
  cost per escalation matters more even as the count falls.)

**The slope intensifies (fak's durable seat):**

- **W2 truth.** The trust asymmetry is structural, not a capability gap: a
  self-report carries no evidence *at any intelligence level* — the verifier
  still pays the verification cost. Worse, capability improves the camouflage:
  a stronger model's wrong answer is more plausible, rarer, and better argued,
  so human spot-checking gets *less* effective per unit of attention exactly as
  the fleet's output volume grows. Witnessed completion appreciates on the
  slope. This is the *Mata v. Avianca* lesson generalized: the failure class is
  not dumb output, it is confident unwitnessed output.
- **W3 safety.** Expected loss = error rate × action volume × blast radius.
  The slope shrinks the rate but grows the volume (more agents, more turns) and
  the privilege (prod access, money, comms — you delegate more *because* the
  model is better). And the adversary rides the same slope: injection payloads
  are model-authored too, which is why detection stays ~100% evadable
  (`STATUS.md` §6) and the *containment* floor is the load-bearing guarantee.
  Nobody removed memory protection because compilers got better.
- **W4 collision.** Coordination is a system property, not an intelligence
  property. Two brilliant agents still collide on one file without a lease;
  better models mean *more* concurrency and faster turns, so more races, not
  fewer. The shared-trunk experience already shows this at fleet scale
  (`SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md`).
- **W1 liveness.** A hung process cannot tell you it is hung — self-report
  fails *definitionally* here. And the crash causes are infrastructure-shaped
  (auth expiry, 529 bursts, OOM, network), orthogonal to model quality. The
  evidence is one day old: the
  [2026-07-01 bottleneck map](BOTTLENECK-MAP-2026-07-01.md) has the fleet
  **CRITICAL from the recovery layer** — 18 workers stuck in resume/surface
  queues — while auth-blocked and rate-limited both sit at **zero**. The models
  did not fail; the babysitting layer did.
- **W5 budget.** Unit prices fall, volume rises (Jevons); what stays constant
  is that *burn without evidence* is only detectable by something that can see
  both the spend and the witnessed effects. That join lives at the kernel.
- **W6 severity.** The residual escalations concentrate into genuinely hard
  decisions. An interrupt that arrives as a page of prose ("it seems blocked,
  see transcript") costs the human thirty minutes of context reconstruction;
  the same interrupt as a typed packet costs thirty seconds. As frequency
  falls, packet quality dominates total cost.

The economics underneath all of it: **fan-out outruns attention.** Better
models make competent work cheaper, so fleets grow; the operator's attention
budget does not. Total babysitting cost is
`Σ events × P(needs human | event) × cost(handling)`. The slope grows the
event count; fak's leverage is the other two factors — drive
`P(needs human)` down with structure (W1–W5) and drive `cost(handling)` down
with packets (W6). An operator's touches should scale like O(log fleet), not
O(fleet). This is the SRE toil argument, ported: babysitting is toil, and the
kernel is the toil-elimination layer.

One more slope effect, from the harness direction: as harnesses absorb
autonomy features (auto-compaction, background tasks, auto-resume, hooks),
each new feature is *another manager on the wire* — the `compactcohere` lesson
generalizes. The kernel's seat survives harness improvement precisely because
it sits below all 41-of-47 harnesses at the wire and beside the evidence (git,
ledgers), which is where every watch's witness lives. Harness autonomy
features change *who acts*; they do not change *what must be witnessed*.

## What exists at HEAD (the honest map)

Per-watch, from a repo survey (maturity per `CLAIMS.md`; details and paths in
the source files named):

- **W1** — `fak resume plan|watchdog` (cross-account, ledger-first, outcome-
  gated), `fak watchdog` autoheal with debounce + attempt caps, `fak loop
  recover` for orphaned runs (ledger-only; no pid-liveness probe yet), session
  images/rehydrate, horizon-gated dormancy recovery. Shipped, but the live
  fire is exactly where the bottleneck map found the fleet CRITICAL: the
  watchdog *runs* and the queue *is not draining* — LIVE-mode drain is unproven.
- **W2** — `dos verify` / `commit-audit` / `review` (CLEARED/RESIDUAL bands),
  the `(fak <leaf>)` stamp grammar, `CLAIMS.md` + lint, `fak claim-check`, the
  verification ladder with first-class INDETERMINATE, `LOOP_DONE_UNWITNESSED`.
  The most complete watch.
- **W3** — the guard capability floor, gitgate, quarantine/ctxmmu, the
  hash-chained decision journal, repair. Shipped and default-on under
  `fak guard`.
- **W4** — lanes/leases/`dos arbitrate`, the lease-disjointness steward
  (collisions=0 witnessed), `fak sweep`, the dispatch capacity equation.
  Shipped locally; the shared task record and A2A channel are in-process only
  (durable cross-process delivery is [STUB]).
- **W5** — session/token/time budgets with closed stop-reasons, turntax
  overhead envelopes (declared, not measured p99), `loop_budget_burn_without_
  evidence` named as a KPI in the loop note but not yet emitted.
- **W6** — the closed refusal vocabulary with RETRYABLE/WAIT/ESCALATE/TERMINAL
  dispositions, `fak notify` (SIGCHLD for sessions), `fak operator brief
  --check` (pages only on human items), heaviness score. The pieces exist;
  there is no single typed escalation object they all emit.
- **W7** — `dos review` shrinks review to the residual; goal-fit itself is
  deliberately human.

## The doctrine

- **B1 — Poll → interrupt.** Any human or agent habit of *periodically looking
  at a healthy fleet* is a defect to file, exactly as a context-management
  instruction is under #2198. A dashboard is a polling UI — legitimate for
  debugging and for the curious, never load-bearing for the default loop. If
  an operator must check it to be safe, the missing interrupt is the bug.
- **B2 — Green means witnessed, not quiet.** No surface may present "no news"
  as health. A fleet member is healthy when its liveness rung is fresh and its
  last claim is witnessed — both positive evidence. "The worker stopped
  talking" is never a success state (`witnessed_done` vs `claimed_done`, per
  the loop note's four completion states).
- **B3 — Every interrupt is a packet.** A human is interrupted only with: a
  closed-vocabulary reason, the minimal state, the evidence refs, a bounded
  set of actions, a safe default, and the cost of delay. The packet must be
  decidable in seconds without opening a transcript. Prose escalations are the
  W6 equivalent of a manual overlay.
- **B4 — Silence is bounded.** Every unattended run has a maximum silent age;
  past it, the kernel emits a liveness interrupt on its own. Detection latency
  is a first-class SLO ("silent hours"), because on the slope the cost of a
  crash is dominated by how long nobody noticed, not by the crash.
- **B5 — Autonomy is earned mechanically.** The reason humans babysit is that
  trust is binary and vibes-based. Make it a ratchet: an envelope (policy
  floor width, budget, notification threshold, dispatch cap) widens only on
  witnessed track record and contracts automatically on a witness refusal.
  Trust becomes a number derived from the ledger, not a feeling — and
  contraction, like every fak gate, fails closed.
- **B6 — Recovery is the default path.** A crash is a non-event: detected
  within the silence bound, resumed by the watchdog, re-verified by the
  witness, visible only as a ledger row. A crash that reaches a human is the
  exception that names a bug in the recovery layer.
- **B7 — The residual is human on purpose.** Goal-fit, taste, architecture,
  and the decision to delegate more are not automatable by this doctrine and
  are not meant to be. The kernel's whole job is that these arrive as clean
  decisions at dispatch and review time — never as discoveries during a
  firefight. "Can't do everything" is a design input, not an apology.

## The babysitting counter (the doctrine's witness)

Like the manual-overlay counter in #2198, the doctrine is checkable only if
the labor is counted. Four KPIs, all derivable from ledgers that already exist
or are named rungs:

| KPI | Definition | Source |
|---|---|---|
| `touches_per_witnessed_unit` | human interventions ÷ witnessed-done runs (or stamped commits), each touch classified W1–W7 | loop ledger + operator brief |
| `silent_hours` | Σ time a stuck/dead member sat undetected | liveness rungs vs ledger `end` rows |
| `mttr_sessions` | detection → witnessed-resumed, p50/p95 | resume watchdog ledger |
| `escalation_handling_p50` | packet emitted → human decision recorded | notify/ack contract |

The first is the headline ratchet. Today it is not measurable — which is
itself the finding: fak can count kernel decisions per session
(`131 decisions — 121 allowed…`) but cannot yet count *human* touches per unit
of shipped work. You cannot retire what you cannot count.

## The rungs (new here, not duplicating #1860 / #2198 / the loop note)

- **R1 — The babysitting counter.** Emit the four KPIs from the ledgers named
  above; classify each operator touch by watch. Cheapest rung; everything else
  is graded against it. (The dispatch and resume ledgers already record most
  of the denominator.)
- **R2 — The escalation packet.** One typed object (`fak.escalation.v1`) that
  `fak notify`, the refusal ESCALATE disposition, the operator brief, and the
  loop note's notification contract all emit and consume: reason token, state
  digest, evidence refs, bounded actions, safe default, cost-of-delay. B3 as
  schema instead of style.
- **R3 — The waiting-on-human queue.** Every blocked-on-operator item as a
  kernel object in one queue with age, held resources (idle workers, leases),
  and the safe default that fires on expiry. Babysitting inverted: the fleet
  files tickets on the human, with deadlines — the human never scans for them.
- **R4 — The recovery drain witness.** Prove, not assume, that the resume
  watchdog in LIVE mode drains its queue: a witnessed MTTR row per recovered
  session and a red steward when `AUTO_RESUME` depth grows monotonically for
  N ticks. This is the live fire named CRITICAL in the 2026-07-01 bottleneck
  map, and it is W1's B6 obligation.
- **R5 — The autonomy ratchet.** Bind envelope width to ledger-derived witness
  rate: a loop with N consecutive witnessed-dones and zero refusals earns a
  quieter notification threshold and a wider budget; one `witness_refused`
  contracts it a level, automatically. Start advisory (report what the ratchet
  *would* do), enforce later — the maturity-ladder pattern.
- **R6 — The unattended soak witness.** A standing N-hour fleet run whose
  report proves: zero human touches, zero silent-hour breaches, all
  completions witnessed, all interrupts packet-shaped. The CI form of "nobody
  watches a healthy fleet," alongside the flat-context soak of #2198-R8.

## What not to build (the slope will eat it)

- No transcript-reading "is it stuck?" judge models — liveness is a rung, not
  a vibe (and a judge is a recognizer in an arms race).
- No prompt-therapy layer that re-prompts a struggling agent "better" — that
  is the harness's and the model's job, and they are improving faster than we
  could tune it.
- No smarter-than-the-model code grader — W7 belongs to the slope; fak only
  routes the residual.
- No load-bearing dashboard — per B1, a status UI that must be watched is
  institutionalized polling with better typography.
- No per-harness babysitting scripts — every watch mechanism lands at the
  wire/ledger seam where it covers all harnesses at once, or it becomes N
  scripts to babysit.

## Honesty fences

- Nothing ships from this note. Every maturity label above is from
  `CLAIMS.md`/`STATUS.md` at HEAD or the named source file; the gaps
  (loop-recover pid probe, A2A durable delivery, declared-not-measured
  overhead budgets, un-emitted burn-without-evidence KPI) are carried from
  those ledgers, not resolved here.
- The four KPIs are *definitions*, not measurements. `touches_per_witnessed_
  unit` has no baseline yet; R1 exists to create one. No number in this note
  is a claim of current performance.
- B5's ratchet is a design assertion; whether witness rate is a sufficient
  statistic for safe envelope expansion needs adversarial review (a worker
  could farm easy witnessed-dones — the ratchet must weight by work class, and
  that weighting is open design).
- The slope analysis is an argument, not a measurement. Its falsifiable core
  is the counter: if better models alone drive `touches_per_witnessed_unit`
  to zero without W1–W5 structure, the doctrine is wrong — and the counter
  would show it.
- No GitHub issues are filed from this note: the same bottleneck map records
  a live issue-taxonomy regression caused by unlabeled epic bursts, so filing
  the R1–R6 spine without priority/kind/area/owner would worsen the exact debt
  it documents. Filing them *with* labels is the named next step, not a
  casualty of it.

## Next checkable step

Build **R1** (the counter) first — like #2198-R1, it is the cheapest rung and
the doctrine's own witness; R4 (the recovery drain witness) is the live fire
to point it at, since the fleet is CRITICAL there today. Check: the four KPIs
appear in an operator-readable report with a dated baseline, and the
2026-07-01 bottleneck map's 18 stuck workers become the first measured
`silent_hours` / `mttr_sessions` datapoints.
