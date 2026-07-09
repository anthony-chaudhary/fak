---
title: "Ideal working conditions for agentic processes"
description: "A doctrine note that treats an agentic dev process as a worker whose output is bounded by its working conditions, not just its model IQ. Names the specific limits of three layers — the model, the harness, and our own fleet — and maps each limit to the working-condition commitment it demands, grounded in shipped fak surfaces with the gaps marked not-yet."
---

# Ideal working conditions for agentic processes

*Concept note, 2026-07-09. No new mechanism ships here — this is the doctrine
that names why the mechanisms already in the tree exist, and where the next
ones are owed. Every number is fenced; every surface is a real one you can run.*

## The respect principle

An agentic dev process is a **worker**, not an oracle. Its throughput and the
quality of what it commits are bounded at least as much by the conditions we put
it in as by the model's raw capability. A worker given a broken workbench, a
5-second wait on every hand tool, a notebook that gets wiped every few hours, and
a shared bench three peers are also sawing on will produce bad work — and the
mistake is to read that bad work as *the worker is dumb* rather than *the bench is
broken*.

"Treat agentic devs with more respect" is not sentiment. It is an engineering
stance with a testable consequence: **before attributing a failure to the model,
account for the working condition.** Most of the fleet incidents we have
witnessed — crash-loops, amnesia, empty commits after clean turns, refusals on
free seats — were condition failures wearing a model's face. This note gets
specific about the limits so the conditions can be engineered instead of endured.

The limits sort into three layers, from the one we control least to the one we
control most:

1. **The model** — what the token engine can and cannot do, per model and per
   host. We mostly do not get to move these; we design around them.
2. **The harness** — the loop fak runs the model inside: context budget,
   compaction, restart, tool latency, refusal legibility. This is where most of
   the leverage is, because it is ours to shape.
3. **Us / the fleet** — account seats, host load, the shared trunk, hardware.
   The conditions we impose on ourselves by how we run many agents at once.

For each limit below: the specific limit (with an honest number where we have
one), what it does to the worker, the working condition it demands, and the fak
surface that provides it — or the gap, marked **not yet**.

---

## Layer 1 — the model

The model's limits are the hardest floor. The discipline is to know them
precisely so we neither over-ask a small model nor blame a capable one for a
condition failure.

### 1.1 Context window is finite, and the usable window is smaller than the advertised one

A model's context is a fixed budget, and the *usable* fraction is smaller still:
an output reserve, a resident working set, and the point past which recall
degrades all cut into it. fak sizes this explicitly rather than trusting the
headline number — `HardContextCap`, `MVC`, `MECW`, target resident budget, and
output reserve are named quantities with provenance labels (see
[`long-context-defaults.md`](../long-context-defaults.md)).

- **Working condition:** the worker should never have to hand-curate what stays
  in its window. fak re-plans the resident view every turn against a hard token
  budget, keeps recency + durability + predicted relevance, and — critically —
  every elided span is `Faithful`: cold, recoverable by a content-address handle,
  **never destroyed** (`internal/ctxplan`, see the
  [managed-context glossary §2](../managed-context-glossary.md)). A reference to a
  non-resident span is a typed page-fault outcome, not a silent hole.
- **Shipped.** The gap is that a small local model's window is genuinely tiny and
  no planner conjures more; there the condition is *route the work to a model
  whose window fits it*, which is a routing decision (`fak route`), not a context
  trick.

### 1.2 The faithful-model ceiling is a hardware fact, not a preference

On a 36 GB Mac, fak's bit-exact faithful path tops out at **≤ 7B** parameters —
above that the deterministic in-kernel engine does not fit. That is a capacity
*fact* (see [`docs/explainers/hardware-limits-and-capacity.md`](../explainers/hardware-limits-and-capacity.md)
and the profiled rows in [`docs/HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md)),
not a quality opinion. The eighth capacity assumption the explainer names is that
*the box can hold the model at all* — and it frequently cannot.

- **Working condition:** treat "no local capacity" as a **routing input, not a
  dead end**. Heavy/GPU/large-model work dispatches to the fleet's lab GPU
  resources; the laptop's lack of a GPU is never a reason to stop. A worker that
  hits a capacity wall should emit a typed "route elsewhere" signal, not a
  failure.
- **Shipped** for the routing seam and the honest matrix; the automatic
  "capacity-aware reroute" of a specific stuck task is closer to **not yet** than
  to done.

### 1.3 Per-model reliability is not uniform — a model can be capable and still starve on the wrong task

Model *capability* and model *fit for a harness slot* are different axes. Observed
on this fleet: a `claude-fable-5` worker ran ~13 clean turns on a hard, churning
issue and committed **nothing** — its context ballooned past the compaction
target, each overflow triggered a budget-restart that reseeded only a sliver of
context, and it looped into amnesia until it exhausted the restart limit. The same
model does surgical diffs cleanly. The model was not "broken"; it was **placed in
a harness slot its reliability profile could not hold** (witnessed; see the
`fable-worker-restart-amnesia` and `fable-cached-dispatch-knobs` fleet memories).

- **Working condition:** match the model to the *shape* of the work, not just its
  difficulty — reserve the fast/cheap model for bounded, surgical tasks and give
  churning, exploratory issues to a model with the headroom to survive the
  restart profile. This is a placement policy, and it is mostly **carried in
  operator memory today, not enforced by a gate — not yet.**

### 1.4 The provider prompt-cache discount is fragile by construction

The single biggest cost lever on the flagship route is the provider's own
prompt-cache discount, and it survives only while the prefix stays byte-identical
and the trajectory stays append-only. It decays toward zero along three axes:
editing/flexibility, per-turn tool-call density past the **20-block / 4-breakpoint**
budget, and cross-agent fan-out (see
[`docs/explainers/frozen-trajectory-cache-cliff.md`](../explainers/frozen-trajectory-cache-cliff.md)).

- **Working condition:** the worker should get the discount without managing it.
  fak holds the cached prefix byte-identical by splicing on the original bytes (a
  memcpy, never a re-marshal) so the discount survives history compaction, and it
  relays the provider's reuse number rather than claiming it (see
  [`docs/explainers/long-session-economics.md`](../explainers/long-session-economics.md)
  and [`context-shedding.md`](../explainers/context-shedding.md)). **Shipped.**

---

## Layer 2 — the harness

This is the layer with the most leverage, because it is entirely ours. Almost
every "the agent is dumb" incident on this fleet resolved to a harness-condition
bug.

### 2.1 The context budget must fit the actual working set, or the harness eats itself

A context-budget ceiling set *below* the loop's real baseline is not a saving —
it is a crash-loop generator. Witnessed: a flat **48K** `--context-budget-tokens`
(a compaction *target* mis-wired as a *drain ceiling*) crash-looped the fleet,
because the real launch baseline is ~**62K**. The fix derived the ceiling from the
window instead of freezing it — `min(baseline×2, ctxplan window − reserve)` —
with Go/Python golden parity (committed `8a0fcffbb`; see the
`dispatch-worker-context-budget-trap` memory). The deeper fix — measuring the real
launch prompt instead of a frozen baseline constant — is still **not yet**.

- **Working condition:** the budget a worker is handed must be *derived from what
  the loop actually needs*, never a round number pulled from a different concept.
  A budget below baseline is a broken workbench, full stop.

### 2.2 Compaction should shed the stale middle, not the working set — and prove what it shed

A growing transcript re-sends itself every turn; the naive fix (summarize
yourself) throws away the working set. fak compacts by shedding the un-cacheable
*middle* turns while keeping the cached head and the recent turns byte-for-byte,
leaving a restore handle behind (default-on `--compact-history-budget`; compaction
fires once a session sprawls past the ~48k budget). fak's *own* authored share of
the savings climbs with session length — ~1% on a short run to ~**11%** on the
longest actively-compacting session measured — a WITNESSED-shed / MODELED-%
direction from a thin sample, peak-not-pool (fenced in
[`long-session-value.md`](../long-session-value.md) and BENCHMARK-AUTHORITY.md).

- **Working condition:** the worker keeps its working set and its cache discount
  across a long session without being asked to summarize itself. **Shipped**, with
  the honest fence that on the flagship route the provider's discount dwarfs fak's
  authored slice.

### 2.3 A restart must not be an amnesia event

The cruelest harness failure is the one that looks like the model forgetting. A
budget-restart that reseeds only a sliver of context (observed as low as ~**520
tokens**) turns a capable worker into one that re-derives its task from nothing
every few turns and commits nothing (§1.3). The restart *mechanism* was starving
the worker, not the model.

- **Working condition:** a restart should carry the objective, the re-verifiable
  progress cursor, and the next action forward — a **baton**, not a blank page.
  fak has the durable half of this today: `ctxplan.ObjectivePin` carries the
  stated objective across a hidden reset with a content-address digest so
  "the objective survived" is a checkable equality, and `session.ResetTransaction`
  records every omitted span with a reason and digest (managed-context glossary
  §§3, 5). The *within-session* half is also already shipped: a compacted-away
  span leaves a content-address tombstone, and `fak_context_restore` pages it back
  on demand (`fak_context_spans` lists what is restorable, `fak_context_value`
  reports headroom plus a closed `step_advice`) — so a compacted worker is
  amnesiac by default but *curable*, the dropped bytes one call away rather than
  lost. (This note's own author recovered its originating task through exactly that
  path.) The **full** perpetual-session baton — objective pin + progress cursor
  + next action + externalize gate, rotating only at a safe point — is specified
  but **not yet** shipped (relay vocabulary, epic #1860;
  [`CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md`](CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md)).

### 2.4 A single transient wire blip should not kill a worker

A guarded worker is a child process with no auto-retry; observed, a single
transient `anthropic_messages` wire error killed workers at differing sequence
numbers (7, 30) — differing seq ⇒ transient, not systematic. The model was fine;
the harness had no shock absorber (witnessed; `fable-worker-transient-wire-crash-resilient-loop`
memory).

- **Working condition:** transient upstream failures should be absorbed by a
  resilient relaunch loop that rides the blip on a fresh lean resume prompt, not
  charged to the worker as a crash. This pattern is **operator-carried today**
  (a bounded relaunch loop), not a shipped harness guarantee — **not yet.**

### 2.5 Tool latency is a working condition — a slow syscall starves the loop

When the host is loaded, individual git calls run **5–9 s** each, and the
dispatch preflight (which makes several) times out past 60 s and refuses *every*
spawn path with `REFUSE_INSPECT` — on free seats, with a healthy kernel. The
worker is refused not because it did anything wrong but because its tools got
slow (witnessed; `dispatch-refuse-inspect-host-load` memory).

- **Working condition:** the tools a worker calls should be fast and bounded. fak
  serves repeated reads locally and repairs malformed calls in place so a wasted
  turn is avoided (llms.txt "fewer wasted turns"), and the recent
  `safe_ff_sync git()` timeout bound (commit `b649fc74d`) is exactly this
  discipline: **a hand tool that cannot hang.** The general guarantee — every
  syscall on the worker's path has a bounded latency budget under host load — is
  partial; **more of this is owed.**

### 2.6 "No" must be a legible, typed value — not a hang and not prose

A worker that is blocked deserves to know *why* in a word it can act on. fak's
refusal vocabulary is **closed**: a proposed tool call gets a typed verdict before
it runs, and a refusal names a reason from a fixed set (`dos_refuse_reasons` /
`dos_check_reason`; the MCP tool-result wire in
[`docs/mcp-tool-result.md`](../mcp-tool-result.md)). Every reason is
simultaneously emittable, verifiable, and refusable — so "no" routes to a replan
instead of a dead end. (This note itself hit a `TRUST_VIOLATION/ESCALATE` refusal
mid-draft; the point is that it arrived as a typed, actionable value, not a hang.)

- **Working condition:** every block a worker can hit is a named reason with a
  fix, never a silent stall or a wall of prose. **Shipped** as the vocabulary;
  the ongoing work is keeping the set closed as new gates land (the recurring
  `UNCLASSIFIED` prose-drift is the thing to keep killing).

### 2.7 The worker must be able to tell truth from self-report

The deepest harness condition, and the one most specific to *this* fleet. A worker
that cannot verify anything is forced to trust every claim — its own past "done," a
peer's status line, a recalled memory. That is exactly how a plausible-but-false
"shipped" propagates: an `--allow-empty` commit, a "tests pass" that deleted the
assertions, a memory that names a flag deleted three commits ago. Respect here is
*epistemic*: give the worker instruments, not vibes.

- **Working condition:** every load-bearing claim is checkable against a witness —
  git, a ledger, the working tree — never a self-narration, and "claimed" and
  "witnessed" must never be allowed to conflate. fak ships the witness verbs:
  `dos_verify` (did (plan, phase) actually ship, from artifacts), `dos_commit_audit`
  (does a commit's diff match its claim), `dos_status` (a peer-readable run digest
  with **no `claimed` field by construction** —
  [`status a peer can trust`](../explainers/status-a-peer-can-trust.md)), and
  `dos_recall` (is a recalled memory still true against git now). The loop kernel
  makes the split structural: a run ends in one of four *witnessed* completion
  states — `claimed_done`, `witnessed_done`, `witness_refused`,
  `witness_unavailable` — so a bare claim can never masquerade as a witnessed one
  ([long-running agent loops](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md);
  [verification-ladder doctrine](verification-ladder-doctrine.md)). **Shipped** —
  this is the condition fak is *most* complete on, and not by accident: it is the
  one that most respects the worker, because it stops asking the worker to trust
  and lets it check.

---

## Layer 3 — us / the fleet

These limits we impose on ourselves by running many agents at once. They are the
easiest to mistake for capability failures because the worker looks stuck while
nothing is wrong with it.

### 3.1 The real cap is account seats, not worker slots

A worker refused with `REFUSE_NO_ACCOUNT (0/24 live)` is not hitting a
worker-count limit — it is hitting the **account budget**. Fable shares the Opus
pool's rolling session caps; only ~**7** seats serve in practice, often 0 even
with session room, and reaping parked workers does **not** return budget
(witnessed; `fable-blocked-on-session-cap` memory). Seats can be free and dispatch
still refuses (§2.5).

- **Working condition:** respect the refuse — arm a bounded park-and-retry, do not
  burst (bursting spikes preflight load and turns `REFUSE_NO_ACCOUNT` into
  `REFUSE_INSPECT`). Surfaces: `fak accounts next` / `fak accounts headroom` to
  read seat state before spawning. **Shipped** as the read surfaces; the
  automatic backpressure that stops a loop from bursting into its own refuse is
  partial.

### 3.2 The shared trunk is a shared bench — one crashed worker can poison it for all

The fleet shares one working tree and one index. A worker that crashes after a
write can strand a **non-compiling file** in the shared tree, and every *other*
worker then reports `CHILD_CRASH exit 1` against a healthy pool — a fleet-wide
crash-loop caused by one stranded edit (`go build ./...` names it;
`git stash push -- <file>` un-poisons it; witnessed, `fleet-poison-tree-crash-loop`
memory). Symmetrically, a bare `git commit` sweeps peers' pre-staged files.

- **Working condition:** workers must commit **by explicit path** (`git commit -s
  -- <paths>`, never `git add -A`) so one worker's landing never sweeps another's
  bench, and a crashed worker's uncompilable residue must be reverted rather than
  left to poison peers. The commit-by-path discipline is **enforced** (the trunk
  guard); the **automatic revert of a crashed worker's uncommitted poison is the
  root gap — not yet.**

### 3.3 Host load is a shared resource the fleet can exhaust against itself

Process sprawl is a real condition: ~180 python processes make every git call
slow (§2.5), and Windows terminal handle sprawl degrades the whole desktop until
the only fix is an operator-only reboot (witnessed; `desktop-lag-is-sprawl`,
`resume-consolidate-duplicate-loops` memories — a long session accumulates
overlapping detached loops across compactions, and `TaskStop` ≠ kill on Windows).

- **Working condition:** a bounded number of live workers, one drainer, and no
  orphaned duplicate loops burning seats. This is **operator-carried hygiene
  today** (enumerate by command line, keep the one live drainer, stop the rest);
  a fleet-level admission control that caps concurrent load is **not yet**.

---

## The map, folded

| Layer | Specific limit (fenced) | Working condition it demands | fak surface | Status |
|---|---|---|---|---|
| Model | Usable context < advertised | Re-planned resident view, faithful (recoverable) elision | `internal/ctxplan`, `long-context-defaults.md` | shipped |
| Model | Faithful engine ≤ 7B on 36 GB Mac | Treat "no local capacity" as a reroute input | `fak route`, HARDWARE-MATRIX | seam shipped; auto-reroute **not yet** |
| Model | Capable ≠ fits every harness slot | Match model to task *shape*, not just difficulty | operator memory | **not yet** (unenforced) |
| Model | Prompt-cache discount is fragile | Byte-identical prefix preserved for you | byte-splice compaction | shipped |
| Harness | Budget below baseline crash-loops (48K<62K) | Budget derived from real working set | derived ceiling `8a0fcffbb` | partial; measure-real-prompt **not yet** |
| Harness | Restart = amnesia (~520 tok reseed) | Carry a baton, not a blank page | `ObjectivePin`, `ResetTransaction` | durable half shipped; full baton **not yet** (#1860) |
| Harness | Single wire blip kills the worker | Absorb transients in a relaunch loop | operator relaunch loop | **not yet** (operator-carried) |
| Harness | Slow tools starve the loop (5–9 s git) | Bounded-latency syscalls | local reads, `safe_ff_sync` timeout | partial |
| Harness | Blocks must be legible | Typed, closed refusal reasons | `dos_refuse_reasons`, mcp-tool-result | shipped |
| Harness | Can't tell truth from self-report | Every claim checkable vs a witness; claimed ≠ witnessed | `dos_verify`/`commit_audit`/`status`/`recall`; bgloop 4-state completion | shipped |
| Fleet | Seats, not slots, are the cap | Respect the refuse; park-and-retry | `fak accounts next/headroom` | read shipped; backpressure partial |
| Fleet | Shared trunk poisons on one crash | Commit by path; revert crashed poison | trunk guard | commit-by-path enforced; auto-revert **not yet** |
| Fleet | Fleet exhausts host against itself | Bounded live workers, one drainer | operator hygiene | **not yet** (unenforced) |

## Is this measured? Yes — the `agent-readiness` scorecard

This note is doctrine, but the working-conditions idea is not free-floating: it is
already a *measured* program in the tree. The `agent-readiness` scorecard
(`.claude/skills/agent-readiness/SKILL.md`, backed by
`tools/agent_readiness_scorecard.py`) is the shipped instrument, and it is
deliberately the mirror image of the operator-load side:

- **`friction-debt`** — a baseline gate, lower is better, floor 0. It counts the
  room's defects: the affordances an agent is *forced to work around*. The tree
  currently sits at **0 / grade A**
  ([agentic-dev observability score](AGENTIC-DEV-OBSERVABILITY-SCORE-2026-07-01.md)).
- **`experience-frontier`** — unbounded, higher is better. It counts the real
  working affordances an agent *gains* — the conditions this note argues for, made
  countable.

The scorecard names the symmetry itself: it is *"the deliberate mirror of
`internal/heavinessscore`'s unbounded `heaviness_pressure` (the load an operator
carries); this is the surface an agent gains."* So both sides of the same loop have
a number — `heaviness_pressure` for the human's burden (`fak operator heaviness`,
`human_metric = max(0, 100 - heaviness_pressure)`) and
`friction-debt`/`experience-frontier` for the agent's conditions. This note is the
*why* behind those numbers; the scorecard is the *how much*, and it is how a "not
yet" row above eventually shows up as debt paid down.

## The anti-pattern this note exists to kill

**Blaming the worker for the workbench.** Every row above has a failure mode that
*looks* like the model being incapable: the amnesiac worker (a starved restart),
the worker that commits nothing (a budget below baseline), the worker refused on a
free seat (a slow preflight), the fleet-wide crash-loop (one stranded file). The
respect principle is the debugging order: **check the working condition before you
attribute the failure to the model.** A large fraction of "the agent is dumb"
turns out to be "the bench is broken," and only the second one is ours to fix.

The generalization of fak's own thesis: the tool call is a syscall, the model
proposes and the kernel disposes — and a good kernel does not just *gate* its
processes, it gives them a *runnable environment*. Ideal working conditions for an
agentic process are the same thing a good OS gives a program: enough memory that
fits (context budget), fast syscalls (tool latency), state that survives a context
switch (the baton), a scheduler that does not thrash (fleet admission), and errors
returned as values instead of hangs (typed refusals). Where those are shipped,
we say so; where they are **not yet**, this note is the standing list of what the
worker is still owed.

## See also

- [Hardware limits and capacity](../explainers/hardware-limits-and-capacity.md) — capacity as the eighth assumption; the faithful-model ceiling in full.
- [The managed-context glossary](../managed-context-glossary.md) — the resident view, objective pin, budget envelope, and reset transaction that provide Layer-2 conditions.
- [Perpetual sessions](CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md) — the baton/relay that closes the restart-amnesia gap (§2.3).
- [Human operator effectiveness](../human-operator-effectiveness.md) — the human side of the same contract: keep the operator from drowning so the operator can keep the workers' conditions good.
- [No babysitting](CONCEPT-NO-BABYSITTING-2026-07-01.md) and [Agent friction complaint channels](CONCEPT-AGENT-FRICTION-COMPLAINT-CHANNELS-2026-06-29.md) — the "give the worker a voice about its conditions" adjacent surfaces.
- [Bottleneck map](BOTTLENECK-MAP-2026-07-01.md) and [Long-running agent loops](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md) — the throughput-side companions to this quality-side doctrine.
- [Engineering is building loops](../explainers/engineering-is-building-loops.md) — the loop framing this note gives working conditions to.
- [The status a peer can trust](../explainers/status-a-peer-can-trust.md) and [verification-ladder doctrine](verification-ladder-doctrine.md) — the witness spine behind §2.7 (truth vs self-report).
- [Agentic-dev observability score](AGENTIC-DEV-OBSERVABILITY-SCORE-2026-07-01.md) — the shipped numbers (`friction_debt=0`, agent-readiness A) this note is the doctrine behind.
