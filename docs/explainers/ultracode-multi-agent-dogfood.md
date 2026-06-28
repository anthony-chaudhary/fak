---
title: "Ultracode: a concurrent agent fleet as an operational mode on fak's multi-agent substrate"
description: "Defines 'ultracode mode' — orchestrating N independent coding agents on disjoint file lanes of one live trunk — as a discipline composed from fak's already-shipped pieces (the D-007 coordination protocol, the trunk guard, disjoint-lease arbitration, Claude Code subagents/forks), not a new feature. Gives the honest, un-gameable definition of the agent-orchestration value metric (a concurrency factor with its Amdahl ceiling) and states plainly why that multiple must never be blended with fak's separate inference 5–10x claims."
---

# Ultracode: a concurrent agent fleet as an operational mode

This is the **framing and metric-definition** doc. It defines what "ultracode mode" is,
which shipped pieces it composes, and — the load-bearing part — the **value metric** by
which an ultracode run may claim a dogfood multiple, defined so it cannot be gamed. It
reports **no** run results: the orchestrator writes the results doc separately and cites
this one for its definitions. Nothing here invents a CLI command or kernel feature;
ultracode is a *way of using* pieces that already ship.

---

## 1. Definition

**Ultracode mode** is an *operational discipline*: orchestrating **N independent coding
agents working concurrently on disjoint file lanes of one repository**, integrated back
onto a single live trunk. It is not a new subsystem — it is a mode that composes pieces
fak already ships:

- **The D-007 multi-agent coordination protocol** —
  [`multi-agent-coordination-protocol.md`](../multi-agent-coordination-protocol.md), issue
  [#241](https://github.com/anthony-chaudhary/fak/issues/241). The in-kernel substrate the
  mode rides: addressed message passing (`internal/a2achan`), shared co-edited task state
  (`internal/sharedtask`), and wave collectives + declared topology (`internal/comm`,
  `internal/agenttopo`) — **all** folded through the same default-deny adjudicator that
  gates an ordinary tool call (the §2 invariant of that RFC). A `comm.Group` here is an
  ordered set of member agents whose ranks are a deterministic function of identity, and a
  `Split`/`SplitLane` binds each color to a `dos.toml` lane.
- **The trunk guard and disjoint-lease arbitration** — the kernel refuses
  structurally-decidable git hazards before `git` runs (`internal/gitgate`), and admits two
  workers concurrently only when their declared file trees are pairwise disjoint
  (`dos_arbitrate`). Lane overlap serializes by refusal at the arbiter, not by a lock.
- **The Claude Code harness layer** — at the harness, the same discipline is expressed as
  subagents / forks running concurrently, a shared task list the orchestrator owns, and the
  same trunk guard arbitrating who may write which files.

So "ultracode" names a *mode of operation* over these shipped pillars. There is no
`ultracode` binary, no new ABI surface, and no new kernel primitive. If a results doc ever
implies a new feature shipped, it is wrong — the contribution is the *assembly and
discipline*, exactly as the rest of fak's multi-agent story is "no single lever is novel;
the contribution is the assembly."

---

## 2. Why disjoint lanes (the shared-trunk constraint)

fak runs a **single live trunk (`main`)** that every session commits to directly — no
per-agent branch, no per-agent worktree, no VM. The full trade is argued in
[`notes/SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md`](../notes/SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md);
the operational consequence for ultracode is sharp:

- **The git index and worktree are shared across sessions.** A bare commit or `git add -A`
  in one agent sweeps a peer's staged files. So agents must touch **disjoint files**, and
  integration is **commit-by-explicit-path** — each lane lands exactly its own files and
  nothing else.
- **Lane arbitration decides who may write what.** `dos_arbitrate` admits concurrent
  workers only when their file trees are pairwise disjoint; the `comm` lane lease
  coordinates *who may write which files* and **moves no bytes** (the honest line from
  `comm-as-mpi-split.md`). Overlapping lanes serialize by refusal rather than corrupt the
  tree.
- **The orchestrator owns integration.** Concurrency is safe *during* the run because lanes
  are disjoint and the trunk guard refuses hazards pre-call; the orchestrator then
  integrates and commits each lane's output by path. The human moves from a per-agent merge
  gate to **review of one live trunk** — review is not removed, it is relocated.

The mode therefore only fits work that *decomposes into disjoint lanes*. That is a design
constraint, not a detail (see §5).

---

## 3. The value metric (the core of this doc)

A dogfood run wants to claim a multiple. State exactly what that multiple is, and define it
so it cannot inflate itself.

### 3.1 The concurrency factor

> **Concurrency factor** = the number of **independent, reviewed, correct deliverables**
> completed in **one orchestration window**, where the serial-equivalent baseline is those
> same N deliverables produced by **N sequential agent runs**.

In plain terms: if an orchestration window produces N disjoint-lane deliverables that an
auditor could otherwise have obtained only by running N agents one after another, the
window's concurrency factor is N. The multiple is **agent-orchestration throughput**, not
quality and not tokens — it answers "how many independent units of reviewable work did one
window land at once," nothing more.

### 3.2 The Amdahl caveat (the ceiling)

The factor is bounded above by N and is reached only in the limit. The **serial part** is
the orchestration itself — decomposition into lanes, lane arbitration, and the integration
+ commit-by-path fold the orchestrator performs at the end. That serial fraction is real
overhead and it grows with N (the same synchronous-join tax `FANOUT-BENCH-RESULTS.md` §3
measures as a rising critical-path cost: the lead waits on the slowest lane, then folds
all N). So:

- The realizable multiple is `N / (serial_fold + parallel_work)`, **never** a flat N.
- Fanning out to **N = 1 is a net loss** — orchestration overhead with no sibling to
  amortize it across, exactly the fan-out finding restated for agents.
- The mode pays off only when there are several genuinely independent lanes to amortize the
  fold across.

State the realized factor *and* its serial-overhead context; never report a bare N as if
the fold were free.

### 3.3 What makes a deliverable "count"

A deliverable counts toward the factor only if it is **both**:

1. **Independently reviewable** — a concrete artifact (a file, a code change, a proof) that
   an auditor can open and read on its own, without trusting the agent's narration.
2. **Correct and complete, not a stub** — it does what it claims. A placeholder, a TODO, a
   doc that asserts a test exists when the diff added none (`dos commit-audit`'s
   `subject-only` failure), or a half-done lane does **not** count.

This is the anti-gaming clause: the factor measures *reviewed output*, so an orchestrator
cannot pump N by spawning more agents that each emit a stub.

### 3.4 This is NOT the inference 5–10x — keep the axes apart

fak makes a **separate** family of 5–10x claims about **inference** — decode/throughput on
real hardware (the H100 kernel roadmap and the session value stack), not agent
orchestration. Those numbers live in
[`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
(the single source of truth) and `docs/benchmarks/` (e.g.
[`H100-KERNEL-5X-ROADMAP.md`](../benchmarks/H100-KERNEL-5X-ROADMAP.md)). They are a
**different axis** — tokens per second / cost per token on a GPU — measured by an entirely
different method.

> **The one separation law.** The ultracode **concurrency factor** (independent reviewed
> deliverables per orchestration window) and fak's **inference** 5–10x (decode throughput /
> session-value-stack savings) are different axes measured by different methods. They must
> **never** be multiplied together, added, or quoted as one "5–10x." A results doc that
> blends them is making an unsupported claim.

This doc deliberately does **not** restate the inference numbers — it points to where they
live so the two are cited, never conflated.

---

## 4. Honesty rules

The metric obeys the proof method's one rule (`docs/proofs/00-METHOD.md`): a claim is real
only when a deterministic, re-checkable witness corroborates it — never because an agent
says so.

- **A SKIP or a gated artifact is not a delivered one.** If a lane's acceptance is
  host-gated, weights-gated, or GPU-gated and the run records a structured SKIP, that lane
  contributes **zero** to the concurrency factor — recorded honestly, never extrapolated
  into the count. (Same discipline as the fan-out long-context probe: record the ceiling,
  do not fabricate the wall-clock.)
- **The multiple counts only real, reviewed output.** An auditor must be able to open each
  counted deliverable and confirm it; a deliverable the kernel could not corroborate is the
  residual a human reviews, not a unit already banked.
- **Measured vs claimed stay apart.** "N agents ran" is not "N deliverables landed." The
  factor is the second, and only the second.

---

## 5. When NOT to use ultracode mode

The mode is wrong, or net-negative, when:

- **The work has heavy cross-file coupling.** If lanes cannot be made disjoint, agents
  collide on shared files; lane arbitration will serialize them by refusal and the
  concurrency factor collapses toward 1 — with the orchestration overhead still paid. Tasks
  that touch one hot shared file (a peer-contended source) belong to one agent, not a fleet.
- **Serial review is the bottleneck.** If a human must read and approve each deliverable in
  sequence, the review queue, not agent execution, sets the wall-clock — fanning out wider
  buys nothing past the review rate.
- **N is small (especially N = 1).** Below a few independent lanes the orchestration +
  integration fold costs more than doing the work directly (§3.2).

In those regimes a single agent on the trunk, or a smaller fleet, is the honest choice.

---

## 6. References

- **The substrate this mode rides:**
  [`multi-agent-coordination-protocol.md`](../multi-agent-coordination-protocol.md) (D-007,
  #241) · [`comm-as-mpi-split.md`](../comm-as-mpi-split.md) (the lane lease moves no bytes).
- **Why disjoint lanes / shared-trunk trade:**
  [`notes/SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md`](../notes/SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md).
- **The fan-out cost/quality discipline the concurrency factor inherits:**
  [`fleet-benchmarks.md`](fleet-benchmarks.md) ·
  [`benchmarks/FANOUT-BENCH-RESULTS.md`](../benchmarks/FANOUT-BENCH-RESULTS.md).
- **The separate inference axis (do not blend):**
  [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
  · [`benchmarks/H100-KERNEL-5X-ROADMAP.md`](../benchmarks/H100-KERNEL-5X-ROADMAP.md).
- **The honesty method the metric obeys:**
  [`proofs/00-METHOD.md`](../proofs/00-METHOD.md).
