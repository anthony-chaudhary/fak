---
title: "Agent-OS scheduling SOTA readout: lease, heartbeat, fence, schedule, autoscale"
description: "Research/SOTA readout for issue #906: the production orchestration prior art (Kubernetes Lease + leaderelection, Temporal task queues, Argo Workflows synchronization, Ray Serve autoscaling) mapped against fak's CURRENT scheduling primitives with file:line citations. Keeps the five decisions distinct — heartbeat (liveness), lease (admission), fence (stale-write rejection), schedule (overlap policy), autoscale (fair reallocation) — answers the issue's five questions from the real tree, and names the concrete gaps (no fencing/generation token in leaseref.Record, a defined-but-unproduced loop heartbeat row, no overlap coalesce/queue policy distinct from the cadence floor) as the implementation children for the remaining acceptance items."
---

# Agent-OS scheduling SOTA readout: lease, heartbeat, fence, schedule, autoscale

Date: 2026-06-26

Scope: the research(sota) deliverable of issue
[#906](https://github.com/anthony-chaudhary/fak/issues/906) — a single contract
note that compares the mature orchestration patterns (Kubernetes, Temporal, Argo,
Ray) against fak's *current* loopmgr / lease / heartbeat / session behavior, before
any more local scheduling semantics are added. It is a readout and a contract, not
new code. The implementation children it names (a fencing token, a witnessed
heartbeat row, an overlap policy) are scoped to follow-up leaves in `internal/`,
not landed here.

Parent strategy:
[`LONG-RUNNING-AGENT-LOOPS-2026-06-25.md`](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md)
already fixes the framing this note builds on — *schedulers and chat are interrupt
sources; fak owns loop identity, admission, leases, run ledgers, task snapshots,
witnessed completion, notification discipline, and cache-aware scheduling.* This
note is the prior-art audit under that frame.

Companion field-positioning (the isolation axis):
[`SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md)
— why fak bets on a live trunk + disjoint-lease admission instead of a
branch/worktree/VM per agent. The cross-machine atomicity caveats there
([#825](https://github.com/anthony-chaudhary/fak/issues/825)/[#826](https://github.com/anthony-chaudhary/fak/issues/826))
are the same boundary this note draws around `internal/leaseref`.

---

## 1. The five decisions, kept distinct

The issue's load-bearing ask: *distinguish heartbeat, lease, fence, schedule, and
autoscale decisions.* They are routinely conflated ("the worker is alive so it may
keep its lock and run"), and that conflation is exactly the bug class — a stale
holder that is still heartbeating can still corrupt a write. Each row below is a
separate question with a separate answer, a separate SOTA primitive, and a separate
fak seam.

| Decision | The question it answers | SOTA primitive | Current fak seam | Status |
|---|---|---|---|---|
| **Heartbeat** | Is the holder still *alive*? | K8s Lease `renewTime`; Temporal `RecordActivityHeartbeat` | `taskmgr` in-memory beat counter + 30s liveness window (`taskmgr.go:25`, `:106-109`) | process-local, self-reported |
| **Lease** | Who is *admitted* to this resource right now? | K8s Lease `holderIdentity`; Argo semaphore/mutex | `leaseref.Record` over `refs/fak/locks/<id>` (`leaseref.go:72-79`); `gpulease` flock; DOS lane arbitration (`dos.toml`) | cross-machine *visibility*, not atomic acquisition |
| **Fence** | May an *old* holder still write after a new one is admitted? | K8s Lease `leaseTransitions` + a fencing token; ZooKeeper/Chubby monotonic epoch | **none** — no generation/fencing token on any lease record | **gap** |
| **Schedule** | What happens to a run that is due while another is in flight? | Argo `synchronization` (mutex = 1, semaphore = N); cron concurrency policy (Allow/Forbid/Replace) | `loopmgr` governor cadence floor + anti-storm (`governor.go:38-58`) | cadence/back-pressure only; no coalesce/queue/replace policy |
| **Autoscale** | How is freed capacity reallocated *fairly*? | Ray Serve autoscaling (min/max replicas, target queue depth) | speculation governor (`speculation.go`) + dead-backend lane-budget reallocation (`511a8d5`, `9eae36f`) | EV-gated, default-deny effects; pool bounds partial |

The rest of the note expands each column.

---

## 2. SOTA prior art readout

### 2.1 Kubernetes Lease API + `leaderelection`

- **Lease object** (`coordination.k8s.io/v1`): `holderIdentity`, `leaseDurationSeconds`,
  `acquireTime`, `renewTime`, `leaseTransitions`. A holder renews `renewTime` on a
  cadence; a peer treats the lease as expired once `now - renewTime >
  leaseDurationSeconds` and may acquire, bumping `leaseTransitions`.
  <https://kubernetes.io/docs/concepts/architecture/leases/>
- **`client-go/tools/leaderelection`**: the renew loop, the `RenewDeadline` /
  `LeaseDuration` / `RetryPeriod` triad, and the explicit contract that *a holder
  that misses its renew deadline must stop acting as leader* — liveness and
  authority are coupled by a clock the holder does not own.
  <https://pkg.go.dev/k8s.io/client-go/tools/leaderelection>
- **What fak should learn**: the renew/expire/transition triad maps almost
  one-to-one onto `leaseref.Record` — except fak has no `renewTime` (only
  `AcquiredAt` + `TTLSeconds`, `leaseref.go:76-77`) and no `leaseTransitions`
  counter. The transition counter is the seed of a fencing token (see §3.3).

### 2.2 Temporal task queues, workers, heartbeats

- Workers poll task queues; long activities call `RecordActivityHeartbeat`, and the
  server cancels an activity whose heartbeat exceeds `heartbeatTimeout`. Heartbeats
  carry **progress payloads**, so a resumed activity restarts from last-reported
  progress, not from zero. <https://docs.temporal.io/workers>
- **What fak should learn**: Temporal's heartbeat is *witnessed by the server*, not
  self-asserted to the local process — the antidote to "a PID is alive" that the
  issue calls out (#750). fak's witness rung (`taskmgr/evidence.go`) is the right
  primitive for this, but it grades *completion*, not *liveness*; nothing yet emits
  a witnessed *heartbeat* row (see §3.2).

### 2.3 Argo Workflows synchronization

- `synchronization` blocks gate a step on a named **mutex** (a semaphore of 1) or a
  **semaphore** with a configured count, scoped to the workflow or to a namespace
  ConfigMap key. A step that cannot acquire waits; it does not run a second copy.
  <https://argo-workflows.readthedocs.io/en/latest/synchronization/>
- **What fak should learn**: this is the *named-resource overlap policy* fak's
  governor does not have. The governor refuses a too-soon fire (`CADENCE_FLOOR`,
  `governor.go:67`) but has no notion of "this loop holds mutex M; a second fire
  queues / coalesces / replaces." Argo's mutex-vs-semaphore is the vocabulary for
  the missing overlap policy.

### 2.4 Ray Serve autoscaling and request routing

- Per-deployment autoscaling on `min_replicas` / `max_replicas` /
  `target_ongoing_requests`, with upscale/downscale delays so a transient spike does
  not thrash the pool. <https://docs.ray.io/en/latest/serve/autoscaling-guide.html>
- **What fak should learn**: bounded pools + cooldowns + a target-load signal is the
  shape dead-backend reallocation (`511a8d5`, `9eae36f`) needs to stay fair. fak's
  speculation governor already encodes the *economic* half (positive expected value,
  slack-only, default-deny effects, `speculation.go:76-109`); the *pool-bounds +
  cooldown* half is the Ray lesson to import.

---

## 3. fak's contract today (grounded), and the gaps

### 3.1 The lease object

`leaseref.Record` (`internal/leaseref/leaseref.go:72-79`) is the cross-machine lease:

```go
type Record struct {
    ID          string   // ref basename under refs/fak/locks/
    TreeGlobs   []string // repo-relative trees this lease covers
    Holder      string   // machine/session identity, free-form
    AcquiredAt  int64    // unix seconds at acquisition
    TTLSeconds  int64    // lifetime; 0 = no expiry
    Description string
}
```

It rides ordinary `git fetch`/`git push` as a ref under `refs/fak/locks/<id>`, so a
peer on another machine can *see* a held lease. `Record.Expired(now)`
(`leaseref.go:85-90`) makes a crashed holder's lease reapable rather than a
permanent deadlock. Two narrower leases also exist: `gpulease` (a single-host
advisory flock serializing GPU-heavy model loads) and the DOS lane arbiter
(`dos arbitrate`, lane trees in `dos.toml`) that refuses two workers onto
intersecting file trees.

**The honest boundary, kept verbatim from the package doc (`leaseref.go:15-34`)**:
this is *distribution / visibility, not atomic acquisition*. Two machines can still
both write a lease for overlapping trees in the same fetch window; git's merge
converges the *set* of refs, it does not arbitrate a winner. The win is that an
arbiter can now *see* the conflict — final cross-machine race arbitration is out of
scope. This is the same caveat #825/#826 raise.

**Mapped to the K8s field set the issue asks for**:

| Issue-requested field | fak today | Source |
|---|---|---|
| owner / holder | `Record.Holder` (free-form) | `leaseref.go:75` |
| acquired-at | `Record.AcquiredAt` | `leaseref.go:76` |
| expires-at | derived: `AcquiredAt + TTLSeconds` | `leaseref.go:85-90` |
| target resource | `Record.TreeGlobs` | `leaseref.go:74` |
| renewed-at | **absent** (no renew; only acquire + TTL) | — |
| generation / fencing token | **absent** | — |
| proof source | the git ref is content-addressed; a *signed* identity envelope is **deferred** | `leaseref.go:29-34` |

### 3.2 The liveness contract

Two rungs, deliberately separate:

1. **Heartbeat = liveness (self-reported).** `taskmgr` counts beats and classifies a
   task `idle` / `live` / `stalled` against a 30s default window
   (`taskmgr.go:25`, `:29-34`, `:106-109`; `Beat()` at `:311`/`:321`). This is a
   single-observer recency signal — exactly the "a PID/self-report is not enough"
   case #750 names.
2. **Witness = completion (independently observed).** `taskmgr/evidence.go` adds a
   `VerifiedState` rung (`verified_done` / `verified_refused` / `verified_unavailable`,
   `evidence.go:13-28`) that a `Witness` raises only by reading the effect back from a
   source the process did not author (`evidence.go:61-68`). The loop ledger mirrors
   it with `StatusWitnessedDone` / `StatusWitnessRefused` / `StatusWitnessUnavailable`
   (`loopmgr.go:96-98`), and the governor will *hold* a loop whose witnessed/claimed
   ratio collapses (`WITNESS_COLLAPSE`, `governor.go:57`, `:110-119`).

**Gap (#750).** The witness rung grades *completion*, not *liveness*. The loop ledger
*defines* a `heartbeat` event kind (`loopmgr.go:32`, accepted by the validator at
`:550`) but **no in-tree producer emits one** — loop liveness is currently inferred
from `fire`/`start`/`end` cadence plus the taskmgr in-memory beat counter, neither of
which is a persisted, *witnessed* heartbeat with an explicit grace window. The
issue's "two independent observations or an execution witness" contract is satisfied
for *done*, not yet for *alive*.

### 3.3 The fence (the named gap)

There is **no fencing token** on any fak lease. `leaseref.Record` carries no
generation/epoch and no `leaseTransitions`-style counter (`leaseref.go:72-79`), and
the package explicitly defers a signature envelope (`leaseref.go:29-34`). So the
classic failure stands open: holder A's lease expires, peer B reaps and acquires,
then A — still running, still "alive" by its own heartbeat — performs a write that
*should* be rejected because its lease generation is stale. The enforced
call-boundary (`fak`'s refusal at the tool call) is where a fencing check belongs:
admit a write only if the holder's presented generation equals the current lease
generation. **This is the highest-value child this readout produces** (it is what
the issue's "fenced stale-holder" acceptance test would exercise).

### 3.4 The schedule / overlap policy

The governor (`internal/loopmgr/governor.go`) is pure admission back-pressure over a
folded ledger snapshot — no I/O, no scheduling, first-failing-gate-wins
(`governor.go:84-124`). Its closed reason vocabulary (`governor.go:64-71`):
`LOOP_PAUSED`, `LOOP_DISABLED`, `CADENCE_FLOOR`, `REFUSAL_STORM`,
`WITNESS_COLLAPSE`, `POLICY_ADMITTED`. The cadence floor (`MinIntervalSeconds`)
*does* stop two overlapping schedulers from storming one loop faster than intended
(`governor.go:38-42`).

**Gap (Argo's lesson).** What it does *not* express is the overlap policy for a run
that is due *while a prior run is still in flight*: skip / queue / coalesce / replace
under a named mutex or semaphore. Today the only answer is "refuse if too soon";
there is no `run started because the mutex was free` vs `run skipped because the
mutex was held` distinction in the ledger.

### 3.5 Autoscale / fairness

The speculation governor (`internal/loopmgr/speculation.go`) gates best-effort
speculative work with a fixed-order, pure decision: correctness first (default-deny
effects unless proven read-only **and** idempotent **and** non-destructive,
`speculation.go:127-138`), then capacity (slack-only, `:84-87`), then economics
(positive expected value, `:89-96`). Its reasons: `SPEC_ADMITTED`,
`SPEC_EV_NEGATIVE`, `SPEC_NO_SLACK`, `SPEC_EFFECTFUL_REFUSED` (`speculation.go:39-43`).
Dead-backend lane-budget reallocation (`511a8d5`, `9eae36f`) frees a dead holder's
budget to a healthy peer.

**Gap (Ray's lesson).** The economic half is shipped; the *pool-bounds + cooldown +
replayable reason trace* half — min/max bounds on reallocation so a flapping backend
cannot thrash the pool — is the Ray Serve pattern to import, and every reallocation
should leave a reason a downstream can route on (the governor already models this
shape for loops).

---

## 4. Answers to the issue's five questions

1. **What is fak's lease object?** `leaseref.Record` (cross-machine, git-ref-backed,
   `leaseref.go:72-79`) plus `gpulease` (single-host flock) and DOS lane arbitration.
   It carries owner, acquired-at, expiry (via TTL), and target resource; it is
   **missing** renewed-at and a generation/fencing token, and its proof source is
   git content-addressing, not yet a signed identity envelope (§3.1, §3.3).
2. **What is the liveness contract?** Two rungs: a self-reported heartbeat
   (`taskmgr`, 30s window) for *alive*, and an independently-witnessed `VerifiedState`
   for *done* (`evidence.go`). Witnessed completion is shipped; a witnessed
   *heartbeat* with an explicit grace window is the open #750 child (§3.2).
3. **What is the overlap policy?** A cadence floor + anti-storm back-off
   (`governor.go`). A missed/overlapping run is currently only *refused if too soon*;
   skip/queue/coalesce/replace under a named mutex/semaphore (Argo's model) is not
   yet expressed (§3.4).
4. **What is the fencing rule?** There is none today. The intended rule — *a stale
   lease generation must not let an old holder commit writes after a new holder is
   admitted* — belongs at fak's enforced call boundary and needs a generation token
   added to the lease record first (§3.3).
5. **How does autoscaling interact with fairness?** Speculation and dead-backend
   reallocation are EV-gated and default-deny on effects (`speculation.go`); the
   missing piece is pool bounds + cooldowns + a replayable reason trace so
   reallocation cannot thrash (§3.5).

---

## 5. Implementation children (the remaining #906 acceptance items)

This readout satisfies the first acceptance box (*the SOTA readout exists and
distinguishes the five decisions*). The other three boxes are code/process children,
named here so the sibling agent-OS issues can link to this contract instead of
re-defining the semantics inline:

- **C1 — Fencing token (highest value).** Add a monotonic `Generation` (and a
  `RenewedAt`) to `leaseref.Record`; bump it on every acquire/transition; have the
  call-boundary admit a write only when the presented generation matches the current
  lease. Unblocks the issue's *fenced stale-holder* acceptance test
  (`go test ./internal/leaseref`).
- **C2 — Overlap-lock test.** Two workers racing for one named job: prove the
  mutating section runs once, and that a stale holder cannot write after the fence
  advances (`internal/loopmgr` or `internal/leaseref`). Acceptance box 2.
- **C3 — Ledger explainability.** Make `fak ps` / the loop ledger emit and surface
  *why* a run started, skipped, waited, or was fenced — a `schedule` decision row
  distinct from the existing admission reasons. Acceptance box 3. (The governor's
  closed reason vocabulary, `governor.go:64-71`, is the seam to extend with a
  fence/skip/wait reason.)
- **C4 — Witnessed heartbeat (#750).** Emit the already-defined `EventHeartbeat`
  (`loopmgr.go:32`) as a persisted row with an explicit grace window, and require two
  independent observations (or an execution witness) for liveness, not the worker's
  own string.
- **C5 — Link the family.** Update
  [#748](https://github.com/anthony-chaudhary/fak/issues/748)/[#749](https://github.com/anthony-chaudhary/fak/issues/749)/[#750](https://github.com/anthony-chaudhary/fak/issues/750)/[#763](https://github.com/anthony-chaudhary/fak/issues/763)/[#764](https://github.com/anthony-chaudhary/fak/issues/764)/[#765](https://github.com/anthony-chaudhary/fak/issues/765)/[#825](https://github.com/anthony-chaudhary/fak/issues/825)
  to this contract. Acceptance box 4.

---

## 6. Honest boundary (what this note does NOT do)

- It does **not** ship a fencing token, a heartbeat row, or an overlap policy — those
  are the children in §5. The claim here is the *readout + contract*, nothing more.
- It does **not** re-measure or re-benchmark; every fak citation is a file:line in
  the tree at this commit, every SOTA citation is a primary doc URL.
- It does **not** claim parity with any of the four systems. fak's contribution is
  the *enforced call-boundary* under these primitives (a refusal at the tool call),
  not a re-implementation of Kubernetes/Temporal/Argo/Ray.
