---
title: "Random time horizons of agent usage: dormancy as a measured quantity and horizon-gated rehydration"
description: "Research/SOTA readout and design map for first-class support of agents that restore after arbitrary dormancy (minutes to months) and cycle on/off on a cadence. Surveys durable-execution timers, Durable-Object hibernation, snapshot/restore (CRaC), lease fencing, and long-horizon agent memory; maps them against fak's CURRENT half-built time primitives (internal/resume idle-vs-TTL projection, session WaitResume/WarmKVSplicer, leaseref TTL, bgloop fixed interval) with file citations; and names the net-new layer: a dormancy clock, a horizon-gated rehydration protocol that scales revalidation rigor to the gap, and durable long-horizon wake/duty-cycle. Planning/research only — the implementation children are the epic issues this note backs."
date: 2026-06-28
---

# Random time horizons of agent usage

Date: 2026-06-28. Status: research + design contract (planning only, no code).

This note backs a GitHub epic. It is a readout and a contract, not new code. Every
fak citation is a path/symbol in the tree at this commit; every SOTA citation is a
primary doc URL.

Tracked in epic [#1178](https://github.com/anthony-chaudhary/fak/issues/1178) with ten
children: dormancy clock (#1179) and observability (#1180); the rehydration orchestrator
(#1181) and its rungs — lease fence (#1182), credential revalidation (#1183), recall
revalidation (#1184), cold-cache handling (#1186); durable wake/duty-cycle (#1188),
wake-on-event + herd safety (#1191), and the time-travel test harness (#1192).

## 1. The problem the existing notes do not name

fak has built a strong **loop kernel** and **session-restore** substrate:

- `internal/loopmgr` — durable hash-chained loop-event ledger (`armed/fire/admit/
  start/heartbeat/end/witness/notify`) + a governor with a cadence floor and a closed
  refusal vocabulary (`governor.go`).
- `internal/bgloop` — an in-kernel `Supervisor` that runs registered loops in their
  own goroutines on the `fak serve` lifecycle, panic-contained, observable at
  `GET /v1/fak/loops` and `fak_bgloop_*`.
- `internal/sessionimage` — `DumpDir`/`LoadDir`/`Rehydrate`, with a *sound-restore*
  property: a restored session consults an independently-read witness rung before
  re-firing an irreversible effect (the ACRFence replay defense,
  [`ACRFENCE-SOUND-RESTORE-2026-06-25.md`](ACRFENCE-SOUND-RESTORE-2026-06-25.md)).
- `internal/leaseref` — cross-machine git-ref leases with `AcquiredAt + TTLSeconds`;
  the scheduling SOTA readout
  [`ORCHESTRATION-LEASE-HEARTBEAT-SCHEDULE-SOTA-2026-06-26.md`](ORCHESTRATION-LEASE-HEARTBEAT-SCHEDULE-SOTA-2026-06-26.md)
  (issue [#906](https://github.com/anthony-chaudhary/fak/issues/906)) already names
  the fencing-token and witnessed-heartbeat gaps.
- `internal/resume` — a cold-resume **cost projection** that compares idle time
  against the provider cache TTL (5m / 1h) and recommends `CUT` vs `RESUME_FULL`.
- The strategy frame
  [`LONG-RUNNING-AGENT-LOOPS-2026-06-25.md`](LONG-RUNNING-AGENT-LOOPS-2026-06-25.md):
  *schedulers are interrupt sources; fak owns loop identity, admission, leases,
  witness, resume state.* Its "green thread" model — keep thousands of logical loops
  as cheap records, bind a worker only while a tick executes — is the on/off substrate.

**What none of them treats as first-class is the _length of the gap_.** Every one of
these organs runs the same path whether the agent was off for 5 minutes or 5 months:

- `internal/resume`'s idle-vs-TTL projection ships **shadow-only** — it is computed,
  never wired to drive re-entry.
- `internal/session/resume.go` defines `WaitResume` + a `WarmKVSplicer`, but the KV
  reattach is **unwired**; warm re-entry is advisory-only. `sessionimage` sets
  `KVIncluded=false` by design, so resume is **always a cold re-prefill** regardless of
  how long the session slept.
- `internal/leaseref.Record` has a TTL but **no generation/fencing token** (#906 §3.3),
  so a holder that goes dormant past its TTL and returns is the worst-case stale writer
  — and nothing keys on the fact that it was dormant.
- `internal/accounts` tracks identity but **no token expiry**; a long-dormant seat's
  OAuth token is discovered stale only on a 401 (the ~13-minute headless gap operators
  hit; [`ACCOUNT-LIFECYCLE-RUNBOOK-2026-06-26.md`](ACCOUNT-LIFECYCLE-RUNBOOK-2026-06-26.md)).
- `internal/bgloop` loops tick on a **fixed `Interval`** set at registration — there is
  no "sleep until T" (T possibly weeks away), no declarative duty cycle, no wake-on-event.
- `internal/cachemeta` has a `COLD_TTL` break reason but it is projection-only, not a
  live time-gated eviction.

So fak has the *organs* of dormancy handling, half-built and disconnected, with **no
shared notion of how long an agent/session/lease has been dormant, and no protocol that
scales what it revalidates on wake to the size of that gap.**

## 2. Three usage shapes to support

1. **Cold restore after arbitrary dormancy** — an agent comes back after hours, days,
   weeks, or months. Over that horizon, progressively more of its cached world is stale:
   prompt cache (minutes), credentials (hours), recalled memory and plan (days), lease
   ownership (re-granted), repo HEAD (thousands of commits), model availability (renamed/
   deprecated). The longer the gap, the more must be re-verified before the first action.
2. **Regular-cadence on/off cycling** — a loop that is "on weekdays 9–5, off otherwise"
   or "wake every 6h." A duty cycle, not a rate limit. Held as a cheap record while off,
   bound to a worker only while on.
3. **Wake-on-event after hibernation** — a dormant loop reconstituted by an inbound
   signal (a chat fire, a webhook, a GitHub event) rather than a live process, then
   re-entered through the same rehydration gate.

## 3. SOTA readout (cited)

### 3.1 Durable execution — durable timers and signals
Temporal, Restate, and DBOS persist **progress, not the transcript**, and survive long
sleeps with **durable timers**: DBOS `sleep` stores its wake time in Postgres and, if the
process dies mid-sleep, looks the wake time back up and keeps sleeping toward it — "sleep
one month" works across restarts. **Durable messaging** waits with a long timeout for a
notification delivered via a Postgres trigger + `NOTIFY`. Recovery is transparent:
completed steps replay their journaled results, the rest re-execute.
- <https://docs.dbos.dev/> · <https://www.restate.dev/what-is-durable-execution> ·
  <https://docs.temporal.io/workers>

**Lesson for fak:** the durable-wake primitive (§5.3) is a `loopmgr` row + a re-arm on
`fak serve` startup, not a held process. The "persist progress, not transcript" maxim is
already fak's witness discipline.

### 3.2 Cloudflare Durable Objects — hibernation + alarms
The DO lifecycle is **idle → hibernate → evict**. In-memory state is a cache; **durable
state survives, in-memory does not**. An incoming event reconstructs the object (its
constructor re-runs); the **Alarms API** schedules an explicit future wake. `setWebSocket
AutoResponse` answers routine pings *without waking* the object. Project Think (preview)
adds durable fibers with an `onFiberRecovered` hook for resuming after eviction.
- <https://developers.cloudflare.com/durable-objects/concepts/durable-object-lifecycle/> ·
  <https://developers.cloudflare.com/durable-objects/best-practices/websockets/>

**Lesson for fak:** this is the exact shape of the green-thread model in
`LONG-RUNNING-AGENT-LOOPS`. "Reconstruct from durable storage on the next event" = the
rehydration gate (§5.2). The auto-response-without-waking idea maps to keeping a dormant
loop a cheap record that a cheap signal can probe without a full re-entry.

### 3.3 Snapshot/restore — Firecracker, Lambda SnapStart, CRIU/CRaC
Snapshot a fully-initialized runtime and restore it in milliseconds. The load-bearing
detail for dormancy: **CRaC's `afterRestore` hook** is where the app **reconnects DB
pools and refreshes secrets** captured stale in the frozen snapshot — the restore is not
trusted to be still-valid, it is *re-validated at the checkpoint boundary*.
- <https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html> ·
  <https://openjdk.org/projects/crac/>

**Lesson for fak:** `sessionimage.Rehydrate` is fak's restore. It already re-reads a
witness rung; the missing piece is a CRaC-style `afterRestore` that, **gated by image
age**, refreshes the credential, lease, recall, and cache state that a long freeze
invalidates (§5.2).

### 3.4 Lease TTL + fencing tokens — etcd, ZooKeeper, Chubby
The paused-then-resumed holder is the canonical hazard: a holder that pauses past its TTL
has lost the lease but does not know it; when it wakes it may corrupt shared state. A TTL
alone never fixes this — the structural defense is a **monotonic fencing token** (etcd
revision, ZK `zxid`, Chubby sequencer) that the *protected resource* validates, rejecting
any write carrying a token older than the highest seen. Use **monotonic clocks**; treat
lease loss as **halt-and-recover**, not a silent state change.
- <https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html> ·
  <https://etcd.io/docs/v3.5/learning/why/>

**Lesson for fak:** a long-dormant returner is the worst-case stale writer. #906's
fencing token (child C1) is the mechanism; this epic supplies the **dormancy-driven
trigger** — on a cold/frozen/ancient re-entry, re-check lease generation and
halt-and-reacquire *before* any write (§5.2).

### 3.5 Long-horizon agent memory — staleness and consolidation
After many sessions, agent memory accretes **contradictory entries, stale notes
referencing deleted files, and relative timestamps ("yesterday") that lose meaning**.
The fixes: re-validate facts at read time, normalize temporal references to absolute,
and consolidate on idle ("sleep-time compute") rather than on the hot path.
- <https://www.letta.com/blog/sleep-time-compute/> ·
  <https://arxiv.org/abs/2304.03442> (generative-agents reflection)

**Lesson for fak:** fak already has the read-time re-validation primitive — the DOS
`dos_recall` verdict (`RECALL_FRESH / RECALL_STALE / RECALL_UNVERIFIABLE`) re-checks a
memory's named artifacts against git/worktree NOW. The gap is **wiring it into the wake
path** so a memory recalled after a long gap is re-verified before it is trusted, and
normalizing the temporal drift the memory index already warns about.

## 4. Concept: dormancy as a measured quantity

Define one durable, monotonic-clock-based **`LastActiveAt`** per loop / session / lease,
and a pure **horizon bucketer** that turns a gap into a band whose thresholds map to what
actually decays at that scale:

| Horizon | Gap | What has decayed | Rehydration rigor |
|---|---|---|---|
| **warm** | < ~5 min | nothing (prompt cache still warm) | none — resume verbatim |
| **cool** | < ~1 h | prompt cache cold (TTL), KV gone | re-warm plan; cache marked cold |
| **cold** | < ~24 h | + creds may be expired; recall aging | + cred refresh, recall re-validate |
| **frozen** | < ~30 d | + lease re-granted; plan stale; HEAD moved | + lease fence-check/reacquire, plan freshness |
| **ancient** | ≥ ~30 d | + model renamed/deprecated; world changed | + full re-validation, treat as fresh start |

The thresholds are not arbitrary: ~5 min and ~1 h are the provider cache TTLs
`internal/resume` already encodes; ~24 h and ~30 d are credential and "world materially
changed" scales. The bucket is the single input every rehydration rung keys on — longer
gap ⇒ strictly more rungs run before the first post-wake action is admitted.

## 5. The net-new layer (epic children)

### 5.1 Measure (Phase 1)
- **Dormancy clock + horizon bucketer** — a durable `LastActiveAt` on session/loop/lease
  and a pure `Horizon(gap)` (stdlib, no I/O). Consumes `loopmgr` fire/end timestamps,
  `leaseref.AcquiredAt`, and the idle figure `internal/resume` already computes; promotes
  it from shadow to a first-class field. (An enabling cleanup: `session.RunState` (uint8)
  and `loopmgr.LoopState` (string) duplicate pause/drain/stop — a shared lifecycle type
  makes the clock uniform.)
- **Dormancy observability** — emit horizon into the ledger + a `fak dormancy` view +
  `fak_dormancy_*` metrics (distribution of gap lengths, restores by bucket). Crucially,
  **distinguish "intentionally dormant" from "stuck"** — today `loop_stuck_age_seconds`
  cannot tell a planned sleep from a hang.

### 5.2 Rehydrate (Phase 2) — the cold-restore-after-long-off half
- **Rehydration orchestrator** — a staged re-entry gate keyed by horizon bucket (the CRaC
  `afterRestore` analog). Each rung is independently witnessed and refuses with a closed
  reason (`STALE_CRED / STALE_LEASE / STALE_RECALL / COLD_CACHE / STALE_PLAN`). It
  *composes* the rungs below; it does not re-implement them.
- **Lease fence on long-dormancy re-entry** — consume #906-C1's fencing generation: a
  cold+ returner re-checks lease generation and **halts-and-reacquires** before any write.
  This issue is the dormancy *trigger* + halt policy, not the token itself.
- **Credential revalidation by elapsed dormancy** — fix the account-switcher expiry
  blind spot: a cold/frozen/ancient return forces a token freshness check + refresh
  *before* the first upstream request, instead of a reactive 401.
- **Memory/recall revalidation rung** — wire `dos_recall` into the wake path; re-verify
  any recall page admitted after a cold+ gap against git/worktree; normalize temporal
  drift ("yesterday" → absolute).
- **Cold prompt-cache handling** — wire `internal/resume`'s idle-vs-TTL projection (today
  shadow-only) so a wake past TTL marks the cache COLD (surface `cachemeta` `COLD_TTL`),
  the planner stops assuming warm-cache latency/price, and plans a re-warm; this is also
  where a real warm-KV restore tier (today `KVIncluded=false`) would attach.

### 5.3 Wake & cycle (Phase 3) — the on/off-cadence half
- **Durable wake timers + declarative duty cycle** — a loop record can declare "sleep
  until absolute T" or "duty cycle: on `<spec>`, off `<spec>`" with horizon-aware jitter;
  the wake survives process death and is **re-armed on `fak serve`/bgloop startup** (the
  DBOS durable-sleep / DO Alarms analog). Distinct from the cadence *floor*, which only
  rate-limits a fire that already arrived.
- **Wake-on-event + thundering-herd safety** — a dormant loop held as a cheap record is
  reconstituted by an inbound signal (DO hibernation analog); a synchronized fleet wake
  (shared cadence, or mass restore after an outage) is staggered/jittered through
  admission so it does not storm (the resume-storm hazard).
- **Time-travel test harness** — a deterministic dormancy simulator with an injectable
  clock that fast-forwards hours → months to prove every rung, the fence-on-return, the
  durable wake, and the duty cycle behave at each horizon **without real waits**. This is
  the acceptance substrate for all of the above (and avoids the `Date.now` non-determinism
  the repo already forbids in scripts).

## 6. Honest boundary

- This note ships **no code**. The children in §5 are GitHub issues under the epic; each
  has its own acceptance test and lane.
- It **reuses, not re-implements**: the fencing token is #906-C1; `dos_recall` already
  exists; `sessionimage.Rehydrate` is the restore; `internal/resume` already has the
  idle-vs-TTL model. The net-new is the **time-horizon dimension threaded across them**.
- It does **not** claim parity with Temporal/DBOS/Durable Objects. fak's contribution is
  the *enforced re-entry boundary* — a refusal at the wake gate when the world a long
  sleep invalidated has not been re-validated — not a re-implementation of a durable
  execution engine.
