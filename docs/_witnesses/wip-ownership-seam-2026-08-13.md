---
title: "Shift-left WIP control at the shared-tree ownership seam"
description: "Date: 2026-08-13. Repo: C:\\work\\fak, branch main. Shipped: internal/wiplease (pure projection leaf + behavior tests)."
---
# Shift-left WIP control at the shared-tree ownership seam

Date: 2026-08-13. Repo: `C:\work\fak`, branch `main`.
Shipped: `internal/wiplease` (pure projection leaf + behavior tests).

## Problem centrality

The shared always-on checkout is the substrate every fleet surface writes through. Two
dozen sessions edit one working tree concurrently. Ownership of a path is therefore not a
bookkeeping detail — it is the precondition for every other guarantee the repo makes about
concurrent work: sweep safety, reconcile correctness, lane admission, and whether a peer
can tell "somebody is working here" from "somebody left this here."

Centrality is **high, and structurally so**: the ownership fact is consumed by
`laneadmit` (admission), `gitgate/sweepguard` (staging safety), the `wip reconcile` /
tree-doctor family (recovery), and `flowmetrics` (the `local_wip` KPI). All four read the
same underlying question. None of them can answer it early.

## P1–P4

- **P1 — The fact exists only retroactively.** `wipattr` attributes a dirty hunk by
  matching it against session *checkpoints*. A checkpoint snapshots a delta that already
  exists, so a session's **first edit is `ORPHAN` by construction** and stays `ORPHAN`
  until its next checkpoint boundary. The window where collision is cheapest to avoid is
  exactly the window where the tree knows least.
- **P2 — The lease plane knows owners without paths.** Of **24 live leases** under
  `refs/fak/locks/*` on 2026-08-13, **21 are `session-*` heartbeat records** carrying only
  `{id, host, pcb_state, updated_at, ttl_seconds}` — liveness with **no `tree_globs`**.
  Exactly **2** declared a footprint (`issue-6603-portability-lifecycle`, 6 explicit
  paths; `resolve-issue-6573-relativity-census`, 1 glob) and **1** claimed an issue
  number, not paths. So the geometry `laneadmit.Decide` reasons over is **empty for ~88%
  of live sessions**.
- **P3 — Adoption, not absence, is the defect.** Path-scoped early ownership is *already
  implemented and already works*: `leaseref` records carry `Tree []string` + TTL + holder,
  and `laneadmit` refuses on glob overlap with `COLLISION_RISK`. Measured coverage is
  **~7 declared paths against ~226 dirty ones (~3%)**. The mechanism is not missing; it is
  unused at the moment of first mutation.
- **P4 — The two halves never meet.** The WIP plane knows *paths without owners*; the
  lease plane knows *owners without paths*. Nothing in the tree projects one into the
  other. Verified: the only constructor of `laneadmit.Lease` outside tests is a bench
  scenario (`internal/conceptbench/scenario_lane.go:163`).

## For / Problem / Today / Better because / Witness

- **For** — a session about to touch a path in the shared checkout, and the peers already
  in it.
- **Problem** — it cannot learn who is editing that path until someone checkpoints, and
  ~88% of live sessions never declare a footprint at all.
- **Today** — ownership surfaces during *cleanup*: sweep-guard warns at `git add`,
  reconcile recovers after the fact, tree-doctor triages the wreck. The cheap moment —
  before the first edit — has no reader.
- **Better because** — a live session's real dirty footprint becomes lease geometry the
  *existing* admission decision already refuses on, with **no new gate and no behavior
  change for anyone who declares nothing**. Abandoned work is split out rather than
  blocked, so adoptable WIP stays adoptable.
- **Witness** — `internal/wiplease/project_test.go`, 11 behavior tests including
  `TestProjectedLeaseIsRefusedByLaneAdmit`, which drives a projected lease through the
  real `laneadmit.Decide` and asserts the refusal names it.

## Audit against existing surfaces (dedupe)

| Surface | Granularity | When it acts | Overlaps this? |
|---|---|---|---|
| `leaseref` (`refs/fak/locks/*`) | path globs + TTL + holder | on explicit acquire | **No** — supplies the shape; unused at first mutation (P3) |
| `laneadmit.Decide` | lane mode + tree globs | at the act boundary | **Decision reused, not duplicated** — this leaf feeds it |
| `lanebeat` | lane lease | heartbeat/refresh | No — expiry of *declared* leases |
| `wipattr` / `sweepguard` | dirty hunk | at `git add` (retrospective) | No — supplies the attribution input |
| `wip reconcile` / tree-doctor | dirty path | during cleanup | No — the surface being shifted left |
| `session-*` records | session | heartbeat | No — liveness only, carries no paths (P2) |
| DOS lane leases | lane | pre-admission | No — lane granularity, no per-path tree |

**Deduped away:** an earlier draft in this session added `wipattr.AdmitStart`, a
start-of-task gate folding attribution + liveness into its own ADMIT/HOLD verdict. The
audit showed it re-implements `laneadmit.Decide`'s glob-overlap decision with a second,
independently-drifting refusal vocabulary. It was **not shipped**; the projection feeding
the existing decision was shipped instead. A second gate over the same geometry is a
second answer that can disagree with the first.

## What shipped

`internal/wiplease.Project(attrs, live, opts) Occupancy` — pure, no git/clock/I/O.

- `Active []laneadmit.Lease` — one lease per **live** owner, `Tree` = its real dirty
  footprint, id namespaced `wip:` so an *observed* lease is never mistaken for an
  *acquired* one (different authority; a refusal must not conflate them).
- `Reclaimable []Reclaimable` — dirty paths with no live owner, reason
  `OWNER_DEAD` or `UNATTRIBUTED`. Deliberately **not** projected as leases: no live holder
  remains to release them, so blocking would wall off adoptable work permanently.
- `Undeclared int` — the adoption gap in one number.

Fail-safe direction is toward *reclaimable*, never toward *blocked*: over-reporting a
lease deadlocks a peer behind a session that no longer exists, and nothing can release it.

## Live evidence captured while auditing

The tree at audit time did not build, in two places, from undeclared concurrent WIP:

```
$ go build ./cmd/fak
internal\armbench\run.go:246:6: Run redeclared in this block
internal\gateway\messages_stream_generation.go:130:7: p.gen undefined
```

Both are peer working-tree edits with **no declared lease**, i.e. exactly the ~88% class
in P2. Neither was touched.

## Concrete next actions

1. Wire `Project` into the surface that reads leases before acting, so projected
   occupancy joins declared leases in `laneadmit.Decide`.
2. Emit `Undeclared` as a `flowmetrics` KPI — the adoption gap is currently unmeasured.
3. Expiry/check-in for projected occupancy: a live owner whose footprint has not changed
   for N minutes is a check-in candidate, distinct from a dead one.

## Reproduce

```
go test ./internal/wiplease/          # 11 behavior tests
go test ./internal/architest/ -run 'Tier|Layer|Import'   # layering gate
git for-each-ref refs/fak/locks/ --format='%(refname)' \
  | while read r; do git cat-file -p "$r"; done          # the 24-lease measurement
```

Note: `go test ./internal/architest/` (full) fails on two **pre-existing** defects
unrelated to this change — SBOM effect-register drift and `ci.yml` referencing deleted
`tools/tool_coverage_audit*.py`. The layering gate itself passes.
