---
title: "Region admission: how loops, super loops, dispatch, and manual sessions coordinate"
description: "One admission decision (internal/regionadmit) over one lease fabric (internal/leaseref) — the dispatch tick, fak loop drive, and fak loop region all ask the same question before mutating a file tree, and refuse with the same closed COLLISION_RISK vocabulary."
---

# Region admission (`internal/regionadmit`)

> The fleet runs many execution surfaces at once: dispatch workers, `fak loop
> drive` loops, super-loop walks that tell an operator what to enter next, RSI
> candidates, and plain manual sessions. Before this seam, only the **dispatch
> tick** checked anybody else before acting — and only with raw tree geometry.
> A loop and a dispatch worker could edit the same tree with **no mutual
> visibility** at all. Region admission is the one shared answer to *"may THIS
> actor act on THIS (lane, tree) right now?"*

## The seam — one decision, one fabric, one vocabulary

Three pieces, deliberately separated:

- **The fabric** — [`internal/leaseref`](../internal/leaseref/leaseref.go)
  persists a lease record (tree globs, holder, TTL, fencing generation) under
  `refs/fak/locks/<id>`, so lease state rides ordinary `git fetch`/`git push`
  between clones. This already existed; dispatch was its only writer.
- **The decision** — [`internal/regionadmit`](../internal/regionadmit/regionadmit.go)
  is pure (state in, verdict out): the in-binary twin of the `dos arbitrate`
  admission contract, fed by whatever live-lease projection the caller holds.
  Its rules, in order:
  1. an **exclusive lane** request (`dos.toml [lanes].exclusive` — `abi`,
     `release`, `dos`, `global`) admits only when nothing else is live;
  2. a live lease **on** an exclusive lane refuses every new region;
  3. a request naming the **same lane** a live lease holds is refused — a
     named lane serializes even on disjoint trees (the lease's lane is
     inferred by matching its tree against the `dos.toml [lanes.trees]`
     taxonomy, so no lease-record schema change was needed);
  4. a requested tree **overlapping** a live lease's tree is refused —
     `dispatchorder.TreesOverlap`, the same prefix geometry the dispatch
     fan-out price uses (one algebra, never re-derived). An empty tree is
     unknown blast radius and collides conservatively.
- **The vocabulary** — every refusal is `COLLISION_RISK` (the same
  `dos.toml [reasons]` token the dos arbiter and the dispatch order speak),
  carrying the *rung* that fired (`tree_overlap`, `same_lane_live`,
  `exclusive_lane_live`, `exclusive_lane_requested`) and the conflicting
  lease (id + holder) as evidence — never free prose.

## Who consults it (the coordination table)

| Surface | Before | Now |
|---|---|---|
| **Dispatch tick** (`fak dispatch tick`) | inline geometric overlap scan, no lane semantics | same acquire path, but the decision is `regionadmit.Decide` — gains lane serialization + exclusive-lane refusal; refusals carry the rung |
| **`fak loop drive`** | nothing — two loops, or a loop and a dispatch worker, could edit one tree blind | a GOAL.md `lane:` / `region:` (or `--lane` / `--tree`) makes the drive refuse over a live overlapping lease, then **hold a fenced lease** on its region for the whole drive (renewed each turn, released on exit, honest-stop on a mid-drive `STALE_LEASE` takeover) |
| **Manual session / script** | nothing to consult | `fak loop region --lane <l> [--tree <g>] --actor session:<id>` — the same decision as a standalone verb (exit 0 admit / 3 refuse); hold with `fak leaseref acquire` if admitted |
| **Super loop** (`fak superloop walk`) | worklist only; two operators could enter the same member | the walk stays read-only and gains nothing automatically **yet**: today an operator entering a member can run `fak loop region` first by hand, and a member that happens to be a lane/region-declaring GOAL loop inherits the hold; the drive rung that enters members *through* this gate is the named follow-on (#2224) |
| **RSI loop** | physical isolation (private worktree) | unchanged — isolation by construction needs no lease |

Because every surface writes into the **same** `refs/fak/locks/*` namespace,
visibility is symmetric: a loop's held region refuses a dispatch spawn, and a
dispatch worker's lane lease refuses a loop drive — witnessed end-to-end in
[`cmd/fak/loop_drive_region_test.go`](../cmd/fak/loop_drive_region_test.go).

## Using it from a GOAL.md loop

```markdown
---
loop: gateway-nightly
witness: commit-audit
lane: gateway                 # the dos.toml lane; its canonical tree is the region
# or explicit globs:
# region: internal/gateway/**, docs/gateway.md
budget: { max_iters: 8 }
---
```

The drive then emits the region lease as ledger evidence
(`region_lease: loop-gateway-nightly` on every turn event), records a
`COLLISION_RISK` refused-admit event when it must yield, and exits `3` so a
scheduler treats it like any other structured refusal. No lane, no region — no
change: the historical uncoordinated drive is preserved byte-for-byte.

## Using it from a manual session

```bash
# before editing internal/gateway/** in a shared checkout:
git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'   # see peers' leases
fak loop region --lane gateway --actor session:$ME     # may I?
fak leaseref acquire --id session-$ME-gateway --tree 'internal/gateway/**' --ttl 3600
# ... work ... (renew with `fak leaseref renew` if it runs long)
```

Once held, the manual lease is not advisory decoration: the dispatch tick and
every lane/region-declaring loop drive will **refuse** to enter that region
until it clears. When the work is done, the TTL bounds the record and
`fak leaseref reap` (or the garden tick) removes it once expired; a CLI
release twin of `acquire` is a named follow-on.

## Operational consequences worth knowing

- **An exclusive-lane lease stalls the fleet on purpose.** A live lease whose
  tree sits inside an exclusive lane (`abi`, `release`, `dos`, `global`)
  refuses *every* new region — that is the dos "runs alone" contract, now
  enforced at dispatch and loop-drive admission. The flip side: a leaked
  600-second release-tree lease blocks all spawns until it expires or
  `fak leaseref reap` / the garden tick clears it. Keep exclusive work's TTLs
  short and reap promptly.
- **A narrowed lease keeps its lane.** A lease on a sub-region
  (`region: internal/gateway/http/**`, or dispatch `--lease-tree`) is
  classified back to its lane by containment, so same-lane serialization and
  exclusive-lane blocking still apply to it; a tree spanning several lanes
  owns no lane and is protected by geometry alone.
- **The loop-drive lease renews per turn (TTL 3600s by default).** A single
  agent turn longer than the TTL lapses the lease mid-turn; if nobody took the
  region meanwhile the next turn boundary reacquires it silently, and if a
  peer did, the drive honest-stops with the fence's verdict. Pass `--deadline`
  to size the TTL to the whole drive when turns may run long.

## Honest boundary

The same one `internal/leaseref` declares, unchanged by this seam:
cross-machine this is **distribution / visibility, not atomic acquisition** —
after a `fak leaseref sync` a peer's lease is seen, but a same-fetch-window
race between two clones is not arbitrated. Same-host, the fence
(`AcquireFenced`'s CAS + generation) is real atomicity. The decision itself is
only as complete as the lease set it is shown: a surface that acquires nothing
(RSI by design; any legacy uncoordinated launch) is still invisible. Named
next rungs, tracked in the backlog: the super-loop drive rung entering members
through this gate (#2224), preflight live-count reading these leases (#2226),
and the relay baton's `held_region` re-acquire on resume (#1860 track H).
