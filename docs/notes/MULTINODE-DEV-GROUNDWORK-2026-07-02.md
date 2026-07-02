# Multi-node concurrent development — groundwork (epic #2254)

**Date:** 2026-07-02 · **Status:** plane 0 shipped, planes 1–2 filed · **Epic:** #2254
(children #2297, #2299, #2300, #2302, #2304)

The fleet already runs many agents against ONE clone on ONE machine (the shared-trunk
discipline: dos.toml lanes, `dos arbitrate`, safecommit's flock, laneadmit). This note
is the groundwork for the next regime: **multiple hardware nodes** — each with its own
clone — developing the same trunk at the same time. The question is not "how do we
lock" (the lease vocabulary exists); it is **how lease state, presence, and intent
TRAVEL between machines**, with what latency, and what happens when a link is down.

## What already exists (the inventory this design builds on)

| Layer | What it gives | Scope today |
|---|---|---|
| `internal/leaseref` (#825) | lease records under `refs/fak/locks/<id>`, riding git fetch/push | cross-machine **visibility**, manual transport |
| `leaseref` fencing (#906-C1) | monotonic generation token; `Fence` refuses a paused-then-resumed stale holder | same-host CAS atomicity |
| session descriptors (#2164) | heartbeated session refs; liveness classifies self / peer-live / peer-dead / peer-unknown, fail-closed reclaim | any clone that fetched |
| intent leases (#2155) | work-target claims (`intent-<key>`), same transport | same |
| `internal/laneadmit` / `regionadmit` | in-binary arbitrate twin: lanes + tree geometry + live leases → COLLISION_RISK | one process, local ref store |
| `internal/safecommit` | commit-time flock + stale-base guard | single host |
| hardware catalog | machine table (`experiments/benchmark/catalog.json`) | static, bench-oriented |
| gateway (`fak guard`) | the fleet's HTTP front door, in-process loopback | no lease surface |

The one structural hole at plane 0 was **transport**: every consumer doc said "run
`git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'` before deciding, push after" —
and nothing in the binary did it. An arbiter on node B was exactly as good as the last
time an operator remembered a refspec.

## The three coordination planes

Degradation ladder, fastest first. Every plane speaks the SAME vocabulary — lease id,
holder, generation, tree globs, TTL — so a consumer folds whatever planes are
reachable into one admission view.

### Plane 0 — durable, git-carried (shipped)

`refs/fak/locks/*` over the ordinary origin remote, converged by the new
**`fak leaseref sync`** (`internal/leaseref/sync.go`):

- **Push then fetch, order load-bearing.** Lease refs point at blobs — no ancestry —
  so both directions need a force refspec, and a force-fetch would regress a
  just-acquired local lease the remote hasn't seen (the caller's own fencing token
  silently rolled back). Publishing first makes the fetch a no-op for everything this
  clone last wrote. A **failed push stops the sync** for the same reason.
- **No prune, ever.** `fetch --prune` would delete this clone's not-yet-pushed
  acquisitions; `push --prune` would delete a peer's not-yet-fetched ones. Releases
  and reaps converge as **TTL expiry + each clone's own `fak leaseref reap`**.
- **Side refs only.** The refspec is confined to `refs/fak/locks/*` at both ends; no
  branch, HEAD, or tag ever moves.
- Honest boundary unchanged: two nodes writing the same id in one sync window
  last-writer-win on the remote. The conflict becomes **visible** (holder/generation
  mismatch at the next `Fence`), not prevented.

Latency: the sync cadence (a loop tick). Availability: anything that can reach
origin. This is the floor every node has. #2302 makes it ambient (sync at dispatch /
loop-drive / garden tick boundaries).

### Plane 1 — fast, dev-server (filed: #2297 read, #2299 write)

The gateway — already the fleet's front door — grows a lease surface for nodes that
can reach the coordinator (LAN/Tailscale):

- **Read plane first (#2297):** `GET /v1/leases` (the live_leases projection) and
  `GET /v1/sessions` (liveness classification), HTTP-latency freshness for any
  node's arbiter. Read-only: no admission decision moves.
- **Single-arbiter acquire second (#2299):** `POST /v1/leases/acquire|renew|release`
  executed against the coordinator's own store. The update-ref CAS that is only
  same-host-atomic today becomes **cross-machine-atomic** for every node that routes
  through the coordinator — the only honest closure of plane 0's atomicity gap, and
  it is opt-in per call: a node that can't reach the coordinator degrades to plane 0
  and is exactly as safe as today. Refusals stay deny-as-value in the closed
  vocabulary (LEASE_HELD / STALE_LEASE / LEASE_CONTENDED / NO_LEASE).

### Plane 2 — durable backup, human-visible (filed: #2300)

GH issue comments as the announce/audit channel: `fak leaseref announce` posts a
structured one-line comment (id, holder, generation, tree, TTL, action) to a
coordination issue; a parser folds comments back into an **advisory** view. A comment
is EVIDENCE, never a lock — it never enters admission on its own. This is the channel
that survives "node can reach GitHub's API but the git remote is wedged and the
coordinator is down", and it doubles as the always-on human-legible trail (the same
posture as the dogfood→tracker bridge and unwitnessedclaim).

## Node identity (filed: #2304)

`Record.Holder` is free-form today. The convention that makes all three planes
debuggable: **holder = `<node-id>/<session-id>`**, node-id stable per machine
(hostname-keyed, bound to the hardware catalog entry), session-id = the heartbeated
session descriptor. Sessions already answer "alive?"; node identity answers "where?".
Legacy free-form holders parse as node-unknown, never an error.

## A node's loop tick (the target protocol)

```
fak leaseref sync                  # converge plane 0 (fetch peers, publish self)
fak leaseref reap                  # bound crashed peers (TTL, fail-closed liveness)
laneadmit / dos arbitrate          # decide against the NOW-fresh live lease set
fak leaseref acquire --id L ...    # fenced take (via #2299 when reachable)
... work ...
fak leaseref fence --id L ...      # before every write/commit: still generation G?
fak leaseref renew --id L ...      # heartbeat while working
fak leaseref sync --push-only      # publish release/renewal promptly
```

## Honest boundaries (kept, not papered over)

1. Plane 0 is visibility + same-host CAS. Cross-machine races converge
   last-writer-wins and are caught at the **next fence**, not prevented.
2. Only a single arbiter (#2299) prevents them, and only for nodes that opt in and
   can reach it. There is deliberately no consensus protocol here — one coordinator,
   graceful degradation, no quorum theater.
3. GH comments are evidence, never locks.
4. Deletion convergence is TTL-bounded (no prune by design); a released lease may
   linger on peers up to TTL + one reap tick.

## Shipped with this note

- `internal/leaseref/sync.go` + `sync_test.go` — `Store.Sync`, exact-argv tested
  through the injected Runner seam (push-before-fetch order, fail-fast on push,
  refspec confinement, argv hygiene on the remote).
- `fak leaseref sync [--remote R] [--push-only|--fetch-only] [--dir DIR]` —
  `cmd/fak/leaseref.go`, documented in `docs/cli-reference.md`.
