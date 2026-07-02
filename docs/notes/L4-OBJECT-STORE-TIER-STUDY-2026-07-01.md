# The object store as a first-class L4 cache tier — positioning study

> Canonical positioning artifact for epic **#2169** (first-class object-store /
> blob L4 cache tier: stateful offload + verified access). Like the L3 study
> ([`L3-DISAGGREGATED-CACHE-REIMAGINED.md`](L3-DISAGGREGATED-CACHE-REIMAGINED.md))
> this is a *planning* document, not a measured benchmark: every "fak fills this"
> claim points at a primitive that exists in this repo today (file cited), and
> every projection of those primitives onto the object tier is **unbuilt** — that
> is the work the children of #2169 track.

## Naming, first (two collisions this repo already carries)

- **"L4" here means the fourth cache level**, the rung under the L3 disaggregated
  DRAM pool — **not** the NVIDIA L4 GPU that `cmd/fak/serve.go` and the bench
  fleet notes mean by "L4". Prose in this doc and the epic always says
  "object-store tier (L4)" on first use.
- **The tier token is `object`, the client package `internal/objstore`** — not
  "blob", because `internal/blob` is already the *in-memory* content-addressed
  store, and its `blob.Digest` identity is exactly what the new tier reuses.
  A disambiguation row accompanies the tier (epic child, WS6).

## The system we are positioning

MinIO, AWS S3, Cloudflare R2, GCS, Ceph — the S3-compatible object store. Its
physics are the opposite corner from the L3 DRAM pool (`TierRemote`,
`internal/cachemeta/hardware.go`): **persistent** where L3 is volatile,
**effectively unbounded** where L3 is capacity-bound, **reachable from every
node and region** where L3 is fabric-local, **cheapest per byte** — and three to
five orders of magnitude **slower to first byte** (~10–200 ms vs ~1–5 µs). It is
also the one external tier almost every deployment already operates. Nobody has
to stand up an RDMA cache cluster to get it.

## Where it sits (the L3/L4 answer)

The residency ladder today (`internal/cachemeta/hardware.go`, `TierRank`):

```
HBM → DRAM → NUMA-far → CXL → Disk → Remote(L3 DRAM pool) → Provider → Recompute
 0     1        2         3     4       5                      6          7
```

The object tier slots at **rank 6, between the L3 pool and the provider cache**:
colder and durabler than `TierRemote`, still a real store fak can verify —
unlike `TierProvider`, which is opaque. Two compositions, both first-class:

1. **L3-less — the common deployment.** Ladder: HBM → DRAM → Disk → **Object
   (L4)** → Recompute. Most operators have an object store and no disaggregated
   DRAM pool, so L4 is the *first external tier a normal deployment can turn
   on*: the spill target past disk (`PlanPlacement`'s demote-instead-of-evict
   argument extends one rung — a ranged-GET restore that beats a re-prefill),
   and the warm-start source for every other node.
2. **Under L3 — the durability floor.** The DRAM pool stays the hot fleet tier;
   L4 is what L3 spills to and warm-starts from, and the concrete "durable tier"
   the S7 durability axis (epic #496) promotes into — child C of the L3 epic
   (#79) finally gets a real target.

**What belongs in L4:** cold KV spans whose restore beats recompute; vCache
snapshots (`internal/vcachesnapshot`); session envelopes
(`internal/session/envelope.go`) for cross-node resume; owner-marked
`ScopeFleet` shared prefix pages. **What does not:** hot per-token KV — the
latency physics forbid it, and no placement policy should ever be able to route
a decode-path read at an object store.

## Why this is a fak win (same thesis as L3, sharper here)

An object store is even more semantics-free than the L3 pool: it stores
anonymous bytes under anonymous keys, forever, for anyone with a credential. It
has no principal, no provenance, no scope, no deletion proof — and unlike the L3
pool it is *durable*, so every mistake persists. fak holds all of that structure
at the syscall boundary (`Ref.Digest`, `ShareScope`, IFC taint, the S7
durability axis, the DOS lease lanes, the closed refusal vocabulary), so the
8-gap map of the L3 study projects onto L4 unchanged — G1 verify-don't-trust,
G3 deletion certificates, G4 scope isolation (a bucket prefix is NOT an
isolation mechanism), G5 taint travel, G6 what-deserves-to-persist, G7 write
arbitration, G8 typed refusals. The gateway gates for G1/G4 already exist in L3
form (`internal/gateway/l3share.go` — `AdmitL3SharedPage`) and the L4 twins ride
the same shape and, where honest, the same tokens.

**Do not rebuild the store.** MinIO/S3 are the routing/addressing/durability
layers fak does not change. fak is the admission-and-attestation referee in
front of `get`/`put` and the verifier behind the content addresses.

## Design tenets (the constraints every child obeys)

1. **Control path only.** fak admits, verifies, attests; bulk bytes flow
   client-direct (presigned URLs are the L4 form of the data-path bypass). A
   child that touches the data path says so and justifies it.
2. **Same cells at N tiers.** The content address is `hex(sha256)` —
   byte-identical to `blob.Digest`, `internal/l3region` page keys, and
   `Ref.Digest`. An object key IS the content address; identical content dedups
   for free, cross-node.
3. **Ride the frozen seams.** `abi.RegionBackend` (the Resolver behind every
   `Ref`), `abi.KVBackend` (`StageSpan`/`RestoreSpan` → typed `KVResidency`,
   already lowered onto the lookup trichotomy by
   `internal/cachemeta/kvresidency.go` — a failed restore is a typed MISS/FAULT,
   never a silent recompute), and the `cachemeta` placement/lifecycle plane.
   Backend swap, not ABI change.
4. **Zero external dependencies.** The module is stdlib-only (no go.sum), so the
   S3 client is a pure-stdlib SigV4 + `net/http` implementation, MinIO-first
   (path-style), conformance-tested against a real MinIO in an env-gated
   harness. This is a feature: the whole tier installs with `go install`.
5. **Costs are honest.** Placement gains a dollar axis (storage + egress) next
   to the time axis; the cache-value ledger (`internal/cachevalueledger`) books
   L4 wins net-true per `docs/standards/net-true-value.md`; externally-observed
   values carry OBSERVED provenance labels.

## Honest limits

- A deletion certificate over a versioned / replicated / soft-deleting bucket
  proves the *addressed working set* is gone; it cannot promise the provider
  kept no replica. The cert scopes honestly or refuses to mint.
- Client-direct presigned reads cannot be digest-checked inline by fak; the
  bypass child carries that fork explicitly (client-side verify handshake vs
  post-hoc audit) rather than silently weakening G1.
- Tier-profile latency defaults are representative; the probe measures the
  operator's real endpoint before placement trusts them.
- Everything here is unbuilt until a child ships with its witness; claims enter
  `CLAIMS.md` per rung under the `[STUB]`/`[SIMULATED]`/`[SHIPPED]` discipline.

## Prior art (consult before building — `fak sota`)

LMCache and the vLLM KV-connector family, Mooncake Store, NVIDIA NIXL / Dynamo,
SGLang HiCache, AIBrix KV offloading — all place KV on colder tiers including
object storage. The SOTA-matrix child records the borrow/bind/stay-minimal route
per operation; kernel-adjacent commits carry `Prior-art:` trailers.

## The work (children of #2169, by workstream)

- **WS1 — tier model foundation** (`cachemeta`/`abi`): `TierObject` in the
  ladder + rank, profile + live probe, spill-to-object placement, dollar-cost
  axis, lifecycle/invalidation, budget-pressure for an unbounded tier,
  presigned-handoff share descriptor.
- **WS2 — `internal/objstore`**: typed interface, SigV4 + presign, credential
  chain, core ops, ranged GET, multipart, retry/backoff, end-to-end integrity,
  MinIO conformance harness.
- **WS3 — L4 backends behind the frozen seams**: `ObjectRegionBackend`,
  versioned manifest, MiB-scale segments + index, `KVBackend` stage/restore,
  async offload queue, restore prefetch, disk staging cache, idempotent dedup
  PUT.
- **WS4 — semantics gates (G1–G8 over L4)**: digest gate, scope gate, envelope
  encryption, deletion certificate, taint persistence, durability admission,
  manifest write leases, closed `L4_*` refusal vocabulary.
- **WS5 — access path + serve wiring**: presigned bypass, serve config/policy,
  metrics, value-ledger booking, snapshot + session-envelope offload/resume,
  cross-node warm start end-to-end.
- **WS6 — ops/CLI/docs/proof**: `fak l4` verbs, GC, failure-injection matrix,
  MinIO example, interop notes, crossover benchmark, SOTA rows, CLAIMS rows,
  disambiguation rows.

## Definition of done for the epic

The ladder places, spills to, and restores from a real MinIO with
digest-verified, scope-gated access; a cold node warm-starts a span another node
offloaded; the value ledger books the win net-true; the failure matrix fails
closed with typed reasons; prior art is recorded. Until each rung's witness
exists, its status is `not yet`.

## Related, in this repo

- `docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md` — the L3 study this extends.
- `internal/cachemeta/hardware.go` · `placement.go` · `lifecycle.go` — the
  tier/placement plane the object tier joins.
- `internal/abi/registry.go` (`RegionBackend`, `KVBackend`) — the frozen seams.
- `internal/l3region/` — the Stage-1 L3 backend the L4 backend mirrors.
- `internal/gateway/l3share.go` — the G1/G4 admission gate shape to reuse.
- `internal/blob/` — the in-memory CAS whose digest identity the tier shares.

Related epics: #79 / #504-family (L3 children A–E), #493 (network
StageTransport), #496 (S7 durability axis).
