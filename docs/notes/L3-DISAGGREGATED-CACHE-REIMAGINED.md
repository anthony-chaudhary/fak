# Re-imagining the external L3 disaggregated KV cache for fak — the CAMA semantics layer

> Canonical positioning artifact for epic **#79**. This is a *planning and positioning*
> study, not a measured benchmark: it fixes the relationship between fak and an external
> L3 KV-cache tier (CAMA) and maps each of fak's shipped semantic primitives onto a gap
> that the cache layer leaves open by design. Every "fak fills this" claim below points at
> a primitive that already exists in this repo (file:line cited). Every *tier-projection*
> of those primitives onto the L3 tier is still **unbuilt** — that is the work the children
> of #79 track, not something this doc asserts is done.

## The system we are positioning against

`C:\work\cama-complete` (CAMA) is an external, production-shaped disaggregated **L3
KV-cache** tier for LLM serving: a Go `cama-server` plus vLLM / SGLang connectors that
back a remote DRAM pool over RDMA / TCP, content-addressed by `SHA256(prefix)`,
slab-allocated, swiss-table-indexed, and W-TinyLFU-evicted. It is a strong implementation
of the three layers fak deliberately does **not** rebuild — routing, addressing, fusion —
and it is semantics-free by its own stated design:

> "CAMA does NOT verify content; it assumes the connector's hash is the source of truth.
> No checksums, no re-validation."

That sentence is the whole opening. CAMA places opaque cache cells across a fabric; it is
the Kubernetes of the KV tier. fak defines what the cell *is* — its mutation rules,
isolation, provenance, and deletion-proof — so fak is the Docker. The two **compose**.
fak rides *above* CAMA as the admission-and-attestation referee in front of its `get` /
`set`, and *behind* its hashes as the verifier. Positioning a fak-native L3 store *against*
CAMA would be a category error: it would rebuild the routing / addressing / fusion fence
that fak's layering explicitly leaves to the cache.

## Why this is a fak win, not a "build a faster cache" trap

CAMA sees an anonymous page key and must trust it blindly. It has no principal, no turn
structure, and no provenance, because the serving connector destroyed all of that before
the bytes reached the wire. fak sits at the **agent syscall boundary**, where the page's
owner (an agent-scoped, tainted `Ref`), its trust label, and the turn that produced it are
*given*, not inferred. fak can attach the semantics CAMA cannot precisely because fak still
holds the structure CAMA's client already threw away.

So the rule that governs every child of this epic:

**Do not rebuild CAMA's L3 store in Go.** The slab allocator, swiss-table, RDMA-ODP, and
W-TinyLFU are the routing / addressing / fusion layers fak does not change. A fak-native
L3 store would be a worse CAMA and a crowded-loser claim. fak's value is the **semantics
layer in front of and behind** a transport it does not own.

## The 8-gap map — each CAMA omission against the shipped fak primitive that fills it

Each "shipped fak primitive" below is a real, tested, single-box primitive in *this* repo.
The "tier-projection" column names the child of #79 that would carry that primitive onto a
real multi-node L3 tier — and every one of those projections is unbuilt.

| # | CAMA does NOT (by design) | Shipped fak primitive (in this repo) | Tier-projection child |
|---|---|---|---|
| G1 | verify a fetched page is the correct one (trusts the hash) | `Ref.Digest` content-hash identity — `internal/abi/types.go:64` | sidecar (child A) + child D |
| G2 | evict a poisoned span from the **middle** of a shared sequence | `KVCache.Evict` / `CanEvict` / `TryEvict` — `internal/model/kvcache.go:92` | child B (`L3RegionBackend`) |
| G3 | prove a deletion happened over a shared pool | `DeletionCertificate` mint + verify — `internal/deletioncert/deletioncert.go:133` | child E (**the headline**) |
| G4 | isolate one tenant's pages beyond a flat `tag` | closed `ShareScope` (Agent / Fleet / Tenant) — `internal/abi/types.go:93` | sidecar + child D |
| G5 | track provenance — who wrote a page, and whether to trust it | IFC taint + sink-gate — `internal/ifc/ifc.go` | sidecar (child A) |
| G6 | decide what *deserves* to persist vs expire | S7 durability axis (epic #496) | child C (durability-tiered promotion) |
| G7 | arbitrate two semantic writers racing on one span | DOS lease-lane / `arbitrate` | child B |
| G8 | refuse with a checkable reason | closed `ReasonCode` vocabulary — `internal/abi/reasons.go` | sidecar (child A) |

The seam that lets a child attach without forking fak is already frozen: the `Resolver`
interface behind every `Ref` (`internal/abi/types.go`, registered via
`RegisterRegionBackend`, `internal/abi/registry.go:527`). Swapping the copy-CAS default for
an L3-backed `RegionBackend` is a backend change, not an ABI change — which is exactly why
child B can ride the existing seam instead of editing the kernel.

## The single sharpest constraint — governs every child

Keep fak on the **control / admission path, never the data path.** CAMA's entire value is
the roughly 1–5 µs L3 RDMA read. Per-op digest verification of a multi-megabyte page at
line rate would destroy that budget. This rules **for** a sidecar / resolver split — fak
admits, attests, and verifies *out of band* on the control path — and **against** an inline
byte-scanner on every read. Every child must state plainly whether it touches the data
path, and justify it if it does. The default answer is "control path only."

## The two theses a first rung must prove

1. **Verify, don't trust (G1).** CAMA trusts the connector's hash as the source of truth.
   fak holds `Ref.Digest` at the syscall boundary, so it can verify a fetched page is the
   page it claimed to be — turning CAMA's "no re-validation" from a silent assumption into
   a checked admission step.
2. **Truth has a duration (G6).** Not every page deserves to persist. fak's S7 durability
   axis (epic #496) is the place to decide what is promoted into a durable L3 tier and what
   is left to expire, instead of letting the cache's eviction policy make a semantic call it
   has no information to make.

## The integration ladder — children of #79

Ordered by depth. The recommended first rung is this design artifact (now in-repo) followed
by a thin control-path sidecar on G1 + G6 — highest value per hour, no CAMA fork, control
path only.

- [ ] **child A — referee sidecar:** durability-gated L3 admission (G6) + return-digest
  verification (G1), control path only. The ship-next rung. Depends on the G6 classifier
  (epic #496 / its #498 child) and, to *prove the theses against a real external store*, on
  a reachable CAMA instance.
- [ ] **child B — `L3RegionBackend`** behind the frozen `Resolver` seam
  (`internal/abi/types.go`): unlocks G2 (middle-evict) and G3 (cert) over the tier. Large;
  depends on the network transport (#493) and an in-kernel engine.
- [ ] **child C — durability-tiered promotion** (G6 in isolation): L3 becomes the "durable
  tier" the S7 axis promotes *into*. Extends epic #496 onto a real multi-tier store.
- [ ] **child D — verified cross-tenant prefix-sharing** (G1 + G4): make CAMA's riskiest
  optimization provably safe.
- [ ] **child E — portable per-span `DeletionCertificate` for a shared L3 pool** (G3, the
  headline): provable forgetting across a multi-tenant cache tier, which a semantics-free
  cache structurally cannot produce. Builds on the shipped mint/verify primitive
  (`internal/deletioncert/deletioncert.go`).

## Honest limits — carry these into every child

- The deletion certificate proves the KV **working set** is gone, not that data never
  leaked into fine-tuned weights, embeddings, backups, or replicas. CAMA's "no replication"
  actually helps here (one copy to prove gone), but the cert must scope honestly — the
  primitive already names its surface via `Scope` (e.g. `l3-working-set`), and a child must
  not widen that claim.
- The secret-pattern detector is lexical and evadable. IFC taint (G5) is the
  phrasing-independent backstop, not the detector.
- This is a planning epic. Of the eight primitives, the single-box versions are shipped and
  tested; **every tier-projection here is unbuilt.** No row above should be read as "fak
  already does this across an L3 tier."

## Open forks for the operator

1. Real CAMA integration, or CAMA-as-exemplar (positioning only)?
2. Control-path-only (cheaper, weaker proofs) vs paying the data-path cost for full G1–G8?
3. Does fak run the model in-kernel here, or co-reside with SGLang? This gates how deep
   child B can go.
4. Is the deletion certificate (child E) the lead, or durability-tiering (child C)?

## Definition of done for the epic

This study is the canonical positioning artifact (this document closes that clause). Child
A ships a thin control-path sidecar proving the two theses — verify-don't-trust and
truth-has-a-duration — against a real external store. Children B–E are tracked as
follow-ons, not blockers on A.

The honest status as of this artifact: **clause 1 (the positioning study) is satisfied
in-repo; clause 2 (child A's sidecar) and children B–E remain unbuilt.** Child A cannot be
demonstrated "against a real external store" from this workspace alone — CAMA lives in a
separate tree (`C:\work\cama-complete`) that this module does not import or build, and
fak's own guard refuses writes outside the repo — so that demonstration is gated on a
reachable CAMA instance plus the G6 classifier (#498), independent of this artifact.

## Related, in this repo

- `docs/CONTEXT-IS-NOT-MEMORY.md` — why context assembly is not durable memory.
- `internal/abi/types.go` — the `Ref` / `Resolver` / `ShareScope` ABI this study rides.
- `internal/deletioncert/deletioncert.go` — the shipped mint/verify behind child E.
- `internal/model/kvcache.go` — the middle-evict surface behind child B (G2).
- `internal/ifc/ifc.go` — the taint / sink-gate backstop (G5).
- `internal/abi/reasons.go` — the closed refusal vocabulary (G8).

Related epics: #496 (S7 durability), #493 (network StageTransport). External source system:
`C:\work\cama-complete` (CAMA).
