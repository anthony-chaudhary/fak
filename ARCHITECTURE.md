# fak architecture — the extension model ("other ideas bake in")

> **Researcher / team building an optimization for a subsystem?** Start with the
> task-first golden path in [`EXTENDING.md`](EXTENDING.md) — *plug in → prove correct →
> prove faster → ship* — then come back here for the full seam catalog. This document is
> the reference; `EXTENDING.md` is the on-ramp.

The whole point of wave 0 is the **frozen ABI** (`internal/abi/`). It is the one tree
every worker imports, so it can never change after freeze without colliding every
worker. The design goal is therefore **a stable minimal spine with real extension
seams** — open enough that any future idea attaches as *a new package + one
`Register*()` call + (optionally) one additive envelope field guarded by a
Capability*, never an edit to the spine — while avoiding the opposite trap of
vaporware "everything is an interface."

## The dependency graph is a layered DAG (enforced by `internal/architest`)

> Correction (2026-06-17): an earlier version of this section called the graph a
> *star* — "every leaf imports only `abi`." That was aspirational; `go list` shows a
> layered DAG (`agent`→7 leaves, `ifc`→`provenance`, `recall`→`ctxmmu`, …). The real,
> **enforced** contract is the five-tier layering in `fak/GROWTH.md` §2, checked by
> `internal/architest` (no upward imports; every leaf declares a tier). What keeps the
> `dos-arbitrate` leases disjoint is the **file-tree** disjointness below — each leaf is
> its own directory — which is true independent of the import edges.

```
                       internal/abi   (FROZEN — no worker may lease it)
                      /   |   |   |   \
   adjudicator vdso preflight ctxmmu model gateway recall ... (30 leaves)
                      \   |   |   |   /
                   internal/registrations  (blank-imports the built-in leaves)
                              |
                           cmd/fak
```

Leaves form a **layered DAG**: a leaf may import lower-tier leaves (e.g. a `composer`
imports `mechanism`s and `foundation`), never a higher tier — `internal/architest`
fails the build on an upward import. A new idea is still a brand new directory + one
blank-import line in `internal/registrations` (use `python tools/new_leaf.py`). Because
each leaf is its **own directory**, two ideas added in parallel by two fleet workers
edit **disjoint files** and cannot collide — which is what keeps the `dos-arbitrate`
file-tree leases disjoint (`dos.toml` declares one lane per leaf), regardless of the
import edges between them.

## How a new idea bakes in (the only mechanism)

A driver package registers itself from `init()` against `internal/abi`:

| To add… | Call | Result |
|---|---|---|
| a policy/PEP rung | `RegisterAdjudicator(rank, impl)` | a new link in the LSM-style chain |
| a vDSO tier | `RegisterFastPath(tier, impl)` | a new local fast-path answer |
| an operation (async submit, spec commit) | `RegisterOp(impl)` | a new entry in the io_uring-style op table (panics on opcode clash) |
| a verdict kind | `RegisterVerdictKind(k>1023, name, foldRank, fallback)` | open-range kind with a declared lattice position |
| a refusal reason | `RegisterReason(code, name)` | additive label-space entry |
| a KPI / steward / label tap | `RegisterEmitter` / `RegisterSteward` | a new observer |
| an engine (local/remote/multi) | `RegisterEngine(id, impl)` | a new backend behind the selector |
| the Ref backend (zero-copy) | `RegisterRegionBackend(impl)` | a Resolver swap (copy → shared arena) |
| an MMU codec (headroom) | `RegisterPageOutBackend(id, impl)` | a swappable page-out backend |
| a witness type | `RegisterWitnessResolver(id, impl)` | backs the require-witness verdict |

The kernel **walks** these registries (`abi.Adjudicators()`, `abi.FastPaths()`,
`abi.LookupOp()`, …); it never imports a driver. Enabling/disabling an idea is one
import line in `internal/registrations`.

## The scaling contract: the hot path stays O(1) as ideas accumulate

The registries are written **once** at `init()` (one `Register*` per enabled idea)
and then read on **every syscall**. So the design rule is *writes may be expensive,
reads must be O(1) and wait-free no matter how many ideas are registered* — the
1000th idea must cost the 1st syscall nothing in framework overhead. Three
mechanisms enforce this (`internal/abi/registry.go`):

1. **Reads load an immutable snapshot, never a lock+copy.** Every accessor
   (`Adjudicators`/`FastPaths`/`Emitters`/`FoldRank`/`Engine`/…) is a single
   `atomic.Pointer` load that indexes a pre-built slice/map. A `Register*` rebuilds
   one immutable `snapshot` and publishes it; readers take **no mutex and allocate
   nothing**. Guarded by `TestRegistryReadsZeroAlloc` (0 allocs/op with 256 drivers)
   and `BenchmarkRegistryReadScaling` (flat ns/op across N=1→1000).

2. **Event fan-out is indexed by kind**, so `emit()` (called several times per
   syscall) runs `O(observers subscribed to this kind)`, not `O(all observers)`. An
   observer scopes itself with the optional `EventSubscriber{ Subscriptions() }`;
   one that doesn't is universal (gets every kind) — the v0.1 default.

3. **Both folds are per-tool.** A driver scopes itself with the optional
   `CallScope{ Tools() }`. For the pre-call **adjudicator** fold, a call for tool
   *T* folds only the unconditional rungs plus the rungs scoped to *T*
   (`abi.AdjudicatorsFor(c)`); for the result-side **admitter** fold, a result for
   tool *T* folds only the unconditional gates plus the gates scoped to *T*
   (`abi.ResultAdmittersFor(c)`). One generic primitive (`byToolScopeIndex[T]`)
   backs both. A driver that doesn't implement `CallScope` is **unconditional /
   always-run — the fail-CLOSED default**, so this never weakens a security
   decision: skipping a contract-honoring scoped driver is verdict-equivalent to
   running it (an adjudicator self-Defers, an admitter self-Allows, for unlisted
   tools). Proven by `TestScopedFoldEquivalentToFullChain` and
   `TestScopedResultAdmitterRoutesByTool`.

**Rule for the next feature:** if your rung/gate/observer only applies to specific
tools/events, declare it (`CallScope` / `EventSubscriber`). Then the 100th
tool-specific policy costs an unrelated call **nothing** — adding features stays
O(1) on the hot path. And the rule applies *inside* a driver too: a driver's own
per-call work must be O(this call), not O(policy size) — index your rules by tool
at install time, the way the rank-100 monitor groups `ArgPredicates` by tool
(`internal/adjudicator`, `BenchmarkAdjudicateArgScaling` shows it flat vs policy
size). The only per-call cost that *should* grow is running the rungs that
genuinely apply to that call.

| Optional scoping interface | Implemented by | Effect |
|---|---|---|
| `CallScope{ Tools() []string }` | an `Adjudicator` or `ResultAdmitter` | folded only into calls for those tools (default: every call) |
| `EventSubscriber{ Subscriptions() []EventKind }` | an `Emitter` | receives only those event kinds (default: every kind) |

## The four seams that MUST be frozen now (a miss = fleet-wide recompile)

These cannot be added later without breaking the shared import, so they are all in
`types.go` today, defaulted so v0.1 ignores them:

1. **Verdict is an additive discriminated union** — `Kind` is a closed trainable
   enum below `VerdictReservedMax` (1023) with an open registered range above;
   `Payload` is keyed by `Kind` so a malformed verdict is *unrepresentable*; a
   registered kind declares a **`foldRank`** so the frozen fold can order it
   without a core edit; an unknown kind resolves via its **`FallbackClass`**
   (fail-closed) and never panics.
2. **Payloads are addressable `Ref`s, not copied bytes** — bytes only materialize
   via `Resolver`. v0.1 backs `Ref` with a content-addressed blob store (a copy);
   zero-copy KV co-residence (brainstorm §3.1a) is a `RegionBackend` swap behind
   Capability `"zerocopy"`. `Ref` also carries `Taint` + `Scope`, so the
   cross-agent shared-result pool has somewhere to express isolation.
3. **Sync `Syscall` is defined OVER async `Submit`/`Reap`** — adjudication always
   happens at `Submit`, so adding io_uring-style async (brainstorm §2.7) never
   splits the single chokepoint. `Completion` and `SubmissionHandle` are *typed*,
   so two async drivers can't collide on the semantics of a shared cursor.
4. **A provisional lifecycle rides the envelope** — `SpeculationContext` + `TxnID`
   + `Outcome` + the `ProvisionalSink` interface mean speculative
   commit/squash (§2.6) and transactional context / KV checkpoint-rollback (§3.4)
   are a backend concern, not an ABI change. Effects under a non-zero epoch/txn are
   provisional until `Promote`/`Rollback` — so "squash actually retracts the
   effect" is a frozen cross-driver contract, not a gap discovered at integration.

## Bake-in walkthrough (all `touchesCore = false`)

- **Speculative execution** → `internal/spec`: registers `OpSpecCommit`/`OpSpecSquash`
  from the reserved `OpsSpec` range; rides `ToolCall.Spec`; the MMU's
  `ProvisionalSink` retracts squashed effects. `Outcome` already has `OutcomeSquashed`.
- **Async / io_uring** → `internal/async`: registers `OpSubmit`/`OpReap`, advertises
  Capability `"async"`; returns `Status=Pending` + typed `Completion`s. Old workers
  never negotiate `"async"` and only ever see synchronous results.
- **Zero-copy fusion** → no message-layer change at all: `Args`/`Payload` are
  already `Ref`s; ship a `RegionBackend` whose `Resolver` hands out `RefRegion`
  handles into a shared arena, advertise `"zerocopy"`.
- **Syscall-tuned small model** → nothing new in the ABI: `ToolCall` is the typed
  input target, the closed `VerdictKind`+`ReasonCode` set is the trainable output
  target, and rung transitions already emit typed `LabelRow`s. The model is later
  wired as one more `Adjudicator`; the fold bounds it even if it emits a kind it
  shouldn't.
- **Unforeseen (e.g. a federated cross-fleet trust gate)** → `internal/fedtrust`:
  one `Adjudicator` ahead of dispatch, advertises `"trust.federated"`, registers a
  new `VerdictKind > 1023` with `FallbackDeny`, carries its score on `Result.Ext`.
  The core never learns federation exists.

## A sibling seam — the in-kernel model's device compute (`internal/compute`)

The registries above live in the frozen `internal/abi`. The model leaf's hardware
portability rides a **separate** registry, `internal/compute` (`compute.Register(Backend)`),
deliberately outside the shared ABI because device compute is internal to the model, not a
cross-worker contract. It obeys the same discipline — a new backend is a new file + one
`Register` call, never a forward-loop edit — and it now carries real load: `cpu-ref`
(Reference) plus `cuda` and `vulkan` (Approx), each witnessed on real silicon (RTX 4070 in
`GPU.md`, Radeon RX 7600 in `VULKAN-AMD-RESULTS.md`). Do not confuse it with `RegisterEngine`,
which attaches an *OpenAI-compatible serving* engine — a different layer (the model that
answers tool calls, not the kernels the in-kernel model runs on).

## What stays CONCRETELY pinned (not vaporware)

The exact `Syscall`/`Submit`/`Reap` signatures; five named-field wire structs; a
six-value closed `VerdictKind` enum below a hard boundary; `Ref` as a real struct
with a `RefKind` discriminant and a `Digest`; a closed `Status`/`Outcome`/`TaintLabel`/
`ShareScope` vocabulary. Openness is only at **named seams**, each backed by a
one-method interface with a real signature. The v0.1 subsystems are *forced* to
attach through these exact registries, proving the seams carry load before any
future idea uses them. The Adjudicator fold (provable→`Deny`, unprovable→`Defer`)
is lifted directly from the shipped `dos-preflake/go/internal/hook/decide.go`.
