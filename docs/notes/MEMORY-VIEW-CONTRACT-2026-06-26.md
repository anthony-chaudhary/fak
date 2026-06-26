# Memory-View Contract — typed virtual views with provenance gates (#904)

> Research/SOTA readout + design note. This is the "done when" deliverable for
> [issue #904](https://github.com/anthony-chaudhary/fak/issues/904): a SOTA/prior-
> art readout with primary sources, a typed view contract precise enough to spawn
> implementation children, one materialized view with source-span provenance that
> invalidates on a source-digest change, and a clean separation of exact
> prompt/KV-cache reuse from lossy semantic memory views. The contract is shipped
> as the `internal/memview` leaf; this note is the *why* and the *map*.

## The stance (one sentence)

A memory/context cell is **canonical** (the raw bytes); every summary, QA pair,
graph, prompt prefix, KV prefix, and skill-manifest slice derived from it is a
**typed virtual view** that carries **provenance** and an **admission gate**, and
no view ever executes a tool effect — a materialized view must re-enter
adjudication first. Memory is **not** one lossy summary blob.

This is the OS-memory analogue applied to agent memory: raw cells are pages, a
view is a (cached) virtual mapping, the source digest is the page-frame number,
and editing the raw bytes invalidates every mapping that pointed at the old frame
— exactly the cache-line / TLB-invalidation semantics an MMU uses. The name
`memview` and the `MemoryViewRecord` type are that analogy made concrete.

## SOTA / prior-art readout (primary sources)

| System | One-line | What fak takes from it | What fak does differently |
|---|---|---|---|
| [MemGPT (2310.08560)](https://arxiv.org/abs/2310.08560) | OS-style memory hierarchy (main/context + external) for LLM agents; the model pages memory in/out of its context. | The page-in/page-out framing; "external is canonical, context is a window." | fak's paging is a **real trust gate**, not a model decision: a quarantined cell is sealed *across the process boundary* (`recall`), and a view re-enters adjudication rather than being trusted because the model asked for it. |
| [MemOS (2507.03724)](https://arxiv.org/abs/2507.03724) | A "memory operating system" abstraction for memory-augmented generation; typed memory operations over a managed store. | The **typed-operation** stance: memory ops are first-class, not string blobs. | fak binds typing to a **content digest + byte span + taint**, and the type system is the closed `abi.TaintLabel` lattice + an open `ViewKind`, not a free-form schema. |
| [Mem0 (2504.19413)](https://arxiv.org/abs/2504.19413) | Production agent memory via selective extraction/retrieval (extract → update → retrieve). | Extractive accountability: a memory op names *what it came from*. | fak makes "what it came from" a **content address**, so an edit to the source is mechanically detectable (`IsValid` flips) rather than a best-effort de-dup. |
| [GraphRAG (2404.16130)](https://arxiv.org/abs/2404.16130) | Graph/community summaries over private corpora; a layered summarize-then-query index. | The "summary/graph is a **derived layer over raw**" separation. | fak refuses to treat a derived layer as canonical: a summary view's body does **not** hash to the source digest (`TestLossySummaryIsNotACanonicalFact`), so it can never stand in for raw bytes. |
| [LightRAG (2410.05779)](https://arxiv.org/abs/2410.05779) | Lightweight graph + vector retrieval for retrieval augmentation. | Cheap derived-index-with-provenance is a legitimate layer. | fak keeps the cheap index **off the hot path** and makes provenance a hard field, not a tag. |
| [HippoRAG (2405.14831)](https://arxiv.org/abs/2405.14831) | Neurobiologically-inspired long-term memory (pattern separation + completion) over a knowledge graph. | Selection integrity: *which* cell you bind matters as much as *that* you bound one. | fak encodes this as the **selector = producer** rule: a different selector is a different record, so a graph mutation is visible (`TestSelectorMutationIsADifferentRecord`). |
| [A-MEM (2502.12110)](https://arxiv.org/abs/2502.12110) | Agentic memory: a self-organizing note network the agent edits. | Memory is editable by the agent — so edits must be **detectable**. | fak's invalidation rule (`InvalidateOnDigestChange`) is exactly the detector: an agent edit to a raw cell changes its digest and invalidates every stale view bound to the old one. |

The convergent claim across all seven: raw memory is canonical; indices,
summaries, and graphs are derived and must stay accountable to the raw. fak's
contribution is to make that accountability a **machine-checked content-address
binding plus a taint gate**, in-kernel, off the hot path.

## Mapping current fak work onto the view table

The contract's value is that it names, in one place, what existing leaves already
do and where the seams are. The table below is ladder step 1 of the issue.

| Existing leaf | What it is | View-table interpretation |
|---|---|---|
| `internal/recall` | durable core image: a page table (roles + digests + quarantine state) over a content-addressed swap device; a sealed page stays sealed across the process boundary. | The **canonical raw cells**. A `recall.Page` is the `RawPage` a view derives from; `recall.Digest` is the same sha256-hex scheme `memview.Digest` uses, so a view's `Source.Digest` is interchangeable with a page digest. |
| `internal/promptmmu` | cache-prefix-preserving inbound prompt MMU; splices `tools[]` past the last `cache_control` breakpoint on the ORIGINAL bytes (a memcpy, never a re-marshal). | **Exact, lossless** prompt reuse. It is *not* a `memview` view — it never projects or summarizes; it preserves the cached prefix byte-for-byte. See the separation below. |
| `internal/vcachechain` | the vCache prefix DAG + the §11.0 cost-gated rebuild decision (replay a chain or send cold). | **Exact KV-prefix reuse** as a *decision* (replay vs cold). The KV materialization itself is the lossless twin of prompt reuse; the recall decision is off-path. Distinct from a lossy semantic view. |
| `internal/ctxplan` | the reachability layer: an O(1) resident *view* over a lossless history store, with page-fault recovery for elided spans. | A **planned residency view** — which raw cells are resident. It composes with `memview`: a plan says *what is resident*, `memview` says *what a derived projection off a resident cell is admissible as*. |
| `internal/ctxmmu` / `internal/kvmmu` | the write-time byte gate and the KV-eviction bridge; a quarantine verdict evicts the matching K/V span. | The **taint gate** a view inherits. `memview.VerdictFor` consumes `abi.TaintLabel`; a `TaintQuarantined` source quarantines the view. |
| `internal/memview` (this issue) | `MemoryViewRecord` + a digest/byte-span provenance binding + a materialization verdict. | The **typed view contract** itself — the seam that names the derived-projection discipline the other leaves already assume. |

## The minimum `MemoryViewRecord`

The shipped type (`internal/memview/memview.go`) carries the fields the issue's
"minimum local MemoryViewRecord" question asks for:

- **raw input digest** → `Source.Digest` (binds the view to exact source bytes)
- **view type** → `Kind` (`snippet` | `summary` | `qa` | `fact` | …)
- **producer** → `Producer` (generator + version; *the selector*, so a graph-
  selector mutation is a producer change)
- **source spans** → `Source` (`Digest` + `Offset` + `Length`, a byte window)
- **scope/taint** → `Taint` (inherited `abi.TaintLabel`)
- **freshness** → `Freshness` (monotonic materialization epoch)
- **quality witness** → `Witness` (optional external trust witness)
- **invalidation rule** → `Invalidation` (default `InvalidateOnDigestChange`)

Prompt/model/tool schema versions are carried *implicitly* by `Producer` (a
schema bump is a new producer string) — kept out of the core type so the contract
stays a tier-2 mechanism importing only the frozen `abi`.

## Selection integrity (the open question, fenced honestly)

The issue asks: *"a view with good source links can still be unsafe if the
selector/graph structure was poisoned; the selector and structural writes need
provenance too."* The contract's answer, implemented and tested:

1. **The selector is the producer.** `Materializer.Producer` is stamped on every
   record. Two views of the *same* span under *different* selectors are *different
   records* (`TestSelectorMutationIsADifferentRecord`) — a graph mutation surfaces
   as a producer change, never a silent in-place rewrite.
2. **The provenance is the binding.** `Source.Digest` + `Source.Offset/Length`
   name exactly which bytes the view came from. A selector that claims a span it
   does not occupy is refused at materialization (`ErrSpanOutOfRange`).
3. **A lossy summary is never a fact.** `MaterializeSnippet` is the only shipped
   auto-materialization and it is lossless (a verbatim sub-slice). A summary/fact
   view is a *contract surface* (`KindSummary`/`KindFact`), not an auto-allowed
   materialization — an unsupported summary claim cannot be materialized as a fact
   without a future quality witness (`TestLossySummaryIsNotACanonicalFact`).

The remaining selection-integrity rung — a signed/structural witness over the
*graph selector itself* (not just the bytes it selected) — is an explicit
follow-on child, not claimed here.

## Exact reuse vs lossy view — the separation (#904 done-criterion 4)

This is the conflation the issue exists to kill, and the contract enforces it:

- **Exact, lossless reuse** (`promptmmu`, `vcachechain`): the question is "are
  these the same bytes / the same prefix?" The answer is a byte-identity check;
  the payoff is a cache hit. No projection, no summarization, no information loss.
- **Lossy semantic view** (`memview` `summary`/`qa`/`fact`): the question is "here
  is a derived projection of canonical bytes; is it admissible, and is it still
  valid?" The answer is a verdict + a digest check; the payoff is a smaller /
  structured context. Information *is* lost — that is the point — so it can never
  masquerade as canonical.

The `internal/promptmmu/doc.go` note states this separation explicitly so a reader
landing on either surface does not confuse a cache-prefix splice with a semantic
projection.

## What this contract owns vs the implementation children

Per ladder step 5, the cross-links and the ownership fence:

- Owns (this contract): the `MemoryViewRecord` type, the digest/byte-span
  provenance binding, the `IsValid` staleness gate, the `VerdictFor` taint gate,
  and the one shipped lossless materialization (`MaterializeSnippet`).
- Does **not** own (children, deliberately fenced):
  - [#751](https://github.com/anthony-chaudhary/fak/issues/751) / [#758](https://github.com/anthony-chaudhary/fak/issues/758) — the inbound prompt-MMU rungs (what *enters* the context window, cache-prefix-safe). `memview` is downstream of / orthogonal to prompt admission; a view body still crosses promptmmu to enter context.
  - [#782](https://github.com/anthony-chaudhary/fak/issues/782) / [#784](https://github.com/anthony-chaudhary/fak/issues/784) / [#786](https://github.com/anthony-chaudhary/fak/issues/786) — ECC-style integrity, patrol-scrub, and parity over `recall` cells. Those protect the *raw cell* (the source); `memview` consumes their guarantees but adds the *view*-level binding on top.
  - [#844](https://github.com/anthony-chaudhary/fak/issues/844) — the `ctxplan` reachability layer (ref-counting / pinning / GC over the context heap). `memview` is a *projection* contract; `ctxplan` is the *residency* contract. A resident cell (`ctxplan`) can carry derived views (`memview`); the two compose without overlap.

## Witnesses

The contract is shipped green; the load-bearing properties are test-pinned in
`internal/memview/memview_test.go`:

- `TestSnippetRoundTripsSourceBytes` — a snippet body is the exact source sub-slice (lossless accountability).
- `TestChangedSourceInvalidatesView` — **the crux**: a view bound to digest A is stale once the source's current digest is B; empty digest is fail-closed stale.
- `TestTaintedSourceQuarantinesNoBody` — a tainted/quarantined source yields `Quarantine` with no body; provenance is still returned for audit.
- `TestSealedSourceRefusesBytes` — a sealed upstream page (the recall/ctxmmu seam) refuses materialization.
- `TestSpanBounds` — out-of-range / empty spans are refused.
- `TestLossySummaryIsNotACanonicalFact` — a summary body does not hash to the source digest and is never auto-materialized as a fact.
- `TestSelectorMutationIsADifferentRecord` — selection-integrity: a selector change is a producer change.
- `TestDeterminism` / `TestDigestMatchesSha256HexScheme` — purity and content-address interoperability with `recall.Digest`.

Run: `go test ./internal/memview/` (and `go test ./internal/architest/` for the
tier/layering gate that admits the leaf).
