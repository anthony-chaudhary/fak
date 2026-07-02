---
title: "The portable context-contract schema — materialized-view-over-lossless-history as a domain-free admission check any memory/cache can satisfy"
description: "The engine-free context-contract schema that lifts fak's materialized-view-over-lossless-history spine (internal/memview MemoryViewRecord, internal/ctxplan demand-paging, internal/vdso serve-locally, the vToolcall design) into a domain-free contract another agent's memory or cache can satisfy, and the `dos context-contract` verb implements. The check answers one question over a declared ContextView: is what a reader sees a WITNESSED FOLD of an append-only, content-addressed log — or a fabrication? It returns allow | fault | deny(reason) | quarantine, the reason from a closed vocabulary. The view binds to its source by digest + byte span; a changed source digest refutes it; a missing source span is a demand-page FAULT (recoverable), never a lost fact; a body that does not re-fold from the current source is a fabrication (UNWITNESSED_FOLD). Three contracts make it portable: (1) it round-trips author -> check -> review as data with no model and no fak engine in the loop; (2) the view-kind set, the taint lattice, the invalidation rule, and the deny-reason set are CLOSED, validatable vocabularies — an out-of-set token is refused at the boundary, never admitted silently; (3) it is fail-closed and evidence-bound — an absent source observation faults rather than passes, a changed digest refutes, and the taint is a kernel-authored label inherited from the source, never the model's self-report. The contract pays on BOTH lenses: it is the safety story (no laundered/fabricated context) and the optimization story (a planned, demand-paged working set). The honest fence: the serve-from-cache-as-if-the-tool-ran mechanism (vToolcall) is real only where fak DISPATCHES the tool; on the /v1/messages proxy the harness runs tools, so that reach is fenced in docs/notes/VTOOLCALL-MATERIALIZED-VIEW-2026-06-25.md. The `dos context-contract` verb is not yet."
---

# The portable context-contract schema

A memory system that *summarizes to save space* throws facts away. When the reader
needs a span that the summarizer dropped, the fact is gone — there is no channel back
to the original bytes, so the system either fabricates a plausible filler or silently
serves a lossy paraphrase as if it were the source. Both are corruption: the model
reasons over content that no longer equals what the log actually holds.

fak's working set is built the opposite way. The store is **lossless and
content-addressed**; the resident set the reader sees is a planned, O(1)
**view** of it ([`internal/ctxplan`](https://github.com/anthony-chaudhary/fak/tree/main/internal/ctxplan)); a repeated read is served
locally with no round-trip ([`internal/vdso`](https://github.com/anthony-chaudhary/fak/tree/main/internal/vdso)); and the typed view
contract — a derived projection that binds to its source by digest + byte span and is
*never canonical itself* — is [`internal/memview`](https://github.com/anthony-chaudhary/fak/tree/main/internal/memview)'s
`MemoryViewRecord`. The honesty property that falls out: **a dropped span is paged-out,
not summarized away, and pages back in on a forecast miss.** That is exactly what lets
fak report exact recall with a far smaller resident set — measured at a **13.3× smaller
resident set with 100% of misses served (0 refused, 0 lost — every miss a recoverable
page fault) and exact recall preserved**, where line-by-line compaction lost facts on
2,016 of 2,055 turns ([CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md), the planned-view line).

The grammar gap this page closes: that logic lives **inside fak's packages**. There was
no standalone contract another memory or cache system could satisfy to make the same
claim — *what a reader sees is a scope-redacted view of an append-only log; a miss is a
demand-page fault, never a lost fact.* This page is that contract, written domain-free:
the view shape, the four observations that decide it, and the closed disposition
vocabulary. It is `G4` of the
[agent-programming-grammar epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md),
the materialized-view sibling of [the portable taint-check schema](taint-check-schema.md)
(`G7`, an IFC admission check), [the portable agent-routing schema](agent-routing-schema.md)
(`G8`, a routing decision), and [net-true-value](net-true-value.md) (`G9`, a value claim).
The verb that walks this contract over a view is `dos context-contract`; its home is the
installed DOS package, and it is **not yet** (see the [honest fences](#honest-fences)).
fak's [`internal/memview`](https://github.com/anthony-chaudhary/fak/tree/main/internal/memview) is the reference implementation — the
offline witness that the fold below is real, not a wish.

## The schema

A context-contract is a small set of nouns. None of them mentions a tool, a model, or a
fak package — the schema is the *decision* shape, and the decision is taken over digests,
never over the view's bytes.

### ContextView — the declared view and the observed source

```
ContextView {
  view_kind         : ViewKind          // CLOSED: "snippet" (lossless) | "summary" | "qa" | "fact" (lossy)
  producer          : string            // DECLARED: selector/generator identity + version
  source            : SourceSpan        // DECLARED: digest-bound [offset, offset+length) into the log
  source_taint      : Taint             // INHERITED from the source: "trusted" < "tainted" < "quarantined"
  invalidation      : InvalidationRule  // CLOSED: "digest" (the only shipped rule; default)
  body_digest       : Digest            // DECLARED: content address of what the reader SEES
  source_digest_now : Digest            // OBSERVED: the source span's CURRENT digest; ABSENT => paged out
  refold_digest     : Digest            // OBSERVED: digest of re-folding the CURRENT source over the span
}
```

The split is the whole point. The **declared** half (`view_kind`, `producer`, `source`,
`source_taint`, `body_digest`) binds the view to exact source bytes — a different
producer *or* a different span is a different view, so a selector mutation surfaces as a
visible change, never a silent rewrite ([`memview.MemoryViewRecord`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memview/memview.go),
the `#904` selection-integrity property). The **observed** half (`source_digest_now`,
`refold_digest`) is read from the **live append-only log**, never the model's assertion —
the evidence-bound contract. `source_digest_now` *absent* means the span is not in the
resident set (paged out); `refold_digest` is the digest of the bytes obtained by
re-folding the current source over the span.

`source_taint` is the **kernel-authored** label inherited from the source page
([`memview.RawPage.Taint`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memview/memview.go)), the closed
[`abi.TaintLabel`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/types.go) lattice (`trusted` < `tainted` <
`quarantined`) — never a caller's self-tag. A view derived from a tainted or quarantined
source may never enter context ([`memview.VerdictFor`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memview/memview.go)).

`view_kind` is closed to the `memview.ViewKind` minimum. Only `snippet` is **lossless**:
its bytes ARE `source[offset:offset+length]`, so it can be reconstructed and witnessed.
`summary` / `qa` / `fact` are **lossy** derived projections whose body is not a sub-slice
of the source; they are admissible only as derived views that must re-adjudicate before
backing an effect, and can never be certified as a reconstructible fold.

### Decision — the reviewable allow | fault | deny | quarantine

```
Decision {
  view     : ContextView   // the echoed input
  decision : "allow" | "fault" | "deny" | "quarantine"
  reason   : DenyReason    // REQUIRED on anything but "allow"; from the closed set below
  witness  : string        // a bounded, payload-free note (the two digests / the disposition, never the bytes)
}
```

Checking a `ContextView` yields a `Decision`: the echoed view, a verdict, and — on
anything but `allow` — a closed reason. The decision is data: reviewable, diffable,
produced with no model in the loop. `allow` is a witnessed fold of the current source
(serve it). **`fault` is a MISS** — demand-page the span back from the log
([`ctxplan`](https://github.com/anthony-chaudhary/fak/tree/main/internal/ctxplan) `DemandPage`); it is **not** a deny and **not** a
loss, the bytes still live in the append-only log. `deny` is refuted or fabricated.
`quarantine` is a tainted source that may not back a view.

### The deny/disposition reason vocabulary

`DenyReason` is **closed and additive** — a new reason is a new named value plus a
decision arm, never a free-text field. Every token maps to a kernel-witnessed property
of `memview` admission + `ctxplan` demand-paging:

| `reason` | when | kernel property it lifts |
|---|---|---|
| `UNKNOWN_VIEW_KIND` | `view_kind` not in the closed set — fail-closed | `memview.ViewKind` minimum |
| `UNKNOWN_INVALIDATION` | `invalidation` not in the closed set — treated as stale | [`memview.IsValid`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memview/memview.go) unknown-rule arm |
| `EMPTY_SPAN` | `source.length == 0` — a view can't be accountable to bytes it doesn't span | [`memview.ErrEmptySpan`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memview/memview.go) |
| `SOURCE_PAGED_OUT` | `source_digest_now` absent — the span is not resident | `ctxplan` demand-page fault (a MISS, recoverable) |
| `STALE_SOURCE` | `source_digest_now` present but `!= source.digest` — the source changed | `memview.IsValid` digest-rule refutation |
| `QUARANTINED_SOURCE` | `source_taint` is `tainted`/`quarantined` — a tainted source can't back a view | `memview.VerdictFor` → `Quarantine` |
| `LOSSY_NOT_RECONSTRUCTIBLE` | a lossy `summary`/`qa`/`fact` presented as a reconstructible fold | `memview` "lossy is never canonical" |
| `UNWITNESSED_FOLD` | a lossless `snippet` whose `body_digest != refold_digest` — a fabrication | the digest-binding identity |

A token outside this set is **not** silently coerced to `allow`. A malformed view fails
at the authoring boundary (the closed enums reject it) or returns a fail-closed
disposition at check time — never a quiet pass.

### The decision table — domain-free, deterministic, fail-closed

The whole check is a pure function of the view shape and the two observed digests. Read
top to bottom; the first matching arm wins; every non-`allow` arm is recoverable
(`fault`) or refusing (`deny`/`quarantine`), never a silent serve:

```
1. view_kind   ∉ {snippet, summary, qa, fact}         -> Deny(UNKNOWN_VIEW_KIND)
2. invalidation ∉ {digest}                             -> Deny(UNKNOWN_INVALIDATION)   (unknown rule => stale)
3. source.length == 0                                  -> Deny(EMPTY_SPAN)
4. source_taint ∈ {tainted, quarantined}              -> Quarantine(QUARANTINED_SOURCE)
5. source_digest_now is ABSENT                         -> Fault(SOURCE_PAGED_OUT)        (a MISS: demand-page it)
6. source_digest_now != source.digest                 -> Deny(STALE_SOURCE)             (refuted: re-fold)
7. view_kind != snippet                                -> Deny(LOSSY_NOT_RECONSTRUCTIBLE)
8. body_digest != refold_digest                        -> Deny(UNWITNESSED_FOLD)         (fabrication)
9. otherwise                                           -> Allow                          (a witnessed fold of the current source)
```

It is **monotone** over the taint lattice: making a source *more* restrictive
(`trusted`→`tainted`→`quarantined`) never flips a refusal to an allow — the same property
[`memview.VerdictFor`](https://github.com/anthony-chaudhary/fak/blob/main/internal/memview/memview.go) keeps. And it is **fail-closed
in the recoverable direction**: the one arm that is not a refusal (step 5) returns a
*fault* the caller pages back, so an absent observation costs one demand-page, never a
fabricated fill.

## The three contracts (the acceptance, made checkable)

This check is portable because it holds three properties an external memory/cache can
verify without fak's kernel:

1. **It round-trips with no engine.** author (declare a `ContextView`) → check (apply the
   table → a `Decision`) → review (read the `Decision` as data) is a pure data transform.
   No model runs; no network is touched. The [round-trip below](#the-round-trip-as-data)
   is the witness, on disk as fixtures — the same declared view is `allow` when resident +
   fresh + re-folding, `fault` when its source span is paged out, `deny(STALE_SOURCE)`
   when the source digest moved, and `deny(UNWITNESSED_FOLD)` when the body does not
   re-fold — the verdict turning on the digests, never on the bytes.
2. **The vocabularies are closed and validatable.** The view-kind set, the taint lattice,
   the invalidation rule, and the deny-reason set are finite enums; a validator decides
   membership with a finite switch, not a lookup against a live service. The schema is
   published as a machine-checkable JSON Schema —
   [`context-contract-schema.json`](context-contract-schema.json) (Draft 2020-12) — so any
   runtime authors and validates a view with an off-the-shelf validator, **no fak engine
   present**: the [positive fixtures below](#the-round-trip-as-data) validate against it,
   and five on-disk [negative fixtures](fixtures/context-contract-invalid/) — an
   out-of-set view kind, an out-of-lattice taint, a zero-length span, an unknown field (a
   smuggled model-authored `summary`), and a `deny` with no closed reason — are each
   rejected at the boundary. The
   [validation recipe](fixtures/context-contract-invalid/README.md) runs the whole
   round-trip (eight positives accepted, five negatives rejected) with a stock Draft
   2020-12 validator, so the "validatable" claim is checkable, not asserted.
3. **It is fail-closed and evidence-bound.** Absence of an affirmative allow is never a
   silent serve: an absent source observation *faults* (recoverable), an unknown taint or
   rule is treated as stale, a tainted source quarantines, a non-re-folding body is a
   fabrication. And the digests are **kernel-observed** — read from the live append-only
   log, never the model's assertion. A runtime that lets the model author
   `source_digest_now` or `refold_digest` has voided the contract; the check assumes the
   observed half is the kernel's reading of the store.

## The round-trip, as data

The schema's whole claim is that author → check → review is data, not narration. The
fixtures under [`fixtures/`](fixtures/) are the on-disk witness — one declared view per
disposition, each paired with the `Decision` a check with no model and no fak engine
produces:

- [`context-view-resident.json`](fixtures/context-view-resident.json) →
  [`context-decision-resident.json`](fixtures/context-decision-resident.json) — the
  **allow** case: a `snippet` over a `trusted` source that is resident
  (`source_digest_now == source.digest`) and re-folds exactly
  (`refold_digest == body_digest`) — a witnessed fold of its current source.
- [`context-view-fault.json`](fixtures/context-view-fault.json) →
  [`context-decision-fault.json`](fixtures/context-decision-fault.json) — the **fault**
  case: the SAME view, but `source_digest_now` is absent — the span was elided from the
  resident set, so the check returns a demand-page `fault` (`SOURCE_PAGED_OUT`), never a
  loss. This is the optimization lens: a planned working set may safely drop this span
  *precisely because* a miss is recoverable.
- [`context-view-stale.json`](fixtures/context-view-stale.json) →
  [`context-decision-stale.json`](fixtures/context-decision-stale.json) — the **deny**
  case: the source was edited after the fold, so `source_digest_now != source.digest`;
  the invalidation contract fires → `deny(STALE_SOURCE)`. The view is not served stale and
  is not silently refreshed; the caller re-folds.
- [`context-view-fabrication.json`](fixtures/context-view-fabrication.json) →
  [`context-decision-fabrication.json`](fixtures/context-decision-fabrication.json) — the
  **fabrication** case: the span is resident and fresh, but `body_digest != refold_digest`;
  what the reader sees is not the re-fold of the current source → `deny(UNWITNESSED_FOLD)`.
  This is the safety lens: a laundered/fabricated body cannot pass as a fold of the log.

A reviewer reads the files and the table binding them; no model and no fak engine are
needed to confirm the check. fak's [`internal/memview`](https://github.com/anthony-chaudhary/fak/tree/main/internal/memview) binds the
same digests, inherits the same taint, and reaches the same dispositions — offline,
witnessed by `go test ./internal/memview`, the reference implementation's proof that the
fold below is real.

## Reference implementation and witness

| Schema element | Reference stick (`internal/memview`, `internal/ctxplan`, `internal/vdso`) | Status |
|---|---|---|
| `ViewKind` (`snippet` lossless; `summary`/`qa`/`fact` lossy) | `memview.ViewKind` + `KindSnippet` (the only auto-materialized, lossless view) | [SHIPPED] |
| `SourceSpan` (digest-bound `[offset, offset+length)`) | `memview.SourceSpan` + `MaterializeSnippet` (span out of range / empty refused) | [SHIPPED] |
| `InvalidationRule` (`digest`; unknown ⇒ stale) | `memview.InvalidateOnDigestChange` + `memview.IsValid` (fail-closed on unknown/empty) | [SHIPPED] |
| Inherited kernel-authored `Taint`; tainted source can't back a view | `memview.RawPage.Taint` + `memview.VerdictFor` (→ `Quarantine`) | [SHIPPED] |
| `Digest` (sha256 hex, content address) | `memview.Digest` (same scheme as `internal/recall.Digest` + the blob store) | [SHIPPED] |
| A miss is a demand-page fault, never a lost fact | `internal/ctxplan` planned resident view + demand-page (CLAIMS.md: 13.3×, 100% served) | [SHIPPED] |
| A repeated read served locally (serve-as-if-it-ran) | `internal/vdso` FastPath `Lookup` (kernel-dispatched only) | [SHIPPED] |
| Offline determinism witness | `go test ./internal/memview` (no model in the loop) | [SHIPPED] |
| Portable `dos context-contract <view>` verb | the installed DOS package | **not yet** |

## Honest fences

- **This contract certifies the FOLD; serve-from-cache is fenced elsewhere.** The schema
  decides whether a view is a witnessed fold of its source — the *invalidation* and
  *no-fabrication* properties. The closed-loop "serve a tool result as if the tool ran"
  mechanism (vToolcall) is a **separate** reach, and it is real **only where fak
  dispatches the tool** (the `fak_syscall` / in-kernel path, where the kernel sees its own
  writes). On the `/v1/messages` provider-forward proxy the harness runs the tools, so
  fak cannot serve a synthesized result there at all — it can only *recognize* a redundant
  one for compaction. That boundary is grounded line-by-line in
  [the vToolcall materialized-view note](../notes/VTOOLCALL-MATERIALIZED-VIEW-2026-06-25.md);
  this page does not re-open it.
- **The contract is data; declaring one introduces no spontaneous refusal.** A
  `ContextView` is a binding, not a runtime gate — authoring one fires no check until a
  verb walks it at an opt-in surface. (The grammar's
  [data-not-code rule](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md): the
  mechanism stays in the installed `dos` package; only policy crosses into the tree.)
- **A lossy view is admissible, but never as a reconstructible fold.** Only a `snippet` is
  lossless and witnessable by re-fold. A `summary`/`qa`/`fact` body is a real, useful
  derived projection — but presenting it as a reconstructible fold of the source is
  `LOSSY_NOT_RECONSTRUCTIBLE`. The contract does not forbid lossy views; it forbids
  *laundering* one as a witnessed fold.
- **The observed half must be the kernel's reading.** The check assumes
  `source_digest_now` and `refold_digest` are read from the live log, not asserted by the
  caller. The schema can carry only digests in the right shape; it cannot itself prove
  they were kernel-observed — that proof is the calling runtime's obligation, the same way
  [the taint-check fence](taint-check-schema.md#honest-fences) requires a kernel-authored
  taint.
- **The `dos context-contract` verb is not yet.** The portable verb that walks this schema
  over a view lives in the installed DOS package (the grammar's home; see the
  [epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)). This page is the
  contract it implements; the verb is a named follow-on, not shipped here.

## Cross-references

- [`context-contract-schema.json`](context-contract-schema.json) — the machine-checkable JSON Schema (Draft 2020-12) for this contract: author and validate a view with any off-the-shelf validator, no fak engine present.
- [`fixtures/context-contract-invalid/README.md`](fixtures/context-contract-invalid/README.md) — the engine-free validation recipe (eight positives accepted, five negatives rejected) that makes contracts (2) and (3) checkable.
- [`internal/memview`](https://github.com/anthony-chaudhary/fak/tree/main/internal/memview) — fak's reference implementation: the typed `MemoryViewRecord`, the digest-bound `SourceSpan`, `IsValid` (digest invalidation), and `VerdictFor` (tainted-source refusal).
- [`internal/ctxplan`](https://github.com/anthony-chaudhary/fak/tree/main/internal/ctxplan) · [`internal/vdso`](https://github.com/anthony-chaudhary/fak/tree/main/internal/vdso) — the planned demand-paged resident set and the serve-locally FastPath the optimization lens rests on.
- [The vToolcall materialized-view note](../notes/VTOOLCALL-MATERIALIZED-VIEW-2026-06-25.md) — the design + honest fence for serving a result as if the tool ran, and exactly which surfaces can.
- [The portable taint-check schema](taint-check-schema.md) (`G7`) · [the portable agent-routing schema](agent-routing-schema.md) (`G8`) · [net-true-value](net-true-value.md) (`G9`) — the sibling grammar standards; same recipe, different verb.
- [The agent-programming grammar](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) — the epic this schema is `G4` of, and the recipe every lift keeps (closed vocabulary, evidence-bound, fail-closed, data-not-code, pays on both lenses).
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — shipped vs stub, claim by claim (the planned-view line: 13.3× smaller resident set, 100% of misses served, exact recall preserved).
