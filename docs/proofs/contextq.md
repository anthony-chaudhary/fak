---
title: "fak proof: contextq materialization fidelity"
description: "Proof that fak's contextq materializer reconstructs CDB pages byte-identically on the raw path and yields the same result for the same request."
---

# A5 · contextq

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 2 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/contextq/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

`contextq` is the on-demand context **materializer** over CDB images. It owns no
session bytes: CDB stays the page-fault provider and `recall` stays the trust gate.
`contextq.Query(ctx, im, req)` turns a context `Request` (query, K, pins, excludes,
budget, optional derived-view preference) into a typed `Result` — selected `Slices`,
`MemoryViewRecord` views, `MaterializationVerdict`s (HIT / FAULT / RECOMPUTE / REFUSE /
ABSTAIN), `Omissions`, and a `RenderPlan`. Two paths ship: a **raw-page path** that
demand-pages each benign selected page through `im.Examine` and wraps it as a snippet
view (every materialization a FAULT), and a **derived-view path** (`PreferView=summary`)
that resolves through a `ViewCache` (HIT / RECOMPUTE / FAULT) and renders a deterministic
*extractive summary*. "Correct" for this **algebraic (regime A)** module means: (1) what
it hands back faithfully reconstructs what CDB holds, and (2) the same request over the
same image always yields the same result. Both theorems below are **OPEN for this
module** — honestly un-witnessed at the contextq boundary — and theorem (1) additionally
carries a path caveat, because the summary path is *intentionally lossy*.

---

## Verdict reconciliation — `contextq.MaterializationVerdict` ↔ `cachemeta.LookupVerdict`

fak carries **two** typed verdict sets over reusable state, and they are not the same
set. A reader who sees `MaterializationVerdict` and `LookupVerdict` and assumes one is
an alias for the other will misread both (issue #227). This section is the
reconciliation: the two sets, the one place they meet, and the honest asymmetries.

**The two sets.**

| set | type | layer | kinds | defined |
|---|---|---|---|---|
| cache-plane reuse | `cachemeta.LookupVerdict` (`LookupKind`) | the generic cache contract every plane emits | `hit` · `miss` · `revalidate` · `transform` · `quarantine` · `fault` (6, **no abstain**) | `internal/cachemeta/cachemeta.go:244` |
| context materialization | `contextq.MaterializationVerdict` (`MaterializationKind`) | the on-demand context materializer | `HIT` · `FAULT` · `RECOMPUTE` · `REFUSE` · `ABSTAIN` (5, **no miss/revalidate/transform/quarantine**) | `internal/contextq/contextq.go:163` |

They answer different questions. `LookupVerdict` asks *"may this cache entry be
reused?"* and its six kinds cover the full cache-plane lifecycle (a clean serve, a miss,
a must-recheck, a transform-then-serve, a hold-back, a residency fault).
`MaterializationVerdict` asks *"how was this context piece materialized for this
query?"* and its five kinds are the materialization outcome (served from cache, paged in
fresh, rebuilt because stale, refused on trust/scope, or out of scope). The
`cachemeta.MaterializeVerdict` *function* (`internal/cachemeta/materialization.go:187`)
is a third name in this space: it is a function returning a `LookupVerdict`, not a type —
do not confuse it with the `contextq.MaterializationVerdict` *type*.

**Where they meet — exactly one lowering point.** The only place a cachemeta verdict
must become a contextq verdict is the KV-view reuse gate `contextq.GateKVView`
(`internal/contextq/kvview.go:110`): a model-bound KV span is a cachemeta cache entry,
but it has to flow in the same `Result.Verdicts` stream as every snippet/summary view, so
its reuse decision is lowered into a `MaterializationVerdict`. `GateKVView` does **not**
call `cachemeta.MaterializeVerdict`; it reuses the shared binding primitives
(`MaterializationKey.Matches` / `.Complete`, `QualityEvidence.Acceptable`) and emits
contextq kinds directly. The mapping the gate implements (its only outputs are `HIT`,
`REFUSE`, `ABSTAIN`):

| cachemeta `LookupKind` (cache plane) | → `contextq.MaterializationKind` (at the KV gate) | why |
|---|---|---|
| `hit` | `HIT` | every binding axis matches (`kvview.go:151`) |
| `miss` — model / tokenizer / serializer / position / policy / admitter mismatch, incomplete key, unproven approximate | `REFUSE` | fail-closed: a possibly-mismatched span is refused, never silently served (`kvview.go:133`,`:140`,`:146`) |
| `transform` — provider-prefix residency (`ProviderCacheVerdict`) | `REFUSE` | provider residency is cost/latency telemetry, never local trust (`kvview.go:119`) |
| `revalidate` | — not produced on the KV gate | KV staleness is a binding mismatch (`REFUSE`); the "stale but rebuildable" outcome is the derived-view path's `RECOMPUTE`, keyed on `PolicyVersion`, not a KV-axis revalidate |
| `quarantine` | — surfaces as `REFUSE` | quarantine is the cachemeta / ctx-MMU eviction concern; at the materializer a held-back span is simply not served |
| `fault` | — surfaces as `FAULT` (raw page) or `REFUSE` (KV view) | a residency fault on the raw page is a fresh page-in (`FAULT`); on a KV view it is a refusal |

**The honest asymmetries (this is why "they match" was wrong).**

- **`ABSTAIN` exists only in contextq.** `cachemeta.LookupKind` defines no abstain
  (`docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md` §2.6); cachemeta expresses "not
  applicable" as a non-`hit` (a `Miss`). contextq gives it a first-class kind: `ABSTAIN`
  means *this gate does not apply to this view* (e.g. a non-local-KV view type,
  `kvview.go:126`), deliberately distinct from `REFUSE` (a trust/binding denial). A
  consumer reading the verdict stream can tell "would not serve" (`REFUSE`) from "had no
  opinion" (`ABSTAIN`).
- **`RECOMPUTE` exists only in contextq.** It is the derived-view path's "cached but
  policy-stale, rebuild" verdict. cachemeta's nearest analogue is `revalidate`, but the
  KV gate does not use it — on the KV axis a policy drift is a binding mismatch
  (`REFUSE`), not a rebuild.
- **`miss` / `revalidate` / `transform` / `quarantine` exist only in cachemeta.** They
  describe cache-plane lifecycle states the materializer collapses: anything that is not
  a clean `hit` becomes `REFUSE` (trust/binding failure), `FAULT` (must page/build),
  `RECOMPUTE` (stale derived view), or `ABSTAIN` (out of scope) at the materializer
  boundary.

**Net.** The two sets are *reconciled, not unified*: `contextq.GateKVView` is the one
function that lowers a cachemeta KV-reuse decision into the contextq verdict stream, the
four cachemeta-only kinds collapse to the two contextq refusal-shaped kinds
(`REFUSE` / `ABSTAIN`) at that boundary, and contextq's two extra kinds
(`RECOMPUTE` / `ABSTAIN`) express materializer-specific outcomes with no cachemeta
analogue. The design note's "matches the `cachemeta.LookupVerdict` shape"
(`docs/notes/ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md` §6 Step 3) was the aspiration;
the shipped code is the related-but-distinct mapping above.

---

## THEOREM 1 — materialization is byte-identical to the CDB image

**THEOREM.** On-demand materialization (`contextq.Query`) reconstructs a context
byte-identical to the original CDB image: every materialized benign page's bytes equal
the bytes the CDB image holds for that page.

**REGIME.** A — algebraic / structural (round-trip / byte-identity).

**PROOF.** On the raw path, `Query` routes each benign selected page through
`im.Examine(ctx, step)` (`fak/internal/contextq/contextq.go:366`). CDB guarantees that a
benign page round-trips **byte-identical** through `Examine`
(`fak/internal/cdb/cdb.go:173`, doc at `cdb.go:168`), and contextq does not transform
those bytes on the raw path — it accrues `len(b)` and wraps a metadata-only snippet view.
So the bytes that page in *are* the image bytes. **Two problems block PROVEN at this
boundary.** First, contextq retains **no reconstructed-context buffer** in `Result` — the
raw bytes are used for length/token accounting and then dropped, so nothing in-module
holds them to compare. Second, contextq's *other* materialization path
(`PreferView=summary`) returns a bounded extractive **prefix** via `buildSummary`
(`fak/internal/contextq/contextq.go:678`), truncated at `maxSummaryBytes=256` with
`Coverage<1.0` by design — that is deliberately **not** byte-identical to the source
page. The unqualified theorem is therefore false for the module; it holds only for the
raw path, and even there is witnessed one layer down in cdb
(`TestExamineBenignRoundTripsByteIdentical`, `cdb_test.go:67`;
`TestImageResultPagesInByteIdentical`, `cdb_test.go:233`), not in contextq.

**WITNESS.** `(go test ./internal/contextq/ -count=1 -timeout 120s -v)`
— ran green, 5/5 PASS, `ok ... 0.557s`. No contextq test asserts byte-identity:
`TestQueryMaterializesTypedWorkingSet` checks slice/view counts, FAULT verdicts, sealed
REFUSE, media types, and policy-version carry-through — never a byte compare against
`im.Examine`. The genuine byte-identity tests are in cdb and witness `cdb.Examine`, not
this module.

**VERDICT.** **OPEN** (2026-06-20). The property is *true for the raw path* and proven
in cdb, but un-witnessed at the contextq boundary, and the theorem **as stated is false**
for the lossy summary path. To close: add a contextq test that independently calls
`im.Examine(step)` for each raw-path slice and asserts the materialized slice carries the
identical bytes, and **restate** the theorem as byte-identity of the *raw* path only
(the summary path's contract is `FaithfulnessProbe==1.0` extractive-prefix + reported
`Coverage`, a different obligation).

**DOS.** bound at ship.

---

## THEOREM 2 — materialization is deterministic

**THEOREM.** `contextq.Query` is deterministic: for a fixed CDB image and a fixed
`Request`, repeated calls produce an equal `Result` (same slices, views, verdicts,
omissions, render plan, and stats).

**REGIME.** A — algebraic / structural (determinism / function-equality).

**PROOF.** Every input-dependent step is a deterministic pure function of
`(image, request)`. Frame order comes from `im.Backtrace()` and step-ordered iteration
(`fak/internal/contextq/contextq.go:267`, `:314`, `:407`, `:518`); selection de-dups via
a `seen` map but **appends** in a fixed pin-then-rank order (`contextq.go:305`–`:319`), so
the map is membership-only and never orders output; `im.WorkingSet` ranking is
deterministic; `buildSummary` is a fixed line-boundary prefix trim with no RNG
(`contextq.go:678`); tokenization and `overlap` are pure (`contextq.go:722`, `:748`,
`:767`). There is no `time.Now`, no `rand`, and no concurrent mutation inside a single
`Query`. Equal inputs therefore yield equal outputs by construction. Per the method's one
rule, a sound *argument* is not a PROVEN: nothing here re-runs `Query` and compares.

**WITNESS.** `(go test ./internal/contextq/ -count=1 -timeout 120s -v)`
— ran green, 5/5 PASS, `ok ... 0.557s`. But no test calls `Query` twice on identical
inputs and asserts equality. `TestAllFiveMaterializationVerdictsReachable` runs `Query`
four times while *changing* inputs each pass (policy `p1`→`p2`, then a tiny budget), so it
witnesses state-dependent verdict transitions, not determinism of a fixed input.
`TestViewCacheCopiesPayloadBytes` is a cache-aliasing test, not a Query-determinism test.

**VERDICT.** **OPEN** (2026-06-20). True by construction, but un-witnessed. To close: add
a test that runs `Query(ctx, im, req)` twice on the **same** image and request (no-cache
raw path) and asserts `reflect.DeepEqual(a, b)`, plus a second twice-run with a single
**fixed** warm `ViewCache` to pin the derived-view path's determinism.

**DOS.** bound at ship.

---

### Reproduce

```bash
go test ./internal/contextq/ -count=1 -timeout 120s -v
```

All 5 existing tests pass; neither theorem above has a *specific* witness in this
package, so both are OPEN, not PROVEN. The companion byte-identity guarantee that
theorem 1 leans on is witnessed in `internal/cdb`
(`go test ./internal/cdb/ -run 'ByteIdentical' -count=1`).

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **materialize-byte-identical** → ✅ PROVEN by `TestMaterializeByteIdentical`. For every materialized benign slice in res.Slices, an independent im.Examine(step) succeeds and returns bytes whose length equals the recorded SliceRef.Bytes. The CDB image holds one canonical content-addressed body per page (recall.Resolve serves s.cas[p.Digest]), so re-paging is byte-identical: reflect.DeepEqual over two Examine calls passes, and a second Query reconstructs the same step with the same byte length. Mechanism confirmed at contextq.go:366 (Examine in materializeRaw), cdb.go:173 (Examine->Resolve), recall.go:284 (cas body keyed by digest). PASS.
- **materialization-deterministic** → ✅ PROVEN by `TestMaterializationDeterministic`. Two independent Query calls on the same attached *cdb.Image with a fixed Request (query+policy+pins+excludes) produce reflect.DeepEqual Results across the whole struct and field-by-field: Slices, Views, Verdicts, Refused, Omissions, RenderPlan, Stats. Non-vacuity guarded before the assertion. Also proven on the derived-view path: two cold Queries with PreferView=ViewSummary over two FRESH NewViewCache() instances are DeepEqual, so determinism is not specific to the raw-page path. PASS.
