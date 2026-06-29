# Context-safety visual-primitive schemas (#1223, C6 of epic #1217)

_Research / design only. This is the **C6 visual-primitive schema spec** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs the **data schema of each of the six visual primitives** — what each
consumes, what cross-check each carries, and the shared data envelope they all
ride — reusing the in-tree scorecard-payload and HTML-render precedents rather
than inventing a new rendering stack. No code ships here — the deliverable is
this committed schema set. It schemas the doctrine mechanisms A–D of the C1 note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
and renders the failures cataloged in C2
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
The rendering *home* (which panel lives where in the debug window) is child C11
([#1229](https://github.com/anthony-chaudhary/fak/issues/1229)); this note owns
the *data*, not the placement._

---

## Design rule — schema the data, reuse the render

Every primitive is a **roll-up or time-series, never a single point** (#1217),
and every primitive ships **with its cross-check**. So each schema below names
three things: what it **consumes** (the witnessed source), its **shape**
(time-series / roll-up / heatmap), and the **cross-check** (A paired-axis / B
conservation / C provenance lane / D re-derivation) it carries. None invents a
new renderer — they reuse three in-tree precedents:

- **The data envelope** — the scorecard-payload precedent
  `internal/covmatrix` (`covmatrix.go:33` `const Schema = "fak-coverage-matrix/1"`):
  a versioned, JSON-serialized payload with a schema tag, so a consumer can
  validate the shape and a checker (C9) can re-derive from it.
- **The text/emoji fallback** — `internal/scoreboard/render.go:41`
  `FromPayload(title string, p scorecard.Payload, debtKey string) Update`: the
  grade-fold that turns a scorecard payload into a terminal-renderable `Update`,
  so every primitive degrades to a sourced text line when no HTML surface is
  present.
- **The HTML render** — `internal/cdb/report_html.go:91`
  `(*Image).HTMLReport`: the one existing in-tree HTML report (working-set:
  residency / faults-avoided), the rendering precedent the richer panels follow.

> **Schema-version tag.** Every primitive payload carries
> `Schema = "fak-ctxsafety-<primitive>/1"` (mirroring `fak-coverage-matrix/1`),
> so C9's re-derivation checker binds to a known shape and C10's CI gate can
> detect a schema drift.

---

## The shared envelope every primitive rides

```
CtxSafetyPanel {
  Schema      string         // "fak-ctxsafety-<name>/1"  (covmatrix precedent)
  Primitive   int            // 1..6
  Window      [2]int64       // [fromSeq, toSeq] journal range, or [fromTs, toTs]
  Points      []PanelPoint   // the time-series / roll-up segments (never a single scalar)
  CrossChecks []string       // {"A","B","C","D"} carried by this primitive
  Provenance  string         // dominant lane; each PanelPoint also carries its own
}

PanelPoint {
  At          int64    // journal Seq or timestamp (the time axis)
  Value       float64  // the primary axis
  PairedValue float64  // the paired axis (A) — e.g. realized against saved
  Segment     string   // the conservation segment label (B) — e.g. "signal"/"noise"/"unaccounted"
  Lane        string   // WITNESSED | OBSERVED | MODELED  (C, the provenance channel)
  Rederivable bool     // does this point re-derive from a tamper-evident source? (D, C9)
}
```

A point that is a single scalar with no paired axis, no segment, and no lane is
**not a valid context-safety point** — the envelope structurally forbids the
single-trust-me number the doctrine exists to kill.

---

## The six primitives

### 1 — Context-event ledger ribbon (the *event* half)

- **Consumes:** the hash-chained journal `internal/journal/journal.go`
  `CAP_EVICT` / `CAP_FAULT` / `CAP_VERSION_BIND` rows (one lane per event kind,
  height = tokens affected), plus the secondary tier-transition sources
  `internal/engine/cacheevents.go` `CacheEventRecorder` (`cacheevents.go:304-317`,
  demote/spill/evict normalized into a `cachemeta.Entry` + the `fak_engine_cache_*`
  family) and `internal/cachemeta/kvtransfer.go` `KVTransfer{Direction, Tokens,
  Outcome}` (`kvtransfer.go:34-48`) with `Outcome ∈ {ok, missed, fault}`
  (`kvtransfer.go:24-28`) — the tiered page-down event with a visible outcome,
  ties #985.
- **Shape:** time-series, one lane per event kind.
- **Cross-check:** **A** (event kind paired with outcome — a `missed` restore
  against a recorded `offload` is the gap) + **C** (each row's provenance lane).
- **Renders:** C2 failures F2 (evict-on-recurrent refused) and F6 (tier
  page-down missed).

### 2 — Saved-vs-realized gap chart (the *value* half)

- **Consumes:** the two WITNESSED taps specced in full by child C4
  ([#1221](https://github.com/anthony-chaudhary/fak/issues/1221)) — claimed-saved
  (journal `CAP_EVICT` + `capacity_adapter.go`) vs realized reuse
  (`internal/cachewitness` `KVPrefixWitness.ReuseRatio()` `cachewitness.go:155`
  over `fak_gateway_kv_prefix_reused_tokens_total` `metrics.go:1531`, fed by
  `cacheobs.Default.Snapshot()`).
- **Shape:** paired time-series; the gap is the error bar between the two axes.
- **Cross-check:** **A** (saved with realized) + **D** (both re-derive from
  tamper-evident sources). **Never blended with the OBSERVED provider
  `cache_read`** (#1217 fence b).
- **Renders:** C2 failure F3 (saved-but-never-reused). Full datum schema:
  C4 (#1221).

### 3 — S/N + Faults paired strip

- **Consumes:** `internal/ctxplan/signalnoise.go` — `Ratio()` (`:69`, the
  caching/length-invariant signal axis), `FaultRatio()` (`:81`, the orthogonal
  under-resident axis), `Grade()` (`:249`, lean/ok/bloated/**starving**), over
  `ctxplan.Outcome.{Hits,Wasted,Faults,UnaccountedTokens}`.
- **Shape:** paired time-series (S/N lane + Faults lane).
- **Cross-check:** **A** (S/N with Faults — a raise-S/N-by-starving spikes the
  visible Faults lane) + **B** (`resident = signal + noise + unaccounted` must
  sum; `unaccounted` kept in the denominator, `signalnoise.go:48`) + **D**.
- **Renders:** C2 failure F4 (S/N-up-but-faults-up); the `starving` grade is the
  named failure verdict.

### 4 — Evict-correctness heatmap

- **Consumes:** the G1 bit-identity witness `internal/model/kvcache.go:44-105`
  (`CanEvict`/`TryEvict`/`RecurrentEvictUnsupportedError`), per-evict `max|Δ|`
  proven by `internal/model/paged_evict_test.go:84` and `longrope_test.go:179`;
  **extended** per-cell to the G4 integrity witness `internal/recall`
  `PageSyndrome` / `PageSyndromeFor` (digest re-derivation against the CAS body).
- **Shape:** heatmap (per-evict / per-cell cell, green only at `max|Δ|=0` /
  every syndrome axis clear).
- **Cross-check:** **C** (a REPAIRABLE syndrome is MODELED-recoverable, an
  ERASURE is a hard WITNESSED loss — distinct lanes) + **D** (the number proves
  its own correctness: re-derive the digest from the CAS body, RED on any
  mismatch).
- **Renders:** C2 failures F1-adjacent (evict correctness) and F5 (cell rot).
  G1 surface is child C3 (#1220); G4 is the per-cell extension.

### 5 — Residency conservation bar

- **Consumes:** `internal/ctxresidency` per-span `State`
  (`ctxresidency.go:16-31`, resident / evictable / held) and the per-span
  `EvictBlastRadius BlastRadius{Tokens, DependentEntries}`
  (`ctxresidency.go:39-42`, the cost-of-evicting axis), rolled up by `Snapshot`
  (`ctxresidency.go:73-89`: `ResidentTokens` / `HeldSpans` / `ByteHeld` /
  `ByteCleared` / `PollutionRate`).
- **Shape:** conservation roll-up — `resident = signal + noise + unaccounted`;
  segments must sum to the total.
- **Cross-check:** **B** (the defining conservation sum — if the segments don't
  add up the bar is *visibly* broken, the eye does the check) + **C** + **D**.
- **Renders:** the residency half of F2 (the `refused` evict segment is
  non-zero) and the standing "where did the window go" roll-up.

### 6 — Liveness-contradiction light

- **Consumes:** two heartbeats — the liveness pulse `fak_bgloop_up`
  (`internal/gateway/bgloops.go`) vs the decode-seconds / decode-rate accumulator
  (`internal/gateway/metrics.go`). Reuses the #750 two-heartbeat oracle pattern.
- **Shape:** a single binary light, but driven by a **paired** time-series (the
  two heartbeats) so the contradiction is re-derivable, not asserted.
- **Cross-check:** **A** (`/healthz` up vs decode-throughput zero) + **D**
  (re-derive both signals from their metrics; RED when they disagree).
- **Renders:** C2 failure F1 (the #1143 GDN evict panic: healthy but serving
  nothing). The RED-on-disagree rule is detailed by child C12 (#1230).

---

## Primitive → consumes → shape → cross-check → failure at a glance

| # | Primitive | Consumes (witnessed source) | Shape | Cross-check | Renders (C2) |
|---|---|---|---|---|---|
| 1 | event ribbon | journal `CAP_EVICT`/`CAP_FAULT`; `KVTransfer` ok/missed/fault | time-series, per-kind lane | A + C | F2, F6 |
| 2 | saved-vs-realized gap | `cachewitness.go:155` vs journal claim (C4) | paired time-series | A + D | F3 |
| 3 | S/N + Faults strip | `signalnoise.go:69/:81/:249` | paired time-series | A + B + D | F4 |
| 4 | evict-correctness heatmap | `kvcache.go:44-105` + `recall.PageSyndrome` | heatmap | C + D | F1-adj, F5 |
| 5 | residency conservation bar | `ctxresidency` `State` + `BlastRadius` | conservation roll-up | B + C + D | F2 |
| 6 | liveness light | `fak_bgloop_up` vs decode-rate | paired → binary light | A + D | F1 |

Every primitive carries **at least A + B** of the doctrine's mechanisms (most
carry D); none is a single point; each binds to a real `file:line` source and a
cataloged C2 failure.

---

## Honest `not yet` gaps (per fence c)

- **No in-tree HTML render exists for five of the six primitives.** Only the
  residency-style working-set report (`cdb/report_html.go`) has an HTML
  precedent; the ribbon, gap, S/N-strip, heatmap, and liveness-light have
  **schemas here but no renderer** — the renderer is the build follow-on, homed
  by C11 (#1229). The `scoreboard/render.go` `FromPayload` text fallback is the
  *only* render every primitive can claim today.
- **The schema-version tags `fak-ctxsafety-<name>/1` are proposed, not
  registered.** They mirror `fak-coverage-matrix/1` but do not exist until the
  build epic; C9's checker and C10's CI gate depend on them being real.
- **`ctxresidency` `EvictBlastRadius` as a rendered axis is `not yet`.** The
  `Snapshot` carries the cost-of-evicting data, but no panel renders it;
  primitive 5's conservation bar is the first consumer.

---

## Acceptance check (against #1223)

The issue's acceptance: *spec the six visual primitives' data schemas — what
each consumes, what cross-check each carries — reusing the covmatrix payload
envelope, the scoreboard `FromPayload` fold, and the `cdb/report_html.go` HTML
precedent._

- **Six schemas, each grounded.** Ribbon, gap, S/N-strip, evict-heatmap,
  residency-bar, liveness-light — each names its witnessed source (`file:line`),
  its shape (time-series / roll-up / heatmap — never a point), and its
  cross-check (A/B/C/D).
- **The shared envelope** (`CtxSafetyPanel` / `PanelPoint`) reuses the
  `covmatrix` versioned-payload precedent and structurally forbids a single
  trust-me point (every point carries a paired axis, a segment, a lane, and a
  re-derivability bit).
- **The render reuse is named** — `covmatrix` envelope, `scoreboard/render.go`
  `FromPayload` text fallback, `cdb/report_html.go` HTML precedent — so the
  primitives do not invent a rendering stack.
- **Each primitive maps to its C2 failure** (F1–F6) and its downstream child
  (C3 G1, C4 gap, C11 home, C12 liveness), and the un-rendered primitives are
  named as honest `not yet` gaps, not assumed shipped.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Schemas the
doctrine mechanisms A–D of the C1 note (#1218); renders the C2 catalog failures
(#1219); consumes the C4 gap datum (#1221); homed in the debug window by C11
(#1229); re-derived by C9 (#1227); gated in CI by C10 (#1228). Design-only — no
implementation ships under #1217 until the notes are reviewed._
