# Context-safety failure-mode catalog (#1219, C2 of epic #1217)

_Research / design only. This is the **failure-mode catalog** child of the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It catalogs every failure the context-safety plane must surface, each mapped to
the **witness** that catches it, the visual **primitive** that renders it, and
the **cross-check** that goes red on it. No code ships here — the deliverable is
this committed catalog. It is the sibling of the C1 doctrine note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218), the property +
four guarantees) and feeds the visual-primitive specs (C6, #1219's downstream
children)._

> **The property (from #1217).** The security floor proves a *bad* call can't
> get *in*. The context-safety floor proves a *good* call's value doesn't
> silently get *lost* when fak sheds context (evict / page-out / compact / shed /
> reset) — and shows it as a **cross-checkable picture**, not a number you have
> to trust. A failure that no witness catches and no visual reds on is, by this
> doctrine, *invisible value destruction* — and that is the worst class.

## How to read a row

Each row binds a failure to four things, every one grounded in a real in-tree
seam (no invented mechanism):

- **What goes wrong** — the value-destruction or liveness event itself.
- **Witness** — the *one* journal `Kind`, typed verdict, metric, or test that
  records the event from a tamper-evident or witnessed source. This is the
  G1–G4 guarantee seam from the C1 doctrine.
- **Primitive** — the *one* visual primitive (numbered per #1217's six) that
  renders it as a roll-up or time-series, never a single point.
- **Cross-check** — the *one* check (paired-axis A / conservation roll-up B /
  provenance lane C / re-derivation witness D) that goes **red** when the number
  contradicts its own parts or a re-derivation from source.

The visual primitives (from #1217 §"Visual primitives this epic standardizes"):
**(1)** context-event ledger ribbon, **(2)** saved-vs-realized gap chart,
**(3)** S/N + Faults paired strip, **(4)** evict-correctness heatmap,
**(5)** residency conservation bar, **(6)** liveness-contradiction light.

The provenance lane: **WITNESSED** = fak authored/controls the number;
**OBSERVED** = relayed from an external party; **MODELED** = projected, not yet
measured on-device. Three lanes, never summed (#1217 honesty fence b).

---

## The catalog

| # | Failure | What goes wrong | Where it's observable (journal `Kind` / metric / verdict) | Witness (the seam) | Primitive | Cross-check | Provenance |
|---|---|---|---|---|---|---|---|
| F1 | **GDN evict panic** (#1143) | `Evict` is called on recurrent (Gated-DeltaNet) state on budget eviction → handler dies; `/healthz` still answers **200** but every decode **hangs**. *Looks healthy, serves nothing.* | No journal row at all on the un-recovered path (the panic kills the handler); on the recovered path, the typed verdict + a `CAP_EVICT`/`CAP_FAULT` row. The *contradiction* is observable as `/healthz`=up vs decode-rate=0. | The typed boundary `*model.RecurrentEvictUnsupportedError` (`internal/model/kvcache.go:54`), the handler recovery `recoverRecurrentEvictUnsupported` (`internal/gateway/gateway.go:1216,1266`), and the liveness pulse `fak_bgloop_up` (`internal/gateway/bgloops.go:32,149`) vs the decode-seconds accumulator (`internal/gateway/metrics.go:79`). | **(6) liveness-contradiction light** | **A** (paired-axis: `/healthz` vs decode-throughput) + **D** (re-derive both from `fak_bgloop_up` and the decode-rate metric; RED on disagree). Reuses the #750 two-heartbeat oracle. | WITNESSED |
| F2 | **Evict-on-recurrent refused** | `CanEvict`/`TryEvict` returns `RecurrentEvictUnsupportedError` cleanly (no panic) — but the eviction *did not happen*, so the value axis the operator was promised (room reclaimed / tokens saved) goes to **zero** while the event ribbon shows an "evict" was attempted. The refusal is correct; the *silent zero realized value* is the failure. | The typed verdict surfaced as a `CAP_EVICT` row with the refused outcome (`internal/journal/journal.go:391`), bound to the `serve --reset-on-budget` / `--budget-webhook` directive (`cmd/fak/serve.go:110,113`). | `KVCache.CanEvict` / `TryEvict` (`internal/model/kvcache.go:66,105`) returning the typed error; the conservation segment `evicted = bit-clean + reset + **refused**` (#1217 §B). | **(1) ribbon** (event = evict, lane outcome = **refused**) + **(5) residency conservation bar** (the `refused` segment is non-zero) | **B** (conservation: the `refused` segment must sum into `evicted`; a refused evict that claims realized value breaks the sum) + **A** (saved-vs-realized: realized = 0). | WITNESSED |
| F3 | **Saved-but-never-reused** | A page-out *claimed* it saved N tokens at event time; the saved bytes are never reused on a later turn. The claim was a forecast; the realized reuse never bit. **The gap is the error.** | The claim at event time vs the WITNESSED realized reuse: `fak_gateway_kv_prefix_reused_tokens_total` (`internal/gateway/metrics.go:1531`) and the `ReuseRatio` (`:1538`), folded by `internal/cachewitness` `KVPrefixWitness.ReuseRatio()` (`cachewitness.go:155`). | `internal/cachewitness` (the WITNESSED kv_prefix family, `cachewitness.go:16,56`) + `cacheobs.Default.Snapshot()`. Two **witnessed** taps: claimed-saved at event time, later-realized reuse. | **(2) saved-vs-realized gap chart** | **A** (the two paired axes ARE saved and realized) + **D** (both re-derived from the two witnessed metric taps; RED when claimed-saved has no realized backing). Never blended with the **OBSERVED** provider `cache_read` (#1217 fence b). | WITNESSED (fak's RadixAttention); provider `cache_read` stays OBSERVED |
| F4 | **S/N-up-but-faults-up** | A shed (compact / trim / evict) *raised* the signal-to-noise ratio — but it did so by **starving the turn**: it elided spans the turn then demand-paged back (induced faults). Value destruction wearing a cleanup costume. A single-axis "S/N improved" visual is a **lie by construction** here. | `internal/ctxplan/signalnoise.go`: `Ratio()` (`:69`, the caching/length-invariant signal axis) paired with `FaultRatio()` (`:81`, the orthogonal under-resident axis), graded by `Grade()` (`:249`, lean/ok/bloated/**starving**), over `ctxplan.Outcome.{Hits,Wasted,Faults}`. | The two orthogonal axes already exist as a pure function of `Outcome`; the `starving` grade is the named failure verdict. | **(3) S/N + Faults paired strip** | **A** (the defining paired-axis: S/N *with* Faults — a raise-S/N-by-starving spikes the visible Faults lane) + **B** (`resident = signal + noise + unaccounted` must sum; `unaccounted` kept in the denominator so it can't over-claim, `signalnoise.go:48,116`) + **D** (both re-derived from `Outcome`). | WITNESSED |
| F5 | **Cell rot** (#782) | A paged-out / recalled memory cell **silently corrupted** — the bytes on disk no longer match the digest they were written under. The page reads back, but it is not the page that was saved. | `internal/recall` `PageSyndrome` (`page_syndrome.go:105,118`): the five-axis ECC-style per-page integrity roll-up — **digest** miss (`EvidenceDigest`), **quarantine**, write-time **durability** class (`EvidenceDurability`, `:51,186`), **witness** revocation, **trust-epoch** staleness. The repairable-vs-erasure verdict is `fault_syndrome.go:24,63` (`SyndromeClass`: a digest miss → ERASURE). | `PageSyndromeFor` (`page_syndrome.go:122`) recomputes the syndrome against the page's CAS body; `Reusable()`/`FailedEvidence()` (`:239,251`) name which axis failed. The scrub/repair read-back path is `internal/recall/scrub.go` + `parity.go`. | **(4) evict-correctness heatmap** (extended: per-cell integrity pass/fail, green only when every syndrome axis clears) | **D** (re-derive the digest from the CAS body and compare; RED on any axis miss) + **C** (a REPAIRABLE syndrome is MODELED-recoverable, an ERASURE is a hard WITNESSED loss — distinct lanes, never blended). | WITNESSED (digest re-derivation is fak's own) |
| F6 | **Tier page-down missed/faulted** (#985) | A tiered offload→restore (`KVTransfer`) reported `missed` or `fault` — the cache the engine wanted from a colder tier was not there (or came back wrong). Ties the multilevel-cache epic (#985) to a *visible outcome* instead of a trusted "it's on disk." | `internal/cachemeta/kvtransfer.go`: `KVTransfer{Direction, Tokens, Outcome}` with `Outcome ∈ {ok, missed, fault}` (`kvtransfer.go:25-27`) and `FaultReason` (`:46`); lowered through `internal/engine/capacity_adapter.go` `CacheEventRecorder` (demote/spill/evict, `capacity_adapter.go:8,16`) into the `fak_engine_cache_*` stream. | `FromKVTransfer` (`kvtransfer.go:54`) carries the outcome in `Labels` so a sink separates ok/missed/fault; the `CacheEventRecorder` is the control path between the placement decision and the metric. | **(1) ribbon** (the tiered page-down event lane, height = bytes, color = ok/missed/fault outcome) | **A** (offload event paired with restore outcome — a `missed` restore against a recorded `offload` is the gap) + **C** (a *projected* page-down-tier value stays MODELED until measured on-device, #1217 fence b). | WITNESSED for the outcome; projected tier value is MODELED |

---

## Two failure modes the family must surface but has NO closed witness yet (honest gaps)

These are real context-safety failures the plane is *supposed* to catch but for
which there is **no current closing witness in the tree**. Per the C1 doctrine
they belong in the catalog as named gaps — a gap that is *named* is catalog
content; a gap that is *silent* is the invisible-destruction failure the whole
epic exists to kill. Each names the missing seam and the child that would close
it.

| # | Failure | What goes wrong | Where it WOULD be observable | Honest gap (the missing witness) | Closing child |
|---|---|---|---|---|---|
| G-A | **Rendered visual drifts from source** | A debug-window number is rendered at scrape/session-end and **trusted thereafter** — the source moves, the picture doesn't, and no check reds. This is the structural failure #1217 §"fact 2" names: *we have the numbers, but no re-derivation cross-check over time.* | A CI gate / a checker that re-derives every shipped visual's numbers from the journal + `/metrics` + git and reds on disagree. | **No Go analog exists.** The journal IS replayable (`internal/journal/journal.go:577` `Verify`, `:619` `VerifyRows` recompute the hash chain) and `ctxplan/image.go` proves `RestoreIndex(ix.Image()) == ix` for the *index* — but **no checker re-derives a rendered visual** and reds when it disagrees. The pattern exists (`check_memory.py`, `dispatch_status.py` re-derive caps/closure from disk); it has no in-tree Go form for visuals. | **C9** (re-derivation witness) + **C10** (the CI gate) |
| G-B | **Harness-rewrite defeats the shed** | fak's cache-preserving compaction (or a quarantine) is **silently undone** by the agent harness's own auto-compaction rewriting the `cache_control` prefix — bursting the cache fak just preserved, or folding a sealed/quarantined span into its summary. fak shed value *correctly*; a second uncoordinated manager destroyed it anyway. | A per-turn prefix event attributing the cause: `harness_rewrite` (the harness burst the prefix) or `quarantine_at_risk` (a sealed span folded into a summary). | **Sensor exists, no closing witness/visual.** `internal/compactcohere` already emits the per-turn `PrefixEvent` (`harness_rewrite`, `fak_world_break`, `cold_ttl`, `compactcohere.go:28-44`) and a standing `PreCompact` block/allow `Posture` (`:130,151`) — but it is a tier-1 *sensor*; there is **no witnessed visual** that shows "fak preserved X, the harness then destroyed X," and no `quarantine_at_risk` survival witness. The actuators (gateway wiring, the guard hook, the quarantine-survival witness) are the open epic ([HARNESS-CACHE-COHERENCE-AUDIT](HARNESS-CACHE-COHERENCE-AUDIT-2026-06-28.md)). | a follow-on of the compactcohere epic; surfaced here so the plane doesn't render a "value preserved" that a second manager silently reversed |

---

## Acceptance check (against #1219)

The issue's acceptance: *every failure maps to (a) one journal `Kind` or metric
where it's observable, (b) one visual primitive, and (c) one cross-check that
goes red on it — no failure is "invisible."*

- **F1–F6** each bind exactly one observability seam (a journal `Kind`
  `CAP_EVICT`/`CAP_FAULT`, a typed verdict, a metric, or `KVTransfer.Outcome`),
  one of the six numbered primitives, and one lettered cross-check (A/B/C/D)
  that reds on it. Every seam is cited to a real `file:line`.
- The six skeleton rows from the issue body are all present and grounded: GDN
  evict panic (**F1**), evict-on-recurrent refused (**F2**), saved-but-never-reused
  (**F3**), S/N-up-but-faults-up (**F4**), cell rot (**F5**), tier page-down
  missed (**F6**).
- The two failures with **no closing witness** (**G-A** rendered-visual drift,
  **G-B** harness-rewrite-defeats-the-shed) are catalogued as **named gaps**, not
  omitted — which is the doctrine's definition of "no failure is invisible." Each
  names its missing seam and the child (C9/C10, the compactcohere follow-on)
  that closes it.

**Honesty fences carried (from #1217).** (a) Value-preservation is kept distinct
from #1147 mediation overhead — not blended. (b) WITNESSED (`max|Δ|=0`,
reuse-bit, digest re-derivation) is never summed with OBSERVED (provider
`cache_read`) or MODELED (projected tier value). (c) A row whose witness does not
yet exist is recorded as a `not yet` gap (G-A, G-B), not as a result.

---

_Filed as research/planning only under epic #1217. Parent: #1217. Sibling C1
doctrine note: #1218. Downstream: the visual-primitive schema (C6) and the
re-derivation witness (C9) consume this catalog's failure→witness→primitive→cross-check
bindings._
