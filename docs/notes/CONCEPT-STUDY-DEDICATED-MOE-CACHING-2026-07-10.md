# CONCEPT-STUDY: dedicated MoE caching systems (PiKV · ExpertFlow · MoEpic · Harvest · HybriMoE · DAOP · FlashMoE · SpecMD) — through the caching-observability lens

> Borrow-hunt pass, 2026-07-10. Uncommitted study note (scout-loop record). Goal: *explore
> dedicated MoE caching solutions* — the systems whose whole reason for existing is caching the
> MoE forward, cross-referenced against what fak already ships. Sibling to
> [[CONCEPT-STUDY-KTRANSFORMERS-2026-07-10]] (kvc2 + kt-kernel) and the M11/M12 ladder in
> [[CONCEPT-MODEL-PROGRESS-CACHING-TAXONOMY-2026-07-07]]. No code fetched (WebFetch is guard-blocked
> here, `TRUST_VIOLATION/ESCALATE`); read from paper text via WebSearch + each project's own claims,
> then witnessed against `C:/work/fak` @ 2026-07-10.

## The landscape — nine dedicated systems, two substrates

Every dedicated MoE cache is a residency manager for one of two objects. fak's existing SOTA map
(#2236) already names ktransformers, EPLB, MoE-Infinity, Pre-gated MoE, SiDA, Fiddler. The pass below
adds the systems the map does **not** reference — most in the 2025–2026 window.

| System (arXiv) | Object cached | Distinctive mechanism |
|---|---|---|
| **PiKV** (2508.06526, ICML'25 ES-FoMo) | **KV** (per-MoE) | expert-sharded KV storage + cache-aware routing (load-variance 10–14%); per-page **scalar utility** eviction with a policy zoo incl. **Belady-Approx** (predictive next-use) and **Hazard-LRU**; multi-strategy compression (SVD/Pyramid/LoRA/Quant) with reconstruction bounds |
| **ExpertFlow** (2410.17954) | expert weights | decoupled **Routing-Path Predictor** (T5-style, all-layers-one-pass, binary active/idle), Token Scheduler (regroup tokens by predicted route), Expert Cache Engine with **runtime misprediction correction** |
| **MoEpic** (2509.08342) | expert weights | **vertical expert split** (cache *top* segment of more experts, prefetch *bottom*); **LCP** priority = LFU⊕LRU (freq + interval); **CCA** divide-and-conquer sizing of per-layer VRAM budget + split-ratio from live hit-rate/pred-accuracy |
| **Harvest** (2602.00328, Jan'26) | expert wts + KV | **peer-HBM as a transient cache tier** over NVLink; framed as a **scheduler-robustness** mechanism for the "fault-service-dominates" regime under expert **workload drift** (arithmetic vs code) |
| **HybriMoE** (2504.05897, DAC'25) | expert weights | dynamic intra-layer CPU/GPU balance + impact-driven inter-layer prefetch + **score-based caching** for activation instability (on ktransformers) |
| **DAOP** (2501.10375, DATE'25) | expert weights | data-aware offload + **predictive pre-*calculation*** (compute the predicted expert on CPU one layer ahead, not just prefetch its bytes) |
| **FlashMoE** (2601.17063, Jan'26) | expert weights (**SSD**) | **ML-based cache replacement approximating Belady** for the SSD tier on edge; recency⊕frequency |
| **SpecMD** (2602.03921, ICML'26, Apple) | expert weights | standardized prefetch/evict **benchmark**; finding: **MoE access violates temporal locality (LRU/LFU are wrong)**; **Least-Stale** (current-vs-stale, layer-sequential) cuts collision misses ≤85× vs LRU; **prediction accuracy ≠ cache performance** |
| **MoE-SpeQ** (2511.14102) | expert weights | speculative quantized decode + proactive prefetch/offload |

## The decisive finding — the eviction *policy assumption* is the contested axis, and fak already owns its instrument (for KV)

Three of the newest systems converge on one claim that the older offload literature took for granted:
**LRU/LFU temporal-locality is the wrong prior for MoE expert access.** MoEpic replaces LRU with
LFU⊕LRU (LCP); FlashMoE approximates Belady with ML; and SpecMD (Apple, ICML'26) shows *empirically*
that MoE access is layer-sequential/deterministic, not recency-driven, so **Least-Stale beats LRU by
up to 85× on collision misses** — and, sharply, that **prefetch prediction accuracy does not equate to
cache performance**. That last point is the same QUALITY-not-RATE lesson the ktransformers study drew
from `get_attn_sparsity`: measuring the *decision's quality against an offline ceiling* matters more
than any single hit-rate or accuracy number.

fak **already realizes exactly that instrument — for KV**: `compute/kvreplay_oracle.go` (#2675, closed)
ships `BeladyKVReplayOracle` (exact offline-optimal DP for ≤63 spans + farthest-next-use fallback),
`ReplayKVCacheMulti` (LRU vs CostAware), and **`GoodDecisionRatio = realized/oracle`** — a policy-regret
gauge fed by real gateway-ledger prefix touches + a deterministic Zipf-bimodal synthetic corpus. This is
the KV-side twin of what SpecMD/MoEpic/FlashMoE argue you must measure for *experts*.

The gap the exploration surfaces is therefore precise and small: **that Belady/regret oracle is never
pointed at the expert-residency surface.** fak's only expert evictor, `pagedRing` (`model/paging_ring.go`,
#3174/#2726), is **plain LRU**, explicitly **off the serve path**, and instrumented only for **RATE**
(`pageIn`/`hit`/`evict` counts) — it has no Belady ceiling, no `GoodDecisionRatio`, and no
**temporal-locality-validity diagnostic** (does recency predict next-use for this trace, or is access
layer-sequential/anti-LRU as SpecMD found?). And #3902 scores a *static* frequency-topk expert **mask**
against observed routing — a different question from the *dynamic eviction policy's* regret.

## Candidates and witness verdicts (against C:/work/fak @ 2026-07-10)

| # | Borrow (system) | Lens | Verdict vs fak | Disposition |
|---|---|---|---|---|
| D1 | **Belady-regret + temporal-locality diagnostic** over the *expert-access* trace (SpecMD / MoEpic-LCP / FlashMoE) | eviction-policy **quality** | **PARTIAL** — engine PRESENT for KV (`kvreplay_oracle`/`GoodDecisionRatio`, #2675); **no expert-access trace source**, `pagedRing`=LRU RATE-only | **FILE** (expert-residency sibling of #3901; distinct from static-mask #3902) |
| D2 | Vertical **expert split** — cache *top* segment of more experts (MoEpic) | residency granularity | **ABSENT** as a *policy* (fak `moe_split.go` is packed-tensor *materialization*, not top/bottom residency; `pagedRing` admits whole weights) | action on off-path ring — **defer** (note only; a D1 gauge would tell us if it's worth it) |
| D3 | **LCP** (LFU⊕LRU) / **Least-Stale** production evictor | eviction policy | **ABSENT** — `pagedRing`=LRU; #2669 (KV hazard-rate LHD/LRB) is the nearest, design-only | **defer** — building a new production evictor on an off-path ring is premature; D1 measures the gap first (OBSERVABILITY-not-ACTION) |
| D4 | Peer-**HBM** as transient cache tier over NVLink (Harvest) | tiering | **ABSENT / blocked** — fak is single-device (`Caps.Collective=false`); no multi-GPU box live | **drop** (hardware-gated); revisit with EP #3382 |
| D5 | Harvest **"fault-service-dominates" regime** self-meter | observability | candidate — adjacent to shipped drain-lag #3394 / cacheability #3390, but expert-fault-service fraction is a distinct gauge | **note only** — fold into D1's report if cheap |
| D6 | Decoupled learned **Routing-Path Predictor** (ExpertFlow/DAOP) | prefetch | **ABSENT by choice** — fak's B2 angle is *prefix-coupled* prefetch (KV-prefix hit ⇒ known experts), not a standalone learned net; conflicts with deterministic-auditable ethos | **drop** (taxonomy already names MoE-Infinity/Pre-gated/SiDA as the SOTA here) |
| D7 | **Predictive pre-calculation** on CPU (DAOP) | compute-offload | **ABSENT** — `splitKernel` computes host-resident experts but not *speculatively*; spec-decode (#23) unwired | **drop** (action; different track) |
| D8 | **CCA** joint per-layer budget×split-ratio sizing (MoEpic) | budget sizing | **PARTIAL** — #3952 (KV budget→reuse sweep, knee/ceiling) is the closest shape; per-layer split-ratio is D2-gated | **drop** (subsumed by D1+#3952 if D2 ever fires) |
| D9 | PiKV multi-strategy **KV compression** (SVD/Pyramid/LoRA) | KV tiering | **PRESENT-frame** — fak KV-quant ladder #3144/#2240; lossy SVD/LoRA KV conflicts with bit-exact-or-fault ethos | **drop** |
| D10 | PiKV **expert-sharded KV** + cache-aware routing | distributed layout | **N.A. single-device**; cache-aware KV routing ≈ M6 (present concept) | **drop** |
| D11 | Token **scheduler** — regroup tokens by predicted route (ExpertFlow/HybriMoE) | batch scheduling | **ABSENT** — not fak's layer; `moe_host_batch.go` batches but does not permute by predicted route | **drop** (serving-throughput action, off the instrument axis) |

## The one filed leaf — D1

**`feat(model): expert-residency eviction-policy regret gauge — point the KV Belady replay oracle at the
observed expert-access trace, + a temporal-locality diagnostic`.** Reuse `compute/kvreplay_oracle.go`
(#2675): add an *expert-access* trace source (per-layer routed-expert touches, sized in resident weight
bytes — Q4_K/Q8, the same footprint `pagedRing` accounts) and replay it through `ReplayKVCacheMulti` +
`BeladyKVReplayOracle` to report `pagedRing`-LRU's **`GoodDecisionRatio`** (regret vs offline-optimal)
**and** a locality diagnostic (recency-predicts-next-use correlation vs the layer-sequential null SpecMD
found). Pure observability, off-path, ships the honestly-observed answer (à la `rescore_test.go` #2626).
It does **not** build LCP/Least-Stale into the production ring (D3, deferred).

Why this and not the flashier borrows: it lands on the **two axes the ktransformers study left open** —
**QUALITY-not-RATE** (D1 is the expert-residency sibling of the KV attention-mass recall gauge #3901) and
**OBSERVABILITY-not-ACTION** (measure the policy gap before actuating a new evictor, exactly as #3902 is
the instrument for the #3886 EPLB actuator). It is **reuse-not-reinvent** (the Belady DP, the
`GoodDecisionRatio`, the synthetic-trace generator, the gateway-ledger reducer all already ship — the
work is a new trace source + a locality stat, not a new simulator).

## Dedup performed

- No `CONCEPT-STUDY-DEDICATED-MOE-CACHING` note existed before this pass. PiKV, Harvest, FlashMoE,
  SpecMD, MoE-SpeQ, DAOP, HybriMoE are **not** in the #2236 SOTA map; MoEpic/ExpertFlow are not either.
- `gh` search empty for `belady OR least-stale OR temporal locality OR expert eviction OR policy regret
  OR pagedRing`. Nearest live issues, all distinct: **#2669** (KV hazard-rate reuse, design), **#3901**
  (KV attention-mass recall — *KV* not expert), **#3902** (expert *static-mask* drift — not the dynamic
  evictor), **#3952** (KV *budget* sweep — not policy regret), **#3393** (radixkv *thrash* detector — not
  regret-vs-optimal), **#2675** (KV Belady oracle — *KV* trace source only, closed).
- Tree grep confirms **no expert-access replay/trace source** exists (`internal/**` — only the taxonomy
  note mentions the idea); `moe_split.go` is packed-tensor materialization, not residency split.
- Hardware-gated borrows (D4 peer-HBM) parked behind EP #3382, not filed.
