# CONCEPT-STUDY: kvcache-ai/ktransformers — 3-tier KV cache + MoE expert placement + sparse-KV residency, through the caching-observability lens

> Borrow-hunt pass, 2026-07-10. Uncommitted study note (scout-loop record). Source repo
> pinned at `kvcache-ai/ktransformers` @ `7c021b430c36a408032c20bbf3833dc1bce6efa4` (`7c021b4`),
> including `kt-kernel/` and the vendored **kvc2** 3-tier KV-cache manager. Read: kvc2 tiering +
> eviction core, kvc2 prefix-cache + lookup path, kt-kernel expert placement/offload + expert-weight
> cache, and the archived expert operators / `optimize_rules` / scheduler (4 parallel deep readers).
> C++ bodies (`kvcache_read_write.cpp`, `kvcache_utils.cpp`, `sft_moe.hpp`) were header/attn-read.

## What ktransformers / kvc2 is

ktransformers is a local-inference engine for big MoE models on modest hardware (CPU/GPU hybrid,
expert offload). Two of its subsystems are KV/residency managers one level below fak:

- **kvc2** — a 3-tier (GPU/CPU/disk) KV-cache manager: cumulative page-boundary prefix-hash block
  identity, longest-prefix match, per-tier dirty/ref state, demand-driven eviction, bump/geometric
  disk allocator, single-flight transfer dedup.
- **kt-kernel expert placement** — chooses a GPU-resident expert set **offline** from an activation-
  frequency table (`experts_base.py:21-72@7c021b4`, `generate_gpu_experts_masks` →
  `torch.topk(freq, num_gpu_experts)`), enforced per-token by `should_skip_expert`
  (`operators/common.hpp:256-258@7c021b4`).
- **kt-kernel sparse KV attention** — per-query keeps init-sink + top-pick-by-similarity + local-recent
  blocks (`kvcache_attn.cpp:737-834@7c021b4`) over online per-block importance/anchors
  (`kvcache.h:53-58,383,386@7c021b4`), and reports its own **recall** via `get_attn_sparsity`
  (`kvcache.h:350-353@7c021b4`).

## The decisive finding — two residency machines, opposite dynamics

ktransformers spends a **STATIC** placement on the *expensive* object (GB-scale expert weights,
picked once from a frequency table, never adapted) and a **DYNAMIC** one on the *cheap-to-reselect*
object (KV blocks, re-selected per query with a built-in recall metric). The lesson for a caching-
observability substrate is the two gaps this exposes in fak: (a) nothing **scores a static placement
against observed access** ("has the pinned hot-set gone cold?"), and (b) nothing measures the
**QUALITY (recall)** of a residency/eviction decision, only its **RATE**. Those became the two leaves.

## Candidates and witness verdicts (against C:/work/fak @ 2026-07-10)

| # | Borrow (source @7c021b4) | Lens | Verdict vs fak | Disposition |
|---|---|---|---|---|
| P7 | Attention-mass **recall** of the selected set (`get_attn_sparsity`, `kvcache.h:350`) | hit-quality | **PARTIAL** (mass carrier #852/#855 + evictor #856 present; recall gauge absent) | **Filed #3901** |
| P2 | Static freq-topk resident **expert mask** (`experts_base.py:21-72`) | expert-residency | **PARTIAL/ABSENT** (narrow `q4kExpertStats`; no drift score) | **Filed #3902** |
| T8 | Match index **recomputed from tokens** on load, not persisted | keying | related to regenerable-KV | **Enriched #3319** |
| E1 | `can_desert = ref==0 && !dirty`, decided **per tier** | eviction | adds clean/dirty axis | **Enriched #3384** |
| M2 | `lookup_alt`/`recompute_ratio` non-contiguous match — **stubbed** | non-prefix reuse | field-evidence datapoint | **Enriched #3143** |
| P4 | Online per-block **importance** (QUEST anchors), `update_importance` | reuse signal | adaptive-value datapoint | **Enriched #2669** |
| T1 | Multi-tier residence state + demote-not-evict | tiering | **PRESENT** (`cachemeta.PlanPlacement`, CXL/NUMA `TierProfile`, per-tier TTL) | drop |
| A2 | Single-flight transfer dedup (present/in-flight/issue) | concurrency | **PRESENT** (blobfs in-flight coalesce #3380) | drop |
| — | Cacheability-vs-realized hit-rate split | observability | **PRESENT** (#3390 shipped) | drop |
| — | Pipeline drain-lag / throughput self-meter | observability | **PRESENT** (#3394 shipped) | drop |
| T5 | Restore-on-access cold→hot promotion | tiering | **PRESENT** (`l3kv` RestoreSpan, #1469) | drop |
| K2 | Key namespacing by (model,quant,K/V,layer) | keying | **PRESENT** (`cachemeta.MaterializationKey`) | drop |
| K4 | 64-bit hash identity + collision seam | keying | **PRESENT** (fak sha256, stronger) | drop |
| M1 | Longest-prefix via binary-search over block hashes | matching | **PRESENT** (radixkv trie, same capability) | drop |
| C1-3 | Column/TP block sharding, atomic multi-lock reserve | GPU layout | **PRESENT** (#3382/#3386) / N.A. for Go | drop |
| P9 | KV compression tiers (Q4/Q8/FP16) bytes-vs-fidelity | tiering | **PRESENT** (cachemeta tiers + #3144) | drop |
| A1,A5-7,T6-7,P11 | Slab allocator, BatchPromise, RAII timer, bump allocator, NUMA arena | substrate | **N.A.** (Go GC / content-addressed blobfs) | drop |
| P5 | 3-band residency (pinned-sink / value-pick / recent) | residency policy | PRESENT-ish (kvmmu + sinks) | note only |
| P6 | Cache the residency **decision** + staleness | 2nd-order | weak fak seam | note only |
| P3 | On-disk materialized-transform cache hit(load)/miss(recompute) | weight cache | weak lens (blobfs adjacent) | note only |

## Dedup performed

- No `CONCEPT-STUDY-KTRANSFORMERS` note existed before this pass.
- **`gh` search empty** for both new leaves ("attention mass captured/recall/selection quality";
  "expert placement observed routing hot-set drift").
- **Adjacent same-day EPLB study (#3886)** is the placement **actuator** (rebalance by load);
  **#3902 is the instrument** (observe routing + score drift — the trigger that fires #3886). Explicit
  scope-boundary cross-link posted on #3902. Distinct from expert-weight residency/offload tiering
  (#3174/#3212, VRAM-vs-host axis).
- Sibling KV-cache studies (LMCache #3366, Mooncake, SGLang-caching, vLLM-M2) already own the
  keying/tiering/eviction/hit-rate surface — hence the long PRESENT/drop column above; only the two
  ktransformers-distinctive residency machines survived as gaps.

## Filed

- **Epic #3900** — `epic(ktransformers-study)`: container + finding + PRESENT-drop ledger.
- **#3901** — `feat(kvmmu)`: attention-mass **RECALL** gauge (captured-fraction of the retained set;
  hit-QUALITY complement to `cacheobs` RATE). Delta on the closed attention-witness epic #851.
- **#3902** — `feat(model)`: expert-placement **drift** scorer (per-expert routing histogram +
  coverage/drift of a static resident mask vs observed routing). Feeds #3886 / #2722.
- **Enriched (no re-file):** #3319 (index keys regenerable-from-tokens), #3384 (dirty axis, per-tier),
  #3143 (kvc2 stubs non-prefix reuse — 2nd project deferring it), #2669 (online importance signal).

## Honest limits

- Witness is lexical + a 2026-07-10 snapshot (`fak_feature_query` / `fak index` + raw grep + `gh`
  dedup); re-witness before building.
- **Zero from-scratch borrows** — both leaves are *specific deltas* on existing fak mechanisms
  (attention-witness #851; `expert_parallel.go` / `q4kExpertStats`), which is why they're small.
- P5/P6/P3 left as notes, not issues: a false-ABSENT is worse than no issue (the EPLB note's rule).
- Companions: `CONCEPT-STUDY-LMCACHE-2026-07-08.md`, `CONCEPT-STUDY-EPLB-2026-07-10.md`,
  `CONCEPT-STUDY-MOONCAKE-2026-07-10.md`, `CONCEPT-STUDY-SGLANG-CACHING-2026-07-10.md`.
