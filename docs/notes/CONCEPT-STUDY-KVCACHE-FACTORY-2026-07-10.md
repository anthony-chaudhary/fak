# Concept study — KVCache-Factory (mined for fak), 2026-07-10

**Upstream:** [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) @ `94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0` (`94255b6`), cloned & read 2026-07-10.
**License gate:** main tree **MIT** (Zefan Cai); `csrc/` CUDA kernels **MIT** (66RING, 2023) — separate attribution. fak is Go, so **every borrow is `inspire` (clean-room, technique-only); no bytes vendored.**
**Sibling studies:** #3366 (LMCache), #3900 (ktransformers/kvc2), #3236 (ZML). Feeds the #2236 superset ranking. Filed under milestone **M2 — "The KV cache value is owned, observed & 2x."**

KVCache-Factory is a reference **KV-cache compression suite** (PyramidKV, SnapKV, H2O, StreamingLLM, AdaKV, Quest, Scissorhands, NACL, MiniCache, ThinK, HeadInfer, MInference, KIVI/GEAR quant). The transferable surface is the **selection/retention/merge management pattern**, mapped onto fak's `memq` retention algebra, `ctxplan` compaction, `l3kv`/`blobfs` durable store, `vcachecal` concentration, and `cachevalue` budget plane. KV token/head/channel ↔ memq cell / recall span / cell field.

## Method (study-repo pass)
A background workflow fanned **13 DEEP readers** over the load-bearing subsystems (press-core PyramidKV/SnapKV/H2O/StreamingLLM/AdaKV, Quest, Scissorhands, NACL, MiniCache, HeadInfer, quant KIVI/GEAR, MInference, ThinK/think-variants, monkeypatch/GQA, eval-harness, tests, CUDA), each emitting **axis-ablated candidate borrows** (method + axis + `path:line@sha` + upstream tradeoff + fak-analog hypothesis). Each candidate was then **witnessed against fak** (`fak_feature_query` + `fak index` + grep + Read of the real seam) → verdict PRESENT / PARTIAL / ABSENT with a cited `fak_seam path:line`, borrow_kind, borrow_value, and a first checkable step. A final **completeness critic** pass deduped overlapping candidates and surfaced unmined axes.

**Yield:** 156 witnessed candidates — **50 PRESENT** (dropped), **98 PARTIAL**, **7 ABSENT**, 1 rate-limited. (170 agents, ~11.7M tokens.)

## Dropped as PRESENT (fak already owns the axis — NOT re-filed)
- **H2O accumulated heavy-hitter scoring** — memq already scores by relevance-to-intent (RankRelevance), not lifetime frequency.
- **StreamingLLM/Quest sink+recent retention & query-conditioned selection** — ctxplan/memq keep a protected recent window + relevance selection.
- **HeadInfer lossless offload + async prefetch-next + zero-copy per-unit partition** — `l3kv`/`blobfs` tiered offload + `paging_ring`; core axes present. (See also #3319 regenerable KV, #3062 restore.)
- **Full-precision recent residual + KV byte quantization (KIVI/GEAR/QuantizedCache)** — fak's recency-tiered fidelity already keeps recent full / compresses old.
- **Multi-tier residency + demote-not-evict** — `cachemeta.PlanPlacement`, TierProfile, per-tier TTL (already established by #3900).
- **Ratio-capped eviction planning** — #3387. **Recall/attention-mass quality gauge** — #3901. **Hit-rate observability** — #3390/#3367/#3391.
- **Eval provenance `run_meta.json` (commit+args+versions), fail-closed `eval_batch_size==1` reject, bit-identical-default staged migration gate, env-gated hot-path debug, middle-truncation budget fit** — fak's benchauthority/rollout discipline already covers these.

## Filed borrows (PARTIAL/ABSENT → child leaves of the study epic)
All target `internal/memq` (+ `ctxplan`, `vcachecal`) retention algebra, which today applies a **single scalar budget over a per-cell-independent relevance sort** — the recurring gap the suite fills.

| Leaf | Axis (borrow) | Upstream `path:line` | fak seam | verdict |
|---|---|---|---|---|
| A | **Neighborhood-pooled relevance** — 1D avg/max pool importance over local neighbors before top-k so contiguous spans beat isolated spikes (SnapKV) | `pyramidkv/pyramidkv_utils.py:520` | `internal/memq/exec.go:195` | PARTIAL |
| B | **Merge-on-evict** — route each dropped cell into its nearest surviving cell and fold, instead of discarding (LOOK-M pivot / CAM / KVMerger) | `pyramidkv/pyramidkv_utils.py:199,723` | `internal/memq/exec.go:397,533` | PARTIAL |
| C | **Concentration-weighted budget** — boost a bucket's claim on a shared budget when its importance mass is concentrated in few items (AdaKV) | `pyramidkv/pyramidkv_utils.py:1043` | `internal/vcachecal/concentration.go:49` | PARTIAL |
| D | **Protected-floor budget reservation** — reserve durable/recent cells before the relevance fill (+ monotone per-tier floor), verbatim-recent invariant (NACL/Scissorhands/PyramidKV) | `pyramidkv/nacl.py:142`, `scissorhands.py:127` | `internal/memq/exec.go:306`; `internal/ctxplan/layout.go:80` | PARTIAL |
| E | **Divergence-outlier exemption** — carve the top-K most-divergent cells and keep them bit-exact, exempt from lossy consolidation (MiniCache) | `pyramidkv/minicache.py:87` | `internal/memq/drivers.go:149`; `exec.go:533` | PARTIAL |
| F | **Per-cell width/field pruning (OpNarrow)** — shrink a retained cell's width (drop low-value fields) as an orthogonal 2nd compression axis after cell-eviction (ThinK) | `pyramidkv/pyramidkv_utils.py:26,589` | `internal/memq/exec.go:306`; `drivers.go:79` | ABSENT/PARTIAL |
| G | **Multi-intent recall with tunable score-agg** — reduce N per-intent relevance rankings into one retained set via a tunable op (mean/amax/sum) (GQA group-reduce) | `pyramidkv/pyramidkv_utils.py:252` | `internal/selfquery/rrf.go:111`; memq recall | PARTIAL |
| H | **Deterministic anti-starvation credit** — grant a survival credit to a non-durable cell that sits just below the cutline for K consecutive passes (NACL diversity / Scissorhands, RNG-free) | `pyramidkv/nacl.py:173`, `scissorhands.py:68` | `internal/memq/exec.go:219`; `drivers.go:96` | ABSENT |

**Deliberately NOT borrowed:** NACL/Scissorhands *stochastic* (RNG) eviction and injected-RNG reproducibility — fak's retention is deterministic-by-construction (strictly stronger for audit). Only the **anti-starvation/diversity axis** is borrowed, deterministically (leaf H).

## Further candidates (surfaced, not filed under this cache epic — wrong home / lower signal)
- **Multi-terminator stop-token assembly** — collect every valid halt token (eos list + tokenizer eos + `<|eot_id|>` + `<|end_of_text|>`, deduped) so dual-terminator models stop correctly (`pyramidkv/eval_utils.py:17`). Real small **generation-correctness** gap; belongs to the agent/harness lane, not this cache epic.
- **Offline-computed per-unit profile shipped as versioned JSON data** (loaded, not runtime-computed) driving budgets/dispatch — MInference sparse-pattern JSON + HeadKV `heads_score/*.json` (`data/heads_score/*.json`). Maps to `dispatchtick/launchprofile`. Candidate for the dispatch/cachevalue lane.
- Critic-flagged unmined axes: version-gated monkeypatch compat gate, needle-in-haystack 2D (length×depth, linear/sigmoid) eval grid, per-task-type scorer registry with partial-credit. (Mostly eval-harness; fak's bench plane largely covers.)
