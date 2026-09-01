# Concept study: mini-sglang's compact serving control loop

> Studied 2026-08-22. Upstream: [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang), pinned at [`9a91cfafe754aa85daee49998176275667eb58f2`](https://github.com/sgl-project/mini-sglang/commit/9a91cfafe754aa85daee49998176275667eb58f2) (2026-05-17). License: MIT. No upstream source is copied here.

## Verdict

mini-sglang is most useful to fak as a **small, readable integration map** for a serving engine, not as a kernel source. Its valuable seam is the explicit loop connecting exact-prefix reuse, page-aware KV allocation, bounded chunked prefill, continuous decode, fixed-shape CUDA-graph replay, and latency/throughput measurement. Fak already has the deeper radix-prefix substrate; the important remaining borrow is to make those mechanisms operate together in one real Qwen3.8 serving campaign under [#8395](https://github.com/anthony-chaudhary/fak/issues/8395).

The strongest near-term default is therefore **integration before invention**: retain fak's existing `internal/radixkv` rather than porting mini-sglang's Python tree, then use mini-sglang's compact scheduler as a readable reference for the missing end-to-end serving spine. CUDA-graph capture remains a follow-on optimization, not a prerequisite for that spine.

## Value frame

- **For:** coding-agent workloads with long shared prefixes and concurrent requests.
- **Problem:** fast kernels and an isolated prefix cache do not by themselves produce low TTFT, stable inter-token latency, or high aggregate throughput.
- **Today:** fak has a mature radix-prefix cache and Qwen3.8 kernel/campaign work, but [#8395](https://github.com/anthony-chaudhary/fak/issues/8395) still tracks the unproven integration of paged KV, prefix reuse, bounded prefill, and continuous batching.
- **Better because:** mini-sglang exposes that integration in a compact control loop whose invariants can be borrowed without importing its Python/CUDA framework.
- **Witness:** the #8395 four-arm sanctioned-GPU campaign reports quality, TTFT, ITL, throughput, VRAM, prefix-hit rate, and fallback count.

Centrality: **Core**. P1 managed context advances through exact prefix identity and explicit eviction; P2 net efficiency is measured rather than inferred; P3 each mechanism stays independently disableable; P4 the campaign exposes occupancy and fallback instead of hiding adaptation.

## Dated source ledger

| Surface | Pinned evidence | Observation (as of 2026-08-22) |
|---|---|---|
| Project intent | [`README.md:9-18`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/README.md#L9-L18) | The project deliberately trades framework breadth for a compact educational implementation while still targeting modern serving optimizations. |
| Prefix cache | [`radix_cache.py:93-237`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/kvcache/radix_cache.py#L93-L237) | A token radix tree performs longest-prefix matching, splits partial edges into reusable boundaries, tracks references, and evicts eligible leaves by recency. |
| Page-aware allocation | [`cache.py:19-91`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L19-L91) | Allocation rounds and reservation decisions through the configured page size; upstream issue/PR #80 added a regression suite after page-size-aware eviction failed. |
| Chunked prefill | [`prefill.py:33-151`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py#L33-L151) | One `token_budget` is consumed across requests; an oversized request becomes a resumable `chunked_req` carrying cache handle, table slot, and cached length into the next batch. |
| Continuous decode | [`scheduler.py:62-106`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L62-L106), [`decode.py:10-39`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/decode.py#L10-L39) | The scheduler chooses prefill when budget permits, otherwise decode, and keeps runnable decode requests resident between steps. |
| Cross-rank order | [`decode.py:32-35`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/decode.py#L32-L35), [commit `9a91cfa`](https://github.com/sgl-project/mini-sglang/commit/9a91cfafe754aa85daee49998176275667eb58f2) | The pinned tip sorts decode requests by stable request UID. The latest commit exists specifically because set iteration produced inconsistent request order across tensor-parallel ranks. |
| CUDA graph replay | [`graph.py:16-156`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/graph.py#L16-L156) | Decode graph capture preallocates fixed buffers for selected batch sizes, pads a live request to the nearest captured size, copies dynamic values in, then replays. |
| Pipeline overlap | [`engine.py:130-215`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py#L130-L215) | CPU scheduling, device execution, and result handling are separated with streams/events so the next scheduling step can overlap the previous device step. |
| Measurements | [`benchmark/client.py:333-383`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/benchmark/client.py#L333-L383), [`benchmark/offline/bench.py:30-38`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/benchmark/offline/bench.py#L30-L38) | Online output reports TTFT, TPOT, and E2E percentiles; offline output reports aggregate token throughput. |
| Tests | [`test_scheduler.py`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/tests/core/test_scheduler.py), [`test_cache_allocate.py`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/tests/core/test_cache_allocate.py) | Scheduler and cache allocation tests cover admission/chunking and multiple page sizes, including the historical page-size eviction failure. |
| History | [171 commits through pinned tip](https://github.com/sgl-project/mini-sglang/commits/9a91cfafe754aa85daee49998176275667eb58f2/) | The initial public sequence introduced engine, CUDA graphs, scheduler, OpenAI API, TP, chunked prefill, and radix cache as separate increments; later fixes concentrated on cancellation, page-size eviction, races, sampling consistency, radix recency, and TP ordering. |
| Issues / PRs / releases | [Issues](https://github.com/sgl-project/mini-sglang/issues), [pull requests](https://github.com/sgl-project/mini-sglang/pulls) | GitHub reported 12 issues and 38 pull requests. There were no releases, tags, or discussions at study time; `main` is the only authoritative release surface. |

Repository metadata observed on 2026-08-22: created 2025-09-01; 4,794 stars; 798 forks; last source push 2026-05-17; 121 tracked files; 171 commits at the pinned clone. These are dated observations, not project-controlled guarantees.

## Architecture in one pass

The useful control path is short:

1. The tokenizer/frontend creates a request with stable identity and sampling state.
2. Prefix matching converts already-computed tokens into a cache handle and table position.
3. The prefill adder spends a bounded token budget; a prompt that does not fit becomes resumable state rather than monopolizing a batch.
4. Decode requests remain in a running set, but are sorted before execution so every TP rank sees the same tensor row ordering.
5. The engine prepares request/token tables and attention metadata, then executes eager or fixed-shape graph replay.
6. Finished/cancelled requests release table and cache references; reusable prefixes remain governed by radix recency and refcounts.
7. The benchmark client measures user-visible TTFT/TPOT/E2E while the offline runner measures aggregate throughput.

This is the transferable contribution: each optimization has an explicit ownership seam and the scheduler composes them without making the cache, graph runner, or frontend own global policy.

## Candidate matrix

| Candidate axis | mini-sglang evidence | Current fak witness | Verdict | Portfolio | Disposition |
|---|---|---|---|---|---|
| Split a partially matched radix edge into an exact reusable boundary | `radix_cache.py:168-231@9a91cfa` | `internal/radixkv/radixkv.go:355-415` performs walk-then-split; `MatchLenNS` exposes read-only hit accounting. | **PRESENT-on-axis** | **DEFAULT** | Keep fak's implementation. It is native Go, namespace-aware, snapshot-tier-aware, and already records why it intentionally does not copy SGLang's Python-oriented galloping compare. |
| Protect in-use prefixes and evict only eligible leaves under bounded residency | `radix_cache.py:101-107,140-165@9a91cfa` | `internal/radixkv/radixkv.go` tracks references and chooses snapshot victims only when unreferenced; `internal/radixkv/eviction_strategy.go` supplies explicit policies. | **PRESENT-on-axis** | **DEFAULT** | Keep and dogfood via [#7750](https://github.com/anthony-chaudhary/fak/issues/7750); no duplicate issue. |
| Spend one bounded prefill-token budget and persist a partially admitted request | `prefill.py:33-151@9a91cfa` | `fak capabilities` finds context reuse but no model-serving prefill admission seam; repository search finds the target only in the open Qwen3.8 campaign. | **ABSENT-on-axis** | **DEFAULT after witness** | Existing [#8395](https://github.com/anthony-chaudhary/fak/issues/8395) is the deduplicated implementation issue. Its four-arm campaign must show TTFT/ITL and fallback effects before default enablement. |
| Compose paged KV, exact-prefix reuse, chunked prefill, and continuous batching in one serving loop | scheduler/cache sources above | Fak has `internal/radixkv` and Qwen3.8 kernel/campaign seams, but no committed real-serving witness proving the combination. | **PARTIAL-on-axis** | **DEFAULT after witness** | Existing #8395 exactly owns this integration and already excludes reimplementing SGLang wholesale. |
| Stabilize per-request row order before a tensor-parallel decode collective | `decode.py:32-35@9a91cfa`; latest fix #113 | Fak does not yet expose the #8395 tensor-parallel serving batch seam to harden. | **ABSENT-on-axis, blocked by spine** | **RECIPE / acceptance invariant** | Carry stable request-ID ordering into #8395's eventual multi-rank batch representation and its deterministic test. Filing a separate implementation leaf now would lead the serving spine. |
| Capture and replay fixed-shape CUDA decode graphs, padding to a supported batch size | `graph.py:16-156@9a91cfa` | Searches across `internal/compute`, Qwen3.8 runtime, and campaign code find kernels/graphs as models and plans, but no CUDA graph-capture runtime; #8395 does not require it. | **ABSENT-on-axis** | **WATCH** | Defer until the four baseline mechanisms are integrated and ablated. Graph capture is optimization after the working spine, and mini-sglang supplies no independent graph ablation proving net gain for fak's target workload. |
| Report TTFT, per-output-token latency, E2E percentiles, aggregate tok/s, VRAM, hit rate, and fallback together | benchmark sources above | #8395's done condition already requires TTFT p50/p95, ITL p50/p95, aggregate tok/s, peak VRAM, prefix-hit rate, and fallback count. | **PRESENT-as-contract; unproven at runtime** | **DEFAULT** | Reuse #8395; do not open a benchmark duplicate. Mini-sglang adds useful p99 precedent, but p99 is too noisy for a first small campaign unless sample size supports it. |

`fak capabilities` was queried with both umbrella and axis-specific phrases, then checked against raw repository searches and the actual `internal/radixkv` implementation. The lexical query correctly found fak's provider-context reuse but could not establish model-serving scheduler coverage; code and issue evidence establish the finer classifications above.

## What not to borrow

- **Do not port the Python scheduler or radix tree.** MIT permits copying, but fak's native Go cache is already broader and integrated with namespaces, multi-tier snapshots, explicit byte budgets, and richer eviction policy.
- **Do not make CUDA graphs part of the first serving spine.** mini-sglang demonstrates the mechanism but not a controlled graph-on/graph-off workload result. The spine-first order is serving integration, captured campaign, then measured optimization.
- **Do not inherit the supported-model matrix as a roadmap.** The compact project serves Llama/Mistral/Qwen variants and assumes NVIDIA-centric Python dependencies; fak's target remains its declared Qwen3.8 campaign and sanctioned compute envelope.
- **Do not treat stars, README performance language, or lack of open bugs as benchmark proof.** No release/tag exists, and the repository's online/offline scripts are measurement surfaces rather than cross-engine controlled evidence.
- **Do not copy fixed global defaults blindly.** `chunked_prefill_size`, graph batch sizes, page size, and memory fraction must be derived from fak's workload and device witness rather than transferred as constants.

## Historical failure lessons to carry into fak

The most valuable history is corrective, not promotional:

- page-size-aware eviction required a dedicated multi-page regression (#80);
- splitting a radix node had to refresh the new parent's recency (#124);
- unordered decode requests diverged across TP ranks (#113);
- cancellation and attention-backend races were later fixes, not properties guaranteed by the initial design;
- sampling consistency failed when random state was not initialized deterministically (#88).

For #8395, these translate to acceptance invariants: test at more than one page size; preserve recency when structure changes; sort stable request IDs before rank-coupled execution; cancel without leaked KV/table ownership; and make random state explicit per request.

## Filed and deduplicated trail

- [#8395](https://github.com/anthony-chaudhary/fak/issues/8395) — **open, primary borrow**: paged KV + prefix reuse + bounded chunked prefill + continuous batching, with isolated arms and a real Qwen3.8 GPU witness. The mini-sglang scheduler/cache sources above are direct implementation references for this existing issue.
- [#7750](https://github.com/anthony-chaudhary/fak/issues/7750) — **open, existing maturity follow-through**: dogfood `radixkv` with a passing runtime proof. This covers adoption of the already-present radix substrate, not the missing serving integration.
- [#8461](https://github.com/anthony-chaudhary/fak/issues/8461) and [#8462](https://github.com/anthony-chaudhary/fak/issues/8462) — **open, adjacent but not substitutes**: Random Access Cache portfolio work composes addressed reuse with radix-prefix candidates; it must not displace the simpler exact-prefix default proven by #8395.

No new issue was opened. Every actionable PARTIAL/ABSENT default candidate deduplicates to #8395; radix dogfooding deduplicates to #7750. CUDA graph replay is explicitly WATCH until the serving spine exists, and stable TP order is an acceptance invariant of that future multi-rank seam rather than an independently runnable leaf today.

## Completion boundary

This study establishes where mini-sglang should influence fak and binds each useful mechanism to current fak code or an existing issue. It does **not** claim that #8395 is implemented, that mini-sglang's performance claims reproduce on fak hardware, or that CUDA graph replay is worthwhile for agent traffic. Those claims require the sanctioned-GPU campaign already named by #8395.

## Exhaustive inventory refresh (2026-08-25)

Issue #9000 refreshed the study denominator without changing the original technical verdict. The pinned machine-readable map is [`docs/research/inventory/sgl-project-mini-sglang.json`](../research/inventory/sgl-project-mini-sglang.json), generated from all **121 files**, **32 directories**, and **10,874 text lines** at `9a91cfafe754aa85daee49998176275667eb58f2`.

The non-tree audit paged the GitHub surfaces through that revision's `2026-05-17T12:37:42Z` timestamp: **25 issues** (12 open, 13 closed), **123 pull requests** (38 open, 85 closed), **0 releases**, and **171 commits**. GraphQL confirms discussions are disabled. The tree has no standalone roadmap or changelog; its roadmap signal is nine unfinished-work marker sites covering FA4/Blackwell, MLA and HiCache, host-cache matching and prefill estimation, decode-first scheduling, sampling parameters, and batch tokenization. `LICENSE` and `pyproject.toml` confirm MIT provenance.

Three candidate-specific `fak capabilities` self-queries covered prefix/KV reuse, serving scheduler and tensor-parallel ordering, and user-visible serving measurement. The six-candidate matrix in the map adjudicates every retained idea as already owned, already tracked by [#8395](https://github.com/anthony-chaudhary/fak/issues/8395), or deliberately stay-minimal. No new borrow issue survived; #9000 is the inventory tracker and #8395 remains the implementation owner.

Completeness critic: the generator walked the entire pinned tree; the refresh separately exhausted issues, pull requests, releases, and commit history, confirmed the absence of discussions, inspected all repository TODOs and license evidence, ran the fak self-query, adjudicated the candidate matrix, and reconciled issue tracking. `.git` is the only skipped control directory.
