# Study: vllm — witnessed borrows for fak

- **Repo:** https://github.com/vllm-project/vllm.git
- **Pinned:** `b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd` (`b6ff8a2f`), HEAD dated 2026-07-19; latest commit `[Core] Add MRV2 virtual-batch PCP for MLA (#46570)`.
- **License:** Apache-2.0 — INSPIRE and INTEGRATE both permitted (all borrows below are **inspire**, clean-room Go, no bytes vendored).
- **Method:** deep `/study --deep`; 7 parallel subsystem readers, each returning `path:line@b6ff8a2f`; on-axis witness against fak by dogfooded index + raw grep + reading fak's code on each seam.

## Fan-out coverage

| Reader | Subsystem | Load-bearing files read |
|---|---|---|
| 1 | V1 scheduler + engine | `vllm/v1/core/sched/{scheduler,async_scheduler,request_queue,interface,output,utils}.py`, `vllm/v1/engine/{core,async_llm,coordinator,output_processor,parallel_sampling}.py`, `vllm/config/scheduler.py` |
| 2 | KV block pool + automatic prefix caching | `vllm/v1/core/{block_pool,kv_cache_manager,kv_cache_utils,kv_cache_coordinator,single_type_kv_cache_manager,kv_cache_metrics}.py` + prefix-cache tests |
| 3 | KV offload/tiering + disaggregation + KV events | `vllm/v1/kv_offload/**`, `vllm/distributed/{kv_transfer/**,kv_events.py,ec_transfer/**}` |
| 4 | Speculative decoding | `vllm/v1/spec_decode/{ngram_proposer,ngram_proposer_gpu,suffix_decoding,eagle,medusa,draft_model,llm_base_proposer,dynamic/*,metadata,metrics,utils,vocab_mapping}.py`, `vllm/v1/sample/rejection_sampler.py` |
| 5 | Structured / guided decoding | `vllm/v1/structured_output/**` (xgrammar/outlines/guidance/lmfe backends), `vllm/reasoning/**` |
| 6 | Distributed load-balancing | `vllm/distributed/{eplb/**,elastic_ep/**,weight_transfer/**}`, `vllm/v1/executor/**` |
| 7 | LoRA + quant + attention registry + platforms/plugins (breadth) | `vllm/lora/**`, `vllm/model_executor/layers/quantization/**`, `vllm/v1/attention/backends/registry.py` + `selector.py`, `vllm/platforms/**`, `vllm/plugins/**` |

### Completeness-critic residue (opened-or-justified)

Not opened, with justification — none yields a *system* borrow beyond what a kernel matrix or fak's own model-support track already owns:
- `csrc/`, `kernels/`, `cute_utils/`, `vllm_flash_attn/`, `third_party/`, `device_allocator/`, `compilation/` (torch.compile/cudagraph), lora `punica_gpu`, quant `schemes/` kernels — GPU/C++ kernel math; route via `sota-check`, and fak's kernels are GPU server gated.
- `models/` — per-model architecture ports; fak's own track (#1026 Ornith, #4867 Bonsai, #4033 VLM), not a system borrow.
- `entrypoints/`, `parser/`, `tool_parsers/`, `renderers/`, `inputs/`, `tokenizers/`, `transformers_utils/` — API/tokenizer surface; fak has its own gateway + adapters + tool parsers.
- `engine/` (top-level V0 legacy) — superseded by `v1/` (the live path, read); vllm is mid-migration.
- `benchmarks/`, `profiler/`, `tracing/`, `usage/`, `ray/`, `pool/` (embeddings), `logits_process.py` — telemetry/glue/task-classes fak doesn't center; fak has its own observability stack.

Verdict: no material subsystem left unopened for a system-level fak borrow.

## Reconstructed worldview (who vllm is built for)

vllm optimizes **aggregate GPU throughput for high-QPS, multi-tenant online serving on a fixed, scarce pool of GPU memory.** Every hardware/algorithm choice is a lazily-resolved *capability-keyed registry* (platform, attention backend, quant method, LoRA resolver) so one binary serves heterogeneous hardware + many tenants with no recompile and O(1) runtime switching. The two recurring levers are (a) **content-addressing everything by block-hash** so GPU/CPU/disk/remote/offloaded KV are all just prefix-cache tiers the scheduler can rematerialize, and (b) **packing more concurrent state into fixed GPU memory** (fp8 KV cache, two-tier LoRA slot pools, phaseless token-budget continuous batching). Its non-goals are visible in the defaults: FCFS, recompute-on-preempt (never swap KV to host), single-backend-per-engine — throughput and GPU utilization over per-request fairness or footprint. That user world (a GPU cluster operator maximizing tokens/sec) is where fak **diverges**: fak governs *audited agent turns* over a *small* model set on *slow CPU/SSD-offloaded* tiers, so most of vllm's GPU-memory-packing and SPMD-collective machinery is a different-user tradeoff, while its content-addressing, speculation, and constrained-decode ideas transfer directly.

## Candidate table

Witness grain = the narrow axis, not the capability name. `path:line@b6ff8a2f`.

| Borrow | Source | Axis | Their-worldview reason | Witness (fak seam) | Verdict |
|---|---|---|---|---|---|
| Model-free prompt-lookup n-gram drafter | `vllm/v1/spec_decode/ngram_proposer.py:206` | zero-head token proposal by copying from repeated context (KMP LPS) | repetition-heavy serving (RAG/code/agents) — huge speedup, no model to host | **ABSENT** — only a SOTA rung `internal/sotamatrix/ladder.go:159`; drafters are FastForward+MTP only; prior #3078 CLOSED | inspire → **FILED #5261** |
| Reasoning-end-gated constrained decode | `vllm/v1/structured_output/__init__.py:351,449` | hold the mask until `<think>` ends; trim end marker | reasoning models must reason unconstrained, then emit schema | **PARTIAL** — `internal/guideddecode/guideddecode.go` + `internal/agent/reasoning_strip.go` exist but UNWIRED | inspire → **FILED #5262** |
| Suffix-decoding trie drafter | `vllm/v1/spec_decode/suffix_decoding.py:81` | variable-length model-free spec from a trie of *past* outputs (cross-request) | repetitive agent output streams | ABSENT | inspire → noted (follow-on in #5261) |
| Per-position acceptance metrics | `vllm/v1/spec_decode/metrics.py:41` | accepted-count-by-draft-position curve to choose depth | operators tuning spec-decode depth | PARTIAL — `selfspecgov.go` tracks only a scalar accept-rate | noted (follow-on in #5261) |
| Dynamic spec length by batch size (DSD) | `vllm/v1/spec_decode/dynamic/utils.py:77` | index K by live concurrency | deep drafting only pays at low BS | **DIVERGENT** — fak's `selfspecgov.go` drives depth by accept-rate + **page-in economics**; slow-tier decode is low-concurrency, so BS-indexed K is not the binding lever | noted |
| fp8 KV-cache quantization | `vllm/model_executor/layers/quantization/kv_cache.py:42` | halve KV footprint + transfer bytes | KV dominates GPU memory at long context | PARTIAL — fak MLA (`deepseekv4kv`) already compresses via latent projection (bigger win); fp8 could still cut SSD/DRAM-tier + transfer bytes | noted (relates to milestone #2 / KV study epics) |
| Reliable seq-numbered replayable KV-event bus + TP quorum | `vllm/distributed/kv_events.py:287,124` | lossless catch-up-on-reconnect; only believe "stored" when all shards agree | keep an external router consistent with each engine | PARTIAL enrichment — `internal/engine/cacheevents.go` is recorder→Prometheus, no replay/quorum; the **quorum** maps cleanly onto DOS's "don't believe the workers" | noted (enrichment for #4303) |
| KV-cache events for external prefix router | `vllm/v1/core/block_pool.py:344` / `kv_events.py:51` | externalize block index for prefix-aware routing | multi-node cache-aware steering | PARTIAL but **already filed** #4303 (gossip prefix-cache directory), #3317 (LMCache events) | drop (dedup) |
| EPLB: measured load → replicate hot experts + balanced repack | `vllm/distributed/eplb/policy/default.py:75,274` | flatten MoE routing skew across devices | DeepSeek-scale MoE, hot experts starve GPUs | PARTIAL but **already filed** #3886; single-box residency analog is `expert_residency_lfu.go` (#3902/#5243) | drop (dedup) |
| Heterogeneous-vocab draft bridging | `vllm/v1/spec_decode/draft_model.py:37` | draft/target tokenizer mismatch mapping | reuse an off-the-shelf small model as drafter | PRESENT — **already filed** #4208 (polymodel TLI) | drop (dedup) |
| Stochastic rejection-sampler losslessness | `vllm/v1/sample/rejection_sampler.py:746` | lossless accept of longest valid draft prefix | any drafter can't change output distribution | PRESENT — **already filed** #4202 | drop (dedup) |
| Automatic prefix cache: chained content-hash block reuse | `vllm/v1/core/kv_cache_utils.py:596` | hash-keyed cross-request prefix reuse | shared system prompts / multi-turn | PRESENT — fak `radixkv` radix-tree prefix reuse | drop |
| cache_salt / extra_key prefix isolation | `vllm/v1/core/kv_cache_utils.py:558,579` | cross-tenant prefix isolation | multi-tenant no-leak reuse | PRESENT — `radixkv/namespace.go` (SGLang borrow #3889) | drop |
| KV offload tiering + on-disk atomic/config-hashed store | `vllm/v1/kv_offload/tiering/manager.py:159`, `file_mapper.py:96` | GPU→CPU→disk/S3/P2P hierarchy, crash-safe shared KV | multi-tier long-context reuse | PRESENT — fak `l3kv` (warmresume/store/router) + `cachemeta/placement.go` demote-instead-of-evict | drop |
| ARC adaptive eviction (ghost lists) | `vllm/v1/kv_offload/cpu/policies/arc.py:12` | self-tuning recency↔frequency, scan-resistant | mixed hot-frequent + one-shot-burst | **DIVERGENT** — fak `radixkv` chose SLRU+LFU+cost-aware; SLRU is already scan-resistant, ARC's self-tune is marginal here | noted |
| Disaggregated P/D connector (NIXL pull/push, lease/heartbeat, delay-free) | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:171` | prefill→decode KV handoff | disaggregated serving | PRESENT/PARTIAL — `cachemeta/nixl_lease.go` + epics #50, #3413 | drop (dedup) |
| Elastic EP scaling (standby groups, staged TCP barrier) | `vllm/distributed/elastic_ep/elastic_state.py:82` | reconfigure the parallel world without restart | autoscaling GPU MoE clusters | **DIVERGENT** — fak's elasticity is git-serialized worktree workers under lane leases, not SPMD group rebuild; different user | noted |
| Live in-place weight update (RLHF) | `vllm/distributed/weight_transfer/base.py:152` | swap weights without restart | online RLHF trainer↔inference loop | **DIVERGENT** — fak is not an RLHF loop; `stripeload` pages weights for a different purpose | noted |
| `collective_rpc` uniform SPMD control plane | `vllm/v1/executor/abstract.py:198` | one broadcast-gather channel for all worker ops | TP/PP/DP/EP orchestration | **DIVERGENT** — fak's control plane is dos/guard + fleet dispatch over git (trust-first), not SPMD | noted |
| LoRA two-tier LRU adapter cache + load-before-evict + on-demand resolver + Punica batched GEMM | `vllm/lora/model_manager.py:106` | serve N fine-tunes from a fixed GPU slot pool | SaaS hosting hundreds of fine-tunes | **DIVERGENT** — fak governs agents over a small model set, not a LoRA zoo; the pin-hot + load-before-evict discipline already exists in `expert_residency_lfu.go` + `expert_warmpins.go` | noted |
| Jump-forward / token-healing | `vllm/v1/structured_output/backend_xgrammar.py:141` (doc only) | emit forced-deterministic tokens with no model step | (designed, **unimplemented** in vllm) | **fak AHEAD** — `internal/model/fastforward.go` implements schema fast-forward drafting | noted |
| Chunked-prefill token-budget + recompute-preemption scheduler | `vllm/v1/core/sched/scheduler.py:503,558` | pack mixed prefill+decode under one budget; evict least-important on OOM | high-QPS multi-tenant GPU throughput | **DIVERGENT** — fak's worldview is governed single-stream turns + fleet orchestration; admission is per-turn cache-coverage pricing (#3893) + P/D planner (#2242) | noted |
| Deny-by-default network-facing plugin allowlist | `vllm/plugins/__init__.py:93` | minimize attack surface for route-adding plugins | multi-tenant untrusted plugins | PRESENT — fak manage/allowlist floor (`guard_allow_proposals`, trust-floor #5170) | drop |

## Filed

- **#5261** — feat(model/spec): model-free prompt-lookup (n-gram) drafter feeding the shipped verify-accept substrate — zero draft head.
- **#5262** — feat(guideddecode): gate the constrained-decode mask on reasoning-end — don't constrain the `<think>` block.

## Companions

- Skill: `.claude/skills/study-repo/SKILL.md` (this pass); hand-off to `.claude/skills/field-borrow/SKILL.md` for per-capability re-witness before acting.
- Related epics / leaves (borrow homes, not re-filed): spec-decode substrate #23 / #3197 / #5154 / #4208 / #4202 / #4102; prefix-routing #4303 / #3317 / #3893; disaggregation #50 / #3413; MoE EPLB #3886 / residency #3902 / #5243; guideddecode #26; KV study epics #3366 / #3900 / #3983 / #4207.

## Honest limits

- Witness is lexical + a snapshot (2026-07-18) — re-witness #5261/#5262 before implementing.
- vllm's worldview is my reconstruction from its code/defaults, not their testimony; DIVERGENT rows cite the config/non-goal they rest on.
- GPU-kernel and per-model-port surfaces were justified-skipped, not covered — a kernel borrow should go through `sota-check`, not this note.
