# Study: vLLM deep architecture & serving control-plane — witnessed borrows for fak

- **Repo:** https://github.com/vllm-project/vllm.git
- **Pinned:** `a56654d6de060495ff2db3b1d9ff0b187084d1a9` (`a56654d6`), HEAD dated 2026-09-02; latest commit `[K3 Perf] Enable DSV3 GEMM for inner-contiguous and row-strided tensors (#54565)`.
- **License:** Apache-2.0 — INSPIRE and INTEGRATE both permitted (all borrows below are **inspire**, clean-room Go, zero bytes vendored).
- **Method:** deep `/study-repo --deep`; 7 parallel subsystem readers over the previously-unopened subsystems and post-July delta; on-axis witness against fak code; 6 independently-shippable leaves filed.
- **Durable study receipt:** `study_0a1f15d4842961a69948677e534144ae5ec3c5266e861fe543cb852019a275a1`

## Fan-out coverage

| Reader | Subsystem | Load-bearing files read |
|---|---|---|
| 1 | Worker execution loop & memory management | `vllm/v1/worker/gpu_model_runner.py` (7738 lines), `vllm/v1/worker/gpu_input_batch.py`, `vllm/v1/worker/gpu_worker.py` (sleep/wake mode), `vllm/device_allocator/cumem.py`, `csrc/cumem_allocator.cpp`, `vllm/v1/worker/utils.py`, `vllm/v1/worker/gpu/README.md` (MRV2), `vllm/model_executor/offloader/prefetch.py` |
| 2 | Compilation stack & graph partitioning | `vllm/compilation/{backends,decorators,piecewise_backend,wrapper,cuda_graph,breakable_cudagraph,codegen,compiler_interface}.py`, `vllm/compilation/passes/pass_manager.py`, `passes/fusion/{rms_quant,act_quant,attn_quant}_fusion.py`, `passes/utility/fix_functionalization.py`, `vllm/config/compilation.py` |
| 3 | Distributed device communicators | `vllm/distributed/device_communicators/{cuda_communicator,custom_all_reduce,quick_all_reduce,pynccl,pynccl_wrapper,shm_broadcast,flashinfer_all_reduce,flashinfer_pcie_ipc_all_reduce,symm_mem,pynccl_allocator,shm_object_storage,all_reduce_utils,cpu_communicator}.py`, `csrc/custom_all_reduce.cuh`, `csrc/custom_collective_common.cuh` |
| 4 | Output pipeline & operational metrics | `vllm/v1/engine/{detokenizer,logprobs,output_processor,parallel_sampling}.py`, `vllm/logprobs.py`, `vllm/v1/metrics/{stats,loggers,prometheus,utils,reader}.py`, `vllm/v1/outputs.py`, `vllm/v1/worker/gpu/sample/prompt_logprob.py` |
| 5 | Design docs & worldview | `docs/design/{arch_overview,model_runner_v2,prefix_caching,hybrid_kv_cache_manager,nixl_kv_cache_lease,debug_vllm_compile,vllm_ir,optimization_levels}.md`, `docs/features/{disagg_prefill,sleep_mode,batch_invariance,automatic_prefix_caching,kv_offloading_usage}.md`, `docs/features/speculative_decoding/README.md`, `docs/benchmarking/cli.md`, `docs/usage/{v1_guide,reproducibility}.md` |
| 6 | KV connectors & tiered offloading | `vllm/distributed/kv_transfer/kv_connector/v1/{base,factory,multi_connector,offloading_connector,example_connector}.py`, `vllm/distributed/kv_transfer/kv_connector/v1/offloading/{scheduler,worker,config,common}.py`, `vllm/v1/kv_offload/{base,factory}.py`, `vllm/v1/kv_offload/cpu/{manager,gpu_worker,shared_offload_region,policies/base,policies/arc}.py`, `vllm/v1/kv_offload/tiering/{manager,spec,base,async_lookup}.py`, `vllm/distributed/kv_events.py`, `vllm/v1/core/kv_cache_utils.py` |
| 7 | Scheduler, engine core & spec-decode delta | `vllm/v1/core/sched/{scheduler,async_scheduler,request_queue,output}.py`, `vllm/v1/core/{kv_cache_manager,block_pool,single_type_kv_cache_manager}.py`, `vllm/v1/engine/{core,async_llm}.py`, `vllm/v1/spec_decode/{ngram_proposer,ngram_proposer_gpu,suffix_decoding,draft_model,dynamic/utils}.py`, `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, `vllm/v1/sample/sampler.py` |

### Completeness-critic residue (opened-or-justified)

Not opened, with justification:
- `csrc/libtorch_stable/moe/`, `csrc/libtorch_stable/quantization/`, `csrc/cpu/sgl-kernels/` — GPU/CPU kernel math; covered by SOTA matrix (`sota-check`).
- `vllm/models/` (203 files) — per-model architecture forward definitions; fak has its own native model tracks (#1026, #4867, #4033, #8757).
- `rust/src/` (218 files) — new Rust frontend; outside the Python engine and kernel architecture scope.
- `vllm/entrypoints/` — web server glue (FastAPI/Uvicorn); fak uses its own standard library HTTP/SSE gateway.
- `benchmarks/` — moved to `vllm bench` CLI; the methodology was read from `docs/benchmarking/cli.md`.

Verdict: no material subsystem left unopened for system-level fak borrows.

## Reconstructed worldview (who vLLM is built for)

vLLM optimizes **aggregate throughput for high-concurrency multi-tenant model serving on heterogeneous accelerators**. It achieves near-zero CPU overhead by pre-allocating persistent tensor structures, using a single shared CUDA graph pool across piecewise subgraphs, and running lock-free shared-memory rings between processes.

Its non-goals and refusals are explicitly documented:
1. **Refuses default determinism**: "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance" (`docs/usage/reproducibility.md:3`). Batch invariance and trace replay are opt-in, debuggability-only features.
2. **Refuses rarely-used serving complexity**: V1 permanently deleted CPU<>GPU KV swapping, `best_of`, and per-request logits processors in favor of pure recompute-on-preemption and startup-configured processors (`docs/usage/v1_guide.md:171-190`).
3. **Refuses false throughput claims**: explicitly documents that disaggregated prefill does not improve throughput (`docs/features/disagg_prefill.md:16`), and that prefix caching provides zero gain when decoding dominates.
4. **Refuses to own external infrastructure**: delegates production disaggregation to third-party connectors (Mooncake, NIXL) and production benchmarking to GuideLLM.

Where fak **diverges**: fak is an **audited agent kernel** that prioritizes session determinism, exact capability boundaries, and verifiable cache accounting over raw multi-tenant batch throughput. While vLLM discards determinism for speed, fak uses determinism as its core value proposition.

## Candidate table

Witness grain = narrow axis, `path:line@a56654d6`.

| Borrow | Source | Axis | Their-worldview reason | Witness (fak seam) | Verdict | Filed # |
|---|---|---|---|---|---|---|
| Incremental stop-string holdback buffer | `vllm/v1/engine/detokenizer.py:85,149` | Streaming emission without leaking partial stop sequences across chunks | Strict client parsers choke on split stop sentinels | **ABSENT** — `inkernel_render.go:22` checks suffix on complete text only; `native_serve.go:200` flushes deltas immediately | inspire | **#10719** |
| Slow-consumer delta coalescing under backpressure | `vllm/v1/engine/output_processor.py:48`, `vllm/outputs.py:173` | Decoupling model execution loop from slow client network writes | Worker threads must never block on slow network consumers | **ABSENT** — `native_serve.go:160-218` writes/flushes synchronously on the generator goroutine | inspire | **#10725** |
| Probe-request bystander interference measurement | `docs/benchmarking/cli.md:634` | Measuring scheduler head-of-line interference on background requests | Throughput averages conceal catastrophic tail latency spikes | **ABSENT** — `internal/bench` measures throughput/latency for homogeneous workloads, no bystander probe | inspire | **#10726** |
| MaxQueuedTokens TTFT-QoS queue admission cap | `vllm/config/scheduler.py:74`, `vllm/v1/engine/async_llm.py:282` | TTFT latency protection against long-prompt queuing floods | 10x 100k-token requests destroy TTFT even if sequence count is small | **PARTIAL** — `internal/gateway/admission.go:71` caps sequence count (`MaxWaiting`), but no queued token volume bound | inspire | **#10727** |
| Sliding-window CachingMetrics with running sums | `vllm/v1/metrics/stats.py:35,71` | Bounded, idle-safe O(1) cache hit-rate recency over long sessions | Lifetime hit-rates dilute acute regressions; idle periods decay timers | **PARTIAL** — `cachevalue.go:88`, `stream_metrics.go:30` compute cumulative lifetime sums | inspire | **#10728** |
| Stale-watermark reset for async KV store jobs | `vllm/v1/kv_offload/scheduler.py:1733` | O(1) asynchronous task invalidation on cache reset without locks | Canceling in-flight thread/worker jobs risks deadlocks and races | **ABSENT** — `cachemeta/external_invalidation.go` and `radixkv/remote_l3.go` have no async generation watermark | inspire | **#10729** |
| Per-request latency interval decomposition | `vllm/v1/metrics/stats.py:531` | Per-request queued→prefill→decode timeline with preemption fold | Full latency attribution without steady-state metric collection overhead | **PRESENT** — already filed and closed in **#4261** | drop (dedup) | — |
| Model-free prompt-lookup n-gram drafter | `vllm/v1/spec_decode/ngram_proposer.py:206` | Zero-draft-model token proposal from repeated context | Repetition-heavy serving (RAG/code) gets ~2x speedup with zero GPU overhead | **PRESENT** — already filed in **#5261** | drop (dedup) | — |
| Reasoning-end-gated constrained decode | `vllm/v1/structured_output/__init__.py:351` | Defer structured-output mask until `<think>` block terminates | Reasoning models must think unconstrained before schema enforcement | **PRESENT** — already filed in **#5262** | drop (dedup) | — |
| Dynamic spec decoding batch-size schedule | `vllm/v1/spec_decode/dynamic/utils.py:77` | Indexing speculative depth K by live batch size | Deep drafting only pays off at low concurrency on GPUs | **DIVERGENT** — fak's `selfspecgov.go` drives depth by accept rate + page-in economics; slow-tier decode is single-stream | drop (divergent) | — |
| Tag-scoped VMM allocation with sleep/wake | `vllm/device_allocator/cumem.py:229` | Reclaiming memory without process teardown via virtual-memory unmap | Freeing 90%+ VRAM during idle periods without restarting server | **DIVERGENT** — fak runs on CPU/host DRAM and uses `l3kv` demote-instead-of-evict and turn shedding | drop (divergent) | — |
| Fixed-slot SHM ring with per-reader flag bytes | `vllm/distributed/device_communicators/shm_broadcast.py:250` | Lock-free inter-process broadcast over shared memory | Python multiprocessing communication overhead elimination | **WATCH** — fak's control plane coordinates across git worktrees and OS processes via files/pipes, not SHM rings | drop (watch) | — |
| Real P2P capability check with persistent cache | `vllm/distributed/device_communicators/all_reduce_utils.py:245` | Real write/read-back probe across devices cached to disk | OS/driver APIs lie about peer access capabilities | **PRESENT** — fak's `preflight` and `windows-setup` perform actual verification before declaring readiness | drop (present) | — |
| One-trace compilation with fail-on-recompile | `vllm/compilation/wrapper.py:105,192` | Plan-stability enforcement by dropping guards and failing on re-trace | Re-tracing during production serving causes catastrophic latency spikes | **PRESENT** — fak's Go compiler enforces plan stability at build time; dynamic trace recompilation does not exist | drop (present) | — |

## Filed

- **#10719** — `feat(gateway): incremental stop-string holdback buffer for streaming responses`
- **#10725** — `feat(gateway): slow-consumer delta coalescing under streaming backpressure`
- **#10726** — `feat(bench): probe-request bystander interference measurement in serve benchmarks`
- **#10727** — `feat(gateway): MaxQueuedTokens TTFT-QoS queue admission cap`
- **#10728** — `feat(cachemeta): sliding-window CachingMetrics with running sums and idle-safe updates`
- **#10729** — `feat(l3kv): stale-watermark reset for asynchronous KV store and transfer jobs`

## Companions

- Study note: `docs/notes/CONCEPT-STUDY-VLLM-2026-07-18.md` (prior deep pass, filed #5261, #5262)
- Study note: `docs/notes/CONCEPT-STUDY-VLLM-M2-2026-07-10.md` (M2 KV cache value lens, filed #3893-#3897)
- Study note: `docs/notes/CONCEPT-STUDY-VLLM-DELTA-CLOSURE-2026-07-10.md` (convergence witness, filed #4261)
- Durable study receipt: `study_0a1f15d4842961a69948677e534144ae5ec3c5266e861fe543cb852019a275a1`
- Related epics: #2236 (vLLM superset), #23 (speculative decoding), #35/#36 (native scheduler & admission), #4254 (observability).

## Honest limits

- The study analyzed vLLM commit `a56654d6` (2026-09-02). Model Runner V2 (`gpu/model_runner.py`) is evolving rapidly; its draft-decode metadata protocol may mature further.
- GPU kernel math in `csrc/` was excluded per the study-repo scope rule; kernel-level borrows belong in `sota-check`.
- The six filed leaves are clean-room Go adaptations (`inspire`), adopting vLLM's architectural mechanisms to fak's agent-kernel seams without vendoring any Python code.
