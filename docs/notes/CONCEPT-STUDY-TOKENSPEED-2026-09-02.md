# TokenSpeed Deep Study — 2026-09-02

**Source:** https://github.com/lightseekorg/tokenspeed  
**Pinned revision:** `b174a3186d9a6eb3192389afbb25611a976eefc7`  
**License:** MIT (compatible with fak's Apache-2.0)  
**Study receipt:** `study_712b4c5e773ac0ffab3aa9c85ea45b4bec7ace85d029b61c1d97cace4fff2f77`  
**Parent epic:** [#10741](https://github.com/anthony-chaudhary/fak/issues/10741)  
**Study depth:** Full fan-out across all subsystems — scheduler (C++ FSM + Python execution), kernel (registry/selection), MLA, entrypoint, models

---

## Repository Map (what was opened)

| Subsystem | Path | Key files read |
|---|---|---|
| Scheduler (C++ control plane) | `tokenspeed-scheduler/csrc/` | `fsm/states.h`, `fsm/forward_events.h`, `scheduler/scheduler.h`, `cache/coordinator/cache_coordinator.h` |
| Scheduler (Python execution plane) | `tokenspeed-scheduler/python/` | bindings, tests |
| Kernel registry & selection | `tokenspeed-kernel/python/tokenspeed_kernel/` | `registry.py`, `selection.py`, `__init__.py` |
| MLA kernels | `tokenspeed-mla/python/tokenspeed_mla/` | `mla_decode.py`, `mla_prefill.py`, `fmha.py`, README |
| Entrypoint / AsyncLLM | `python/tokenspeed/runtime/entrypoints/engine.py` | `Engine`, `AsyncLLM`, `launch_scheduler_headless` |
| Event loop (scheduler worker) | `python/tokenspeed/runtime/engine/event_loop.py` | `EventLoop.event_loop`, `build_device_side`, `_commit_forward_results` |
| Models & layers | `python/tokenspeed/runtime/models/`, `layers/` | Attention backends, MOE, quantization, KV cache recipes |

**Completeness critic:** Every load-bearing subsystem was opened. The only skipped areas are third-party vendored code (CUTLASS, FlashInfer, DeepEP, etc.) which are external dependencies, not TokenSpeed's own inventions.

---

## Candidate Borrows (one technique per row)

Each candidate is grounded at `path:line@sha`, states the **single axis** it optimizes, gives TokenSpeed's worldview reason, and provides fak's on-axis witness.

| # | Technique (one line) | Source `path:line@sha` | Axis | Their worldview (why their users made them build it) | fak witness (PRESENT/PARTIAL/ABSENT/DIVERGENT) | Inspire/Integrate | Filed issue |
|---|---|---|---|---|---|---|---|
| 1 | **C++ FSM for request lifecycle with compile-time state safety** | `tokenspeed-scheduler/csrc/fsm/states.h:32`, `forward_events.h:42` @b174a31 | **Correctness-by-construction of request state transitions** — invalid transitions caught at compile time via `std::variant` + `InvalidTransitionHandler` | Research users re-run; ops users need auditable determinism. FSM makes "what state can this request be in?" a type, not a comment. | **PARTIAL-on-axis** — fak has `internal/guard` gate journal but no typed FSM for tool-call lifecycle. fak's guard journal is append-only log; TokenSpeed's FSM is a *state machine with typed transitions*. | **INSPIRE** (MIT) | [#10743](https://github.com/anthony-chaudhary/fak/issues/10743) |
| 2 | **Overlap scheduling via `in_flight_depth` + `results_in_flight` counter** | `tokenspeed-scheduler/csrc/fsm/forward_states.h:116`, `event_loop.py:915` @b174a31 | **CPU-GPU overlap without explicit pipelining code** — scheduler plans step N+1 while GPU runs step N; `results_in_flight` protects in-flight pages from retraction | Agentic workloads = short decode steps, high concurrency. Overlap hides CPU scheduler latency behind GPU compute. | **ABSENT-on-axis** — fak's `internal/engine` has no overlap scheduling; it's a single-threaded tool-call loop. | **INTEGRATE** (MIT) | [#10748](https://github.com/anthony-chaudhary/fak/issues/10748) |
| 3 | **Chunked prefill with `extend_prefix_lens` and per-chunk `input_ids`** | `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`, tests `test_fsm_and_scheduling.py:504` @b174a31 | **Memory-bounded prefill for arbitrarily long prompts** — each chunk only materializes its token slice; prefix lens tell attention what's already cached | Agentic coding = huge contexts (80K+). Chunking lets a 80K prompt run on 16K budget without OOM. | **ABSENT-on-axis** — fak has no prefill concept; it's a tool gateway, not an LLM engine. But the *pattern* (split large work into budgeted chunks with prefix tracking) is transferable to long context tool calls. | **INSPIRE** | WATCH (note-only) |
| 4 | **Capacity retraction with victim selection + recoverable snapshots** | `tokenspeed-scheduler/csrc/scheduler/scheduler.h:238`, `fsm/forward_events.h:174` @b174a31 | **Graceful degradation under memory pressure** — when prefill stalls, retract a victim (decode-origin first, then oldest epoch), snapshot KV to host if available, readmit later | Multi-tenant serving: a runaway request shouldn't block everyone. Host cache = fast readmission; no host cache = re-prefill. | **ABSENT-on-axis** — fak has no memory pressure concept for tool calls; it's stateless per-call. | **INSPIRE** (pattern for fair queuing under resource pressure) | WATCH (note-only) |
| 5 | **Role-based scheduling grammars (P/D/Fused) with `PlanBuild` composition** | `tokenspeed-scheduler/csrc/scheduler/scheduler.h:250`, `event_loop.py:976` @b174a31 | **Disaggregated prefill/decode without code duplication** — three grammars share admission/scheduling primitives but compose different batches | PD disaggregation is the scaling path for agentic workloads (prefill-heavy prompt, decode-heavy continuation). | **ABSENT-on-axis** — fak has no disaggregation; single process. | **INSPIRE** (pattern for composable scheduling policies) | WATCH (note-only) |
| 6 | **Two-tier cache coordinator (Device L1 + Host L2) with `PrefixProbe` admission** | `tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.h:103` @b174a31 | **Prefix cache hit rate at scale** — probe reads both tiers, admits only what fits, streams device→host asynchronously | Long contexts + multi-tenant = prefix sharing is the #1 lever. Host tier absorbs evicted prefixes for fast readmission. | **PARTIAL-on-axis** — fak has `internal/ctxmmu` for context shedding but no prefix cache. Different problem (context window vs KV cache). | **INSPIRE** | WATCH (note-only) |
| 7 | **Kernel registry with `FormatSignature`, capability gating, and per-family oracles** | `tokenspeed-kernel/python/tokenspeed_kernel/registry.py:156`, `selection.py:118` @b174a31 | **Hardware-aware kernel selection without if/else ladders** — capability (vendor, arch, features) + format signature + traits + oracle score = single ranked list | Heterogeneous fleet (NVIDIA/AMD, Hopper/Blackwell, FP8/BF16). Selection must be declarative, not procedural. | **PARTIAL-on-axis** — fak has `internal/model` registry but no format-signature or capability gating. fak selects models, not kernels. | **INTEGRATE** (pattern for declarative capability-aware selection) | [#10749](https://github.com/anthony-chaudhary/fak/issues/10749) |
| 8 | **Selection objectives (latency/throughput/portability/determinism/debug) as first-class enum** | `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:79` @b174a31 | **Explicit optimization target per call site** — `SelectionObjective.LATENCY` vs `THROUGHPUT` changes ranking without code change | Researchers want determinism; production wants throughput; CI wants portability. One enum, not scattered flags. | **ABSENT-on-axis** — fak has no selection objective concept. | **INTEGRATE** (pattern for explicit optimization targets) | [#10750](https://github.com/anthony-chaudhary/fak/issues/10750) |
| 9 | **MLA decode `fold_sq_factor` for small-head decode utilization** | `tokenspeed-mla/python/tokenspeed_mla/mla_decode.py`, README:65 @b174a31 | **Tile utilization in small-batch decode** — fold `q_seqlen` into head dim when `H*q_len <= 128` to keep BMM1 M-dimension full | Agentic = token-by-token (q_len=1) with small heads (64). Without folding, BMM1 M=64 wastes tiles. | **ABSENT-on-axis** — fak doesn't run MLA kernels. | **INSPIRE** (kernel-level technique) | EXCLUDE |
| 10 | **SMG-integrated headless scheduler (`launch_scheduler_headless` + msgpack ZMQ)** | `python/tokenspeed/runtime/entrypoints/engine.py:638` @b174a31 | **External control plane ownership** — SMG (or any orchestrator) drives scheduler directly; no in-process tokenizer/detokenizer | Fleet operators want to own the control plane; TokenSpeed provides the data plane. Headless mode = clean separation. | **PARTIAL-on-axis** — fak's `fak serve` + `fak agent` is similar but fak *is* the control plane. TokenSpeed inverts: scheduler is a *library* driven by external SMG. | **INSPIRE** (architecture pattern) | WATCH (note-only) |
| 11 | **Per-request `reserve_num_tokens_in_next_schedule_event` for decode KV reservation** | `tokenspeed-scheduler/csrc/fsm/forward_states.h:204`, tests `test_update_reserve_num_tokens.py` @b174a31 | **Proactive KV page reservation for decode** — tell scheduler "next decode step needs N tokens" so pages are ready before GPU launch | Decode step latency = page alloc latency. Reservation moves alloc to CPU overlap window. | **ABSENT-on-axis** — fak has no KV cache. | **INSPIRE** (proactive resource reservation pattern) | WATCH (note-only) |
| 12 | **EPD (Encode-Prefill-Disaggregation) admission with async embedding receive** | `python/tokenspeed/runtime/epd/prefill_admission.py`, `event_loop.py:489` @b174a31 | **Multimodal prefill without vision tower on prefill node** — encode workers compute embeddings, ship via Mooncake; prefill admits only when embeddings arrive | Vision-heavy agents (Qwen-VL, GLM-4V). Prefill node shouldn't run vision tower; encode workers are cheaper GPUs. | **ABSENT-on-axis** — fak has no multimodal. | **INSPIRE** (async dependency admission pattern) | WATCH (note-only) |

---

## Worldview Findings (no code to copy, but reframes fak's roadmap)

| Finding | Evidence | Fak implication |
|---|---|---|
| **Agentic workloads = short decode, high concurrency, huge contexts** | README: "agentic workloads with high request concurrency, short decode steps, strict TTFT" | fak's tool-call model is already agentic; but fak assumes *long* tool calls (seconds), not *many short* calls (ms). |
| **Control plane / execution plane separation is the architectural differentiator** | PyTorch blog quote: "first to separate control plane from execution plane" | fak *is* the control plane. TokenSpeed inverts: scheduler is a *library*. Could fak expose its scheduler as a library for external orchestrators? |
| **Kernel selection must be declarative + capability-aware** | `selection.py`: capability + format signature + oracle = ranked list | fak's model selection is procedural (`internal/model`). A declarative registry would help multi-provider routing. |
| **Overlap scheduling is the default for agentic** | `event_loop.py`: `in_flight_depth=1` is default for non-PP | fak's tool calls are sequential. If fak ever runs local models, overlap is table stakes. |
| **Retraction + host cache = graceful degradation, not failure** | `RetractEvent` with `has_recoverable_snapshot` | fak has no graceful degradation; tool calls either succeed or error. Could apply to rate-limited providers. |

---

## Dismissed Candidates (earned by ablation)

| Candidate | Why dismissed |
|---|---|
| MLA-specific kernels (CuTe DSL, FP8 quantized) | Axis = "MLA attention on Blackwell". fak doesn't run attention kernels. DIVERGENT: fak's path is tool gateway, not kernel library. |
| PD cache transfer protocol (Mooncake) | Axis = "KV transfer between disaggregated prefill/decode nodes". fak is single-process. DIVERGENT: different deployment model. |
| Speculative decoding (Eagle, DSpark, MTP) | Axis = "draft model + verification". fak doesn't run draft models. ABSENT but not on fak's roadmap. |
| CUDA graph capture with `breakable_cuda_graph.py` | Axis = "graph capture with dynamic shape escape hatches". fak doesn't capture graphs. INSPIRE only if fak ever runs local models. |

---

## Registration

- **Study note:** `docs/notes/CONCEPT-STUDY-TOKENSPEED-2026-09-02.md` (this file)
- **INDEX.md line:** `docs/notes/CONCEPT-STUDY-TOKENSPEED-2026-09-02.md — TokenSpeed deep study (scheduler FSM, overlap, kernel registry, MLA) — 2026-09-02`
- **Companions:** `field-borrow` (for each filed issue), `sota-check` (for kernel candidates)
- **License gate:** MIT → Apache-2.0 compatible. Direct port allowed with attribution.

---

## Next Actions

1. File GitHub issues for each **PARTIAL/ABSENT** candidate above (one issue per technique, not a monolith)
2. For kernel registry candidates (#7, #8), route through `field-borrow` witness step
3. For scheduler FSM/overlap candidates (#1, #2), scope as `internal/engine` enhancements
4. Update `docs/research/monitored-repositories.json` with TokenSpeed entry