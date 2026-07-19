# Study: llama.cpp — witnessed inspire-only borrows for fak

- **Repo:** `https://github.com/ggml-org/llama.cpp`
- **Pinned:** `@571d0d54` (`571d0d540df04f25298d0e159e520d9fc62ed121`), HEAD `model: rotate injected K/V cache for DFlash (#25823)`
- **License:** MIT (integrate is legally permissible with attribution+provenance; every borrow below is nonetheless **inspire** — clean-room, cite-only, no bytes vendored)
- **Date:** 2026-07-18
- **Method:** `/study … --deep`; 7 parallel sub-readers, one per load-bearing subsystem; on-axis witness against fak Go code + repo-scoped `gh` dedup.

## Fan-out coverage (7 deep readers)
1. ggml-backend scheduler + graph alloc + backend registry + RPC — `ggml/src/ggml-backend*.cpp`, `ggml-alloc.c`, `ggml-rpc/`.
2. GGUF loader + mmap + weight loading + split-shard — `src/llama-mmap.cpp`, `src/llama-model-loader.cpp`, `ggml/src/gguf.cpp`, `tools/gguf-split`.
3. Quantization k/i-quants + imatrix + fit-params — `ggml/src/ggml-quants.c`, `src/llama-quant.cpp`, `tools/imatrix`, `common/fit.cpp`, `tools/fit-params`.
4. KV cache (unified/iSWA/recurrent/hybrid/DSA/DSV4 + DFlash) — `src/llama-kv-cache*.cpp`, `src/llama-memory*.cpp`, `src/llama-kv-cells.h`.
5. MoE expert routing + partial CPU offload (`-ot` / `--n-cpu-moe` / `-ngl auto`) — `src/llama-model-loader.cpp`, `src/llama-model.cpp`, `common/arg.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-graph.cpp`.
6. CPU backend threading/NUMA/repack/AMX/SIMD dispatch — `ggml/src/ggml-cpu/{ggml-cpu.c,repack.cpp,amx,kleidiai,arch}`.
7. Server continuous batching + llama-batch + speculative + llama-bench — `tools/server`, `src/llama-batch.cpp`, `common/{speculative,ngram-*,reasoning-budget}.cpp`, `tools/{llama-bench,batched-bench}`.

## Completeness-critic residue (skipped, justified)
- **GPU vendor kernels** (CUDA/Metal/Vulkan/SYCL/OpenCL/HIP/CANN GEMM + flash-attn): read only via the placement/op-support seam, not the per-kernel code. Not clean-room-borrowable as Go and device-specific; the borrowable axis (placement, op-probe) was covered. Recent commits (OpenCL q6_k MoE, Adreno dp4a) are vendor-GPU-specific, off fak's CPU/GGUF/offload axis.
- **Tokenizer/vocab/unicode/grammar** (`llama-vocab`, `llama-grammar`, `json-schema-to-grammar`, jinja/peg chat parsers): orthogonal to the kernel axis; fak has its own stack (`ggml_tokenizer.go`, `polymodel/vocabmap`). Un-mined — a future structured-output pass could revisit.
- **Sampling zoo** (`llama-sampler.cpp`): PRESENT in fak; only the novel `reasoning-budget` forced-close noted below.
- **Multimodal / mtmd / tts / diffusion / training / ggml-opt / LoRA-adapter / cvector**: off the inference-kernel axis for this study.
- **`convert_hf_to_gguf.py` / gguf-py / quantize tool**: producer-side; DIVERGENT for a GGUF *consumer* (see worldview).

## Worldview (reconstructed, falsifiable)
llama.cpp serves **single-node local/edge inference** on commodity + Apple hardware, zero-dependency C/C++, optimizing **portability + memory footprint + first-token-on-any-hardware**. Defaults betray it: mmap on; one prebuilt binary adapts to the CPU/GPU at *runtime* (scored backend dispatch, `ggml-backend-reg.cpp:498`); CPU is the universal fallback backend (`ggml-backend.cpp:1736`); models bigger than VRAM are handled by CPU+GPU hybrid offload. Its user is **one person on one box** who downloads a quantized GGUF and wants max tok/s with no tuning.

Two divergences frame every borrow:
1. **Producer AND consumer.** llama.cpp *makes* quants (imatrix, per-tensor mixed-precision selection, i-quant codebooks) and *serves* them. fak is a **serving kernel** that consumes pre-quantized GGUF and does its *own* AWQ/EXL2 calibration (`internal/model/awq_group.go`). So llama.cpp's quantization-production stack is DIVERGENT — valuable to its "I make my own quant" user, out of fak's "I serve someone's quant" world. The one crossover was placement/budget fitting (not bits).
2. **Single local user vs audited fleet.** fak's `radixkv.BoundTree` carries a cross-model *materialization-binding* reuse guard llama.cpp's per-slot prefix cache lacks — fak is **stronger** there because a multi-model agent fleet demands it. Earned PRESENT+, not a borrow.

## Candidate table

### FILED (PARTIAL/ABSENT on-axis)
| Borrow | Source `path:line@571d0d54` | Axis | Worldview reason | Witness | Verdict | Filed |
|---|---|---|---|---|---|---|
| Per-tensor name-regex placement override | `src/llama-model-loader.cpp:1162` | arbitrary user `regex→device`, first-match-wins, finer than the expert predicate | local user: "attention fits VRAM, expert stacks don't — pin exactly these tensors" | PARTIAL — fak `onHost` hard-coded `isExpertWeight` (`internal/model/moe_offload.go:108`) | inspire | **#5280** |
| Graded `--n-cpu-moe N` + auto-fit split | `common/arg.cpp:2509`, `common/fit.cpp:527` | graded spill + measured-budget auto-search vs binary all/none | dial for "experts *almost* fit VRAM" without hand-tuning `-ngl` | PARTIAL — fak `CPUOffloadExperts` bool + binary `FitOnDevice` (`internal/ggufload/estimate.go:473`) | inspire | **#5281** |
| Model-free n-gram (prompt-lookup) drafter | `common/ngram-map.cpp:49`, `common/speculative.cpp:2357` | draft from token history, no model/head | can't afford a 2nd draft model; outputs quote the prompt | PARTIAL — fak has verify-accept substrate + head drafters only (`internal/model/config.go:791`) | inspire | **#5282** |
| Hadamard-rotation KV outlier smoothing | `src/llama-kv-cache.cpp:22,319,1857` | involutory rotation recovers low-bit KV accuracy at fixed bit-width | q4 KV quality collapses on a few outlier channels | ABSENT — fak KV-quant ladder (#2240) has no rotation | inspire | **#5283** |
| NUMA-aware weight-mmap policy | `src/llama-mmap.cpp:445`, `ggml/src/ggml-cpu/ggml-cpu.c:709` | `MADV_RANDOM`+no-prefetch under NUMA (first-touch); seq hints otherwise; autonuma warning | dual-socket box: default prefetch faults all pages onto one node | ABSENT — fak plain `MAP_SHARED`, per-expert WILLNEED only (`internal/model/mmap_unix.go:34`); fak HAS NUMA | inspire | **#5284** |
| Online SIMD-interleaved CPU repack | `ggml/src/ggml-cpu/repack.cpp:4726,4528,4573` | load-time interleave into vector-lane-contiguous blocks, width chosen by runtime CPU feature | one-time repack makes every token's matmul streaming-load-bound | ABSENT — fak repacks for GPU only (`internal/compute/cuda.go:378`), no CPU interleave | inspire | **#5285** |

### PRESENT-on-axis (dropped — witnessed, honest "we already have this")
| Borrow | Source | fak seam | Witness |
|---|---|---|---|
| Quantized K/V cache (`cache_type_k/v`) | `src/llama-kv-cache.cpp:231` | KV-quant ladder #2240/#3410/#4874 (filed/building) | PRESENT (tracked) — would dup |
| Prompt-cache save/restore (skip prefill) | `src/llama-kv-cache.cpp:1957` | `internal/model/paged_kv_transfer.go`, `kvcache.go:189` splice-skip-prefill | PRESENT |
| Continuous-batch slot reuse / prefix cache | `tools/server/server-context.cpp:1525` | `internal/radixkv/binding.go` BoundTree (+ cross-model binding guard) | PRESENT+ (fak stronger) |
| CPU `mul_mat_id` expert gather-scatter | `ggml/src/ggml-cpu/ggml-cpu.c:1622` | `internal/model/v4_expert_runtime.go` | PRESENT |
| Auto-fit KV **context** to host | `common/fit.cpp:328` | epic #1045 + #4361 (`--plan-json`) | PRESENT (tracked) |

### DIVERGENT (do-not-file — tradeoff + their user world stated)
| Borrow | Source | Tradeoff / their world |
|---|---|---|
| imatrix-weighted rounding + per-tensor mixed-precision quant-type selection + i-quant codebooks | `ggml/src/ggml-quants.c:1576`, `src/llama-quant.cpp:415` | llama.cpp is ALSO the quantizer tool (produces GGUF from one ftype label + a calibration file); fak CONSUMES pre-quant GGUF and does its own AWQ/EXL2 calibration (`internal/model/awq_group.go`). Their "I make my own low-bit quant" user needs it; fak's "I serve someone's quant" user does not. |
| GGUF defense-in-depth integrity (`offs+nbytes ≤ filesize` before deref) | `src/llama-model-loader.h:46` | The C++ risk is a silent OOB read on `mmap_addr+offset`. fak's Go slice bounds-check already *panics* on an OOB tensor offset — the security property is provided by the language; only error ergonomics (typed error vs panic) differ. fak checks alignment+int-overflow at `internal/ggufload/gguf_config.go:96`. |

### WORLDVIEW / note-only considerations (mostly map to existing epics — not auto-decomposed)
- **Automatic heterogeneous graph-partition by per-op support query** (`ggml/src/ggml-backend.cpp:1013`): fak uses an explicit per-weight predicate forward, not a generic graph-split brain. Architecture consideration under compute-placement epic **#4296**.
- **Op-offload override — run a GEMM on GPU with a host-resident weight** (`ggml/src/ggml-backend.cpp:908`): the "stream host experts to GPU per token" pattern; tracked by MoE-streaming **#5239/#3174/#5240**.
- **RPC distributed backend + content-addressed weight cache + graph-UID recompute** (`ggml/src/ggml-rpc/ggml-rpc.cpp:657,1094,1393`): pool GPUs across machines. Design input to distributed-compute epic **#4296**.
- **Two-pass measure-then-allocate compute arena + `needs_realloc` reuse** (`ggml/src/ggml-alloc.c:824`): size the compute arena once, reuse every token. Related **#4361**.
- **Scored cpuid SIMD backend dispatch** (`ggml/src/ggml-backend-reg.cpp:498`): runtime "best ISA that won't SIGILL"; composes with **#5285**.
- **Split-boundary async copy + N-way double-buffer H2D overlap** (`ggml/src/ggml-backend.cpp:1541`).
- **Chunked KV prefix reuse via position-shift on mid-prompt edits** (`tools/server/server-context.cpp:3168`): editing-heavy agentic workloads; sibling to **#5282**.
- **Context-shift on overflow (`n_keep`/`n_discard`)** (`tools/server/server-context.cpp:2837`): keep generating past `n_ctx`; adjacent **#1045**.
- **Reasoning-budget forced think-close by logit masking** (`common/reasoning-budget.cpp:59`): hard cap on CoT tokens (niche).
- **llama-bench pp/tg-separated mean±stdev discipline** (`tools/llama-bench/llama-bench.cpp:1497`): fak bench infra already extensive — note as PRESENT-ish.
- **Threadpool atomic-counter work-stealing + poll/sleep barrier + false-sharing-padded MoE counters** (`ggml/src/ggml-cpu/ggml-cpu.c:1397,3167,1580`): CPU-decode micro-opts; relate **#4623**.
- **Recurrent-memory rollback snapshots for spec-decode rejection** (`src/llama-memory-recurrent.cpp:181`): relates to the spec-decode family.

## Companions
- **field-borrow** — promote any note-only consideration to a filed leaf via its per-capability witness+file discipline.
- **Epics touched:** #5243 (MoE net-toks), #3900 (ktransformers MoE placement), #2236/#2240 (KV-quant ladder), #4296 (distributed-compute), #1045 (context auto-fit), #4623 (CPU decode), #2722 (mac offload).

## Honest limits
- The witness is lexical + a snapshot; each ABSENT was re-checked with a raw `Grep` of fak Go + a repo-scoped `gh` search (the broad global search leaked cross-repo results and was discarded). Re-witness before acting.
- "Worldview" is reconstruction, grounded in cited defaults/non-goals, not the authors' testimony.
