# Study note — ktransformers/kt-kernel, fresh & deeper pass (2026-07-18)

**Upstream:** [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers), pinned **`0c2912a`**.
**License gate:** ktransformers Apache-2.0, fak Apache-2.0. Every candidate is Python/C++ → Go ⇒ **all `inspire`** (clean-room, source-cited, no bytes vendored).
**Prior pass:** epic **#3900** (pinned `7c021b4`, 2026-07-10) — a **caching-observability** lens that filed #3901/#3902 and enriched #3319/#3384/#3143/#2669, and explicitly scoped out CUDA/AMX/tensor code. This pass mines the **compute / kernel / operator-tooling** surfaces that lens left untouched.

## What changed upstream since the last study
The repo restructured: the integrated framework moved to `archive/` (the **kvc2** 3-tier prefix cache — block-hash prefix match, GPU→CPU→Disk background destaging, O_DIRECT async paging, restart-surviving serialized prefix tree — now lives there), **`kt-kernel` is the primary tree**, and new Day-0 models landed (MiniMax-M3, GLM-5.2, DeepSeek-V4-Flash). The 3-tier prefix cache is therefore legacy/archived here (the live path is in the serving fork), but remains a rich shipped reference.

## Method
6-subsystem parallel fan-out + a completeness pass:
cpu_backend/pipelining · expert placement/scheduling · kvcache/prefix tiering · quantized MoE kernels · CLI/operator UX · SFT/fine-tuning. Each candidate ablated **on-axis** against fak code (`fak_feature_query` + direct reads + `gh` dedup).

## Filed this pass (net-new, deduped)
| # | Borrow | Axis | fak verdict | Source@`0c2912a` | fak seam |
|---|--------|------|-------------|------------------|----------|
| **#5239** | Deferred-expert cross-layer CPU/GPU pipeline | run lowest-router-weight experts one layer late to hide CPU tail | ABSENT on-axis | `python/experts_base.py:347-455`, `operators/llamafile/moe.hpp:789-816`, `cpu_backend/task_queue.cpp:53-59` | `internal/model/moe_offload.go` (synchronous), `expert_readahead.go` (disk overlap only) |
| **#5240** | Native FP8 CPU MoE compute (no f32 expansion) | keep FP8 experts 1 B/elem resident, dequant per-tile | PARTIAL (fak native for Q4_K, expands FP8) — **sota-gated** | `operators/amx/la/amx_raw_kernels.hpp:324,816`, `fp8-perchannel-moe.hpp:92-108` | `internal/model/fp8_blockscale.go` (f32 at load) |
| **#5241** | Inference-hardware serve doctor | ISA-tier / VRAM-for-model / NUMA serve-readiness report | PARTIAL (fak doctor is fleet-ops) | `python/cli/commands/doctor.py:95-507` | `cmd/fak/serve_backend_preflight.go` (forward-path only), doctor family |
| **#5242** | Model-agnostic KV sizing from header | branch MLA/MHA/NSA read from config | PARTIAL (fak GLM-5.2 hardcode) | `python/cli/utils/kv_cache_calculator.py:34-121` | `internal/kvbudget/kvbudget.go` (`GLM52DSA` estimate) |

Enriched **#3886** (load-aware EP placement) with the witnessed hot/cold mechanism: frequency-ranked mask (`experts_base.py:21-72`), zero-copy mutable mask (`:282-296`), `should_skip_expert` (`common.hpp:255-258`), `physical_to_logical` remap (`common.hpp:48`), online update from live routing.

## Dropped — earned by ablation
- **Restart-surviving prefix reuse via persisted+relinked prefix tree** (`archive/.../kvc2/src/prefix.cpp:1155-1186,540-565`) → **DIVERGENT**: fak chose *regenerate-from-durable-text* (`KV = prefill(model, tokenize(text))`, epic **#3319**) — a deliberate alternative answer to the same "reuse survives restart" need.
- **CPU-AMX MoE fine-tuning / LoRA** (`python/sft/*`, `operators/amx/sft_moe.hpp`) → **WORLDVIEW-FINDING** (note-only): fak is inference + agent-fleet, not a training framework. Their user world: researchers LoRA-tuning DeepSeek-671B on 2–4×RTX 4090 (~40 tok/s, ~1.2 TB host RAM). *Claim check:* the README's "6–12× faster than ZeRO-Offload" is **not** in the code/docs (no ZeRO mention; real comparison is vs HuggingFace/Unsloth) — a pitch claim, not filed.
- **Quest/anchor training-free sparse-KV block retrieval** (`operators/kvcache/kvcache.h:475-502`) → **DIVERGENT**: fak has model-native learned sparse attention (GLM-DSA); Quest serves non-DSA models (a different user world).
- **NUMA intra-node work-stealing** (`cpu_backend/worker_pool.cpp:169-210`) → fak's `decode_numaschedule.go` + `numa_replica_store` already place workers per-node on byte-identical replicas (witnessed 2.61×); only dynamic straggler-balancing is a minor unfiled gap.
- **PRESENT on-axis (dropped):** runtime CPU-variant dispatch (fak: pure-Go single binary + `compute/cpuref` asm dispatch) · AVX-VNNI int4 + −128·Σw compensation (`quant_amd64_q4k` int8) · RAWINT4 native layout (fak Q4_K mmap, lazy decode) · block-hash prefix match (fak `radixkv` longest-prefix + SLRU) · MLA latent KV compression (`kvbudget` + GLM-DSA). Plus everything #3900 already dropped (`cachemeta` tiering, `l3kv` RestoreSpan, `blobfs`, `kvmmu`).

## Honest limits
- Witness is a 2026-07 snapshot (`0c2912a`) + lexical query + `gh` dedup; re-witness before acting.
- **0 fully-from-scratch borrows**: all four leaves are *deltas* on existing fak seams (`moe_offload`, `fp8_blockscale`, `serve_backend_preflight`, `kvbudget`).
- kt-kernel C++ kernel bodies were read at the technique level (loop shape, scale application, LUT), not transcribed; #5240 is deliberately sota-matrix-gated before any implementation.
