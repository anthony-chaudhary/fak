# Plan: #9430 - next ten fak-native Qwen Mac performance items

- Owner: Codex coordinator, 2026-08-27
- Umbrella: #9430; parent context #8011
- Authority: `docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md` and `fak native-performance --next`
- Centrality: Core
- Work shape: finite phased deliverable; one accepted Mac receipt per KEEP
- Target: Apple M3 Pro, 18 GPU cores, 36 GiB unified memory; exact Qwen3.8-27B Q4_K_M artifact

## Hero metric

Ship `10 / 10 KEEP`: ten issue-bound Mac items with positive net-true end-to-end movement, preserved quality, `engine=fak-native`, zero fallback, and immutable accepted receipts.

Current result: `0 / 10 KEEP`.

Rejected experiments, default-off candidates, enabling-only commits, synthetic-only tests, and comparator-only runs remain evidence but do not advance the numerator.

## For / Problem / Today / Better because / Witness

- **For:** operators running exact Qwen3.8-27B Q4_K_M through fak-native Metal on a MacBook-class machine.
- **Problem:** accepted native decode is 2.3-2.9 tok/s, the closest approximate point is 3.3 tok/s versus the pinned 6.966061 tok/s comparator, and a fresh exact receipt is blocked by a 55.73 GiB startup estimate on the sanctioned 36 GiB M3 Pro.
- **Today:** the accepted P32/T64 profile is synchronization-bound: 14,833 command buffers, 23,025 encoders, 15.322 s GPU execution, and 39.773 s host wait.
- **Better because:** each phase removes one measured memory, submission, state, or scheduling boundary and is retained only if the full fak-native receipt improves.
- **Witness:** one receipt per KEEP pins artifact/revision, module versions, engine/backend/forward path, fallback count, quality gates, OFF/ON axis, timings, memory, and ambient-system evidence.

## Problem checks

- P1 quality: preserve exact artifact, tokens, context, KV/recurrent state, and deterministic output gates.
- P2 accounting: count load, transfer, command-buffer lifecycle, synchronization, verification, memory, and ambient-system evidence.
- P3 envelope: keep the M3 Pro/Qwen3.8/P32-T64 raw-path controls fixed; serving arms use identical prompts and arrival traces; fail closed on any non-native engine or fallback.
- P4 proof: pushed issue-bound commit, scope-correct tests, replayable command, immutable public-safe receipt, and rollback per KEEP.

## Why ten when the committed graph has eight Metal levers

The eight `metal.*` levers remain the semantic authority. This execution plan adds only two measured prerequisites: no-copy streamed-weight residency to fit the exact campaign safely inside 36 GiB, and the forward-owned quantized sequence boundary required by the synchronization profile. It does not reuse the unrelated RTX/WSL program in #9050.

## Ordered phases

- [ ] M1 - No-copy streamed Q4_K Metal spans (#9073)
- [ ] M2 - Forward-owned quantized Qwen sequence boundary (#9257, #9230)
- [ ] M3 - Q8 projection-to-GDN device handoff (#9216)
- [ ] M4 - Coarse resident hybrid decode graph (#8324)
- [ ] M5 - Quality-clean exact P32/T64 receipt (#8972)
- [ ] M6 - Paged Qwen hybrid state live arm (#9076, #8395)
- [ ] M7 - Exact-prefix block reuse (#8395; ship-alone child required)
- [ ] M8 - Bounded chunked-prefill scheduling (#9066, #1912, #8395)
- [ ] M9 - Resident hybrid co-batching (#9074, #9075, #8395)
- [ ] M10 - Matched parity reconvergence (#8697, #2723)

### 1. M1 - No-copy streamed Q4_K Metal spans (#9073)

Route mapped GGUF spans into Metal without a second host copy. KEEP requires identical output, lower startup/steady memory, a retained mapping lifetime, and an exact fak-native Mac receipt.

### 2. M2 - Forward-owned quantized Qwen sequence boundary (#9257, #9230)

Consume landed #9259/#9267 primitives to encode quantized operations into device activation/result handles owned by one sequence submission. Compatibility wrappers that retain per-op waits do not satisfy this phase. #9257 was reopened after unrelated issue-number collisions falsely closed it.

### 3. M3 - Q8 projection-to-GDN device handoff (#9216)

Encode the linear-attention Q8 projections into the resident GDN submission and read back core once. KEEP requires exact P32 parity and positive end-to-end movement; rejected #9093 grouping remains evidence only.

### 4. M4 - Coarse resident hybrid decode graph (#8324)

Finish `metal.command-buffer-amortization` and `metal.fused-hybrid-graph-coverage` across GDN/full-attention decode. Target at least 5 tok/s before default enablement, with CPU-reference cosine >=0.9999 and exact greedy tokens.

### 5. M5 - Quality-clean exact P32/T64 receipt (#8972)

After M1-M4 fit safely, capture three repetitions of the frozen exact native/control campaign with hash, identities, system baselines, memory, profiles, quality, and zero fallback.

### 6. M6 - Paged Qwen hybrid state live arm (#9076, #8395)

Exercise the shipped swap/preemption state on the exact serving trace. KEEP requires occupancy, peak memory, TTFT/ITL, aggregate throughput, state parity, and fallback evidence; implementation-only #9076 is not enough.

### 7. M7 - Exact-prefix block reuse (#8395)

File or reconcile one ship-alone child before implementation. Run the isolated prefix arm with paged state fixed on and retain only a complete quality/latency/throughput/cache receipt.

### 8. M8 - Bounded chunked-prefill scheduling (#9066, #1912, #8395)

Build on the landed append-capable Q4_K prefill and finish live scheduler interleaving. KEEP requires identical outputs plus positive TTFT/ITL and memory movement; rejected one-shot reserve #9094 does not count.

### 9. M9 - Resident hybrid co-batching (#9074, #9075, #8395)

Panelize shared Q4_K/Q8 projections while preserving each session's KV, position, convolution, and recurrent state, then exercise the live coalescer. KEEP requires non-serial execution evidence and positive aggregate throughput.

### 10. M10 - Matched parity reconvergence (#8697, #2723)

Run the final same-artifact fak-native versus pinned llama.cpp/MLX Mac campaign. Publish the exact current result without mixing envelopes; the plan exits after this phase rather than expanding into another optimization queue.

## Current state

- M1-M4 are the memory/submission spine and are ordered by current dependencies, not by issue age.
- M5 is not runnable on the 36 GiB host until the startup envelope is reduced or a sanctioned >=64 GiB Apple-Silicon node is available.
- M6-M9 have partial/shipped building blocks but no accepted isolated Mac arm under this plan.
- M10 remains the close-out receipt.
- Rejected/default-off #9093, #9192, and #8833 experiments are excluded from KEEP credit.
- Managed worker inventory contains overlapping stale Metal/model worktrees; harvest against git truth before dispatching any overlapping phase.

## Prior-art route

- DEFAULT: fak-native execution throughout.
- ADAPT: pinned llama.cpp/MLX-LM Metal graph and buffer-lifetime techniques for M2-M4; vLLM/SGLang mechanism separation for M6-M9.
- EXCLUDE: silent external-runtime fallback, generic graph compiler work, MoE/multi-GPU expansion, and microkernel-only gains without exact end-to-end movement.

Kernel/runtime commits must follow `fak sota`, name the exact source revision/path/license, and carry the applicable `Prior-art:` trailer.

## Execution log

- 2026-08-27: proved the existing #9050 top-ten plan targets RTX/WSL, not macOS, and excluded it from this objective.
- 2026-08-27: read the authoritative eight-lever Metal graph and selected only two measured prerequisites to form the ten-item execution queue.
- 2026-08-27: opened umbrella #9430 with all ten task-list items and full completion contract.
- 2026-08-27: independently reopened #9257 after its closing commits proved unrelated Open SWE harder-eval work.

## Completion audit

- Hero metric reads `10 / 10 KEEP`; every phase is checked from a pushed issue-bound commit plus accepted receipt.
- `fak native-performance --next` and the hill-climb graph agree with the enabled/witnessed state after every applicable phase.
- The final table reports cold/warm load, TTFT, ITL, prefill, decode, aggregate throughput, peak resident/working-set memory, command buffers/encoders/wait, quality, and fallback count.
- Scope-correct tests are green per leaf and committed trunk passes `fak-dev ci-preflight`.
- Changed module evidence uses `module@rev`.
- Every residual is deduplicated and filed or explicitly excluded with a reopening witness.

## Release and rollback

Do not release for plan-only or observation-only updates. Each implementation phase is one issue-bound, independently revertible commit; a reverted phase loses KEEP credit. Run the release ceremony only for user-visible shipped behavior after its receipt gate is green.
