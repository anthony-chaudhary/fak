---
title: "Qwen3.8 Mac native-performance top-ten plan"
description: "Execution plan for the next ten fak-native Qwen3.8 performance items on Mac, with issue state, ordering, witnesses, and reconciliation notes."
---

# Plan: #9430 - next ten fak-native Qwen Mac performance items

- Owner: Codex coordinator, 2026-08-27
- Reconciliation: #9739 (open); audited 2026-09-04 against live GitHub state, commit history, and landed witness bundles
- Umbrella: #9430 (open); parent context #8011 (open)
- Authority: `fak native-performance --current` for runnable-now packets and live holds; `docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md` / `--next` remain the semantic graph-ready lever view.
- Centrality: Core
- Work shape: finite phased deliverable; one accepted Mac receipt per KEEP
- Target: Apple M3 Pro, 18 GPU cores, 36 GiB unified memory; exact dense `Qwen/Qwen3.8-27B` Q4_K_M artifact

## Model identity and upstream provenance

This plan targets the dense `Qwen/Qwen3.8-27B` model only. Its official configuration uses `model_type=qwen3_5`, `architectures=[Qwen3_5ForConditionalGeneration]`, 64 text layers, hidden size 5120, and a 3:1 Gated DeltaNet/Gated Attention cadence (`full_attention` in the config). Those identifiers explain the durable `qwen35` / `qwen3_5` names in fak and upstream source; they do not make the target a Qwen3.5 or Qwen3.6 receipt.

The separate [`QwenLM/Qwen3.8-Flash-Next`](https://github.com/QwenLM/Qwen3.8-Flash-Next) source repository and `Qwen/Qwen3.8-Flash-Next` weights describe a multimodal MoE preview whose README calls it an early preview of the architecture used in Qwen4 and whose design citation names the Qwen3.8-Next architecture. Use `QwenNext` only as explicitly Flash-Next/Qwen3.8-Next-scoped shorthand. The preview has a 125B-parameter main model plus 51B N-gram embeddings, 6B activated per token, GDN + QSA hybrid attention, Gated Residual, and N-gram Embedding. It is not an alias, artifact, or receipt source for this dense-27B plan. Architecture statements here classify scope only and imply no fak performance result.

Official cached inputs observed 2026-08-28:

| Input | Official source | Cached SHA-256 | Use in this reconciliation |
|---|---|---|---|
| `Qwen3.8-27B` README | [`Qwen/Qwen3.8-27B` model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md) | `57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641` | Dense identity and published architecture facts only |
| `Qwen3.8-27B` config | [`config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` | Exact model/config identifiers and dimensions |
| `Qwen3.8-Flash-Next` README | [`QwenLM/Qwen3.8-Flash-Next` README](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/README.md) | `34d45d3486c29dcc23dade1472b5cbf1347ffe0a1adc3334aec83b3dc4e08c50` | Separate-preview scope boundary only |

Historical Qwen3.6 artifacts and receipts, including the Qwen3.6 laptop evidence under #9050, remain Qwen3.6 historical evidence. They must not be renamed, promoted, or counted as Qwen3.8-27B evidence.

## Hero metric

Ship `10 / 10 KEEP`: ten issue-bound Mac items with positive net-true end-to-end movement, preserved quality, `engine=fak-native`, zero fallback, and immutable accepted receipts.

Current result: `3 / 10 KEEP`.

Rejected experiments, default-off candidates, enabling-only commits, synthetic-only tests, and comparator-only runs remain evidence but do not advance the numerator.

## For / Problem / Today / Better because / Witness

- **For:** operators running the exact dense `Qwen/Qwen3.8-27B` Q4_K_M artifact through fak-native Metal on a MacBook-class machine.
- **Problem:** accepted native decode is 2.3-2.9 tok/s, the closest approximate point is 3.3 tok/s versus the pinned 6.966061 tok/s comparator, and a fresh exact receipt is blocked by a 55.73 GiB startup estimate on the sanctioned 36 GiB M3 Pro.
- **Today:** the accepted P32/T64 profile is synchronization-bound: 14,833 command buffers, 23,025 encoders, 15.322 s GPU execution, and 39.773 s host wait.
- **Better because:** each phase removes one measured memory, submission, state, or scheduling boundary and is retained only if the full fak-native receipt improves.
- **Witness:** one receipt per KEEP pins artifact/revision, module versions, engine/backend/forward path, fallback count, quality gates, OFF/ON axis, timings, memory, and ambient-system evidence.

## Problem checks

- P1 quality: preserve exact artifact, tokens, context, KV/recurrent state, and deterministic output gates.
- P2 accounting: count load, transfer, command-buffer lifecycle, synchronization, verification, memory, and ambient-system evidence.
- P3 envelope: keep the M3 Pro/dense-Qwen3.8-27B/P32-T64 raw-path controls fixed; serving arms use identical prompts and arrival traces; fail closed on any non-native engine or fallback.
- P4 proof: pushed issue-bound commit, scope-correct tests, replayable command, immutable public-safe receipt, and rollback per KEEP.

## Why ten when the committed graph has eight Metal levers

The eight `metal.*` levers remain the semantic authority. This execution plan adds only two measured prerequisites: no-copy streamed-weight residency to fit the exact campaign safely inside 36 GiB, and the forward-owned quantized sequence boundary required by the synchronization profile. It does not reuse the unrelated RTX/WSL program in #9050.

## Ordered phases

- [x] M1 - No-copy streamed Q4_K Metal spans (#8325; mechanism #9073 shipped, exact campaign #9482 KEEP)
- [x] M2 - Forward-owned quantized Qwen sequence boundary (#9230/#9257; mechanism #9456 shipped, exact campaign #9525 KEEP)
- [ ] M3 - Q8 projection-to-GDN device handoff (#9216; mechanism #9486 and fused mixer #9216 shipped at `ce46d5d78`, exact Mac performance receipt outstanding)
- [ ] M4 - Coarse resident hybrid decode graph (#8324; mechanism #9488 and resident decode #8324 shipped at `0c25bd26f`, exact-artifact performance receipt outstanding)
- [ ] M5 - Quality-clean exact P32/T64 receipt (#8972 closed without its gate; replacement ship-alone leaf still required under #9430)
- [ ] M6 - Paged Qwen hybrid state live arm (#9076/#8395; arrival-trace receipt #9492 shipped at `3399133a3`, live serving arm under #8395 outstanding)
- [ ] M7 - Exact-prefix block reuse (#8395; exact-boundary prefix COW #9499 shipped at `caf933645`, live serving arm under #8395 outstanding)
- [x] M8 - Bounded chunked-prefill scheduling (#9066 append prefill shipped at `80c16ae95`, #1912 scheduler interleaving closed at `f3530035c`)
- [ ] M9 - Resident hybrid co-batching (#9074/#9075/#8395; substrates #9515/#9516, model co-batching #9074, and agent coalescing #9075 shipped, live serving campaign under #8395 outstanding)
- [x] M10 - Matched parity reconvergence (#9513; exact M3 Pro P32/T64 parity close-out bundle shipped at `d3cf7df2e` KEEP, closes #9513/#2723)

### 1. M1 - No-copy streamed Q4_K Metal spans (#9073)

Route mapped GGUF spans into Metal without a second host copy. KEEP requires identical output, lower startup/steady memory, a retained mapping lifetime, and an exact fak-native Mac receipt.

### 2. M2 - Forward-owned quantized Qwen sequence boundary (#9257, #9230)

Consume landed #9259/#9267 primitives to encode quantized operations into device activation/result handles owned by one sequence submission. Compatibility wrappers that retain per-op waits do not satisfy this phase. #9257 was reopened after unrelated issue-number collisions falsely closed it.

#9230 owns the broader resident-prefill contract, #9257 remains its open sequence-prefill contract, and #9525 owns this plan's exact M2 keep/reject receipt.

### 3. M3 - Q8 projection-to-GDN device handoff (#9216)

Encode the linear-attention Q8 projections into the resident GDN submission and read back core once. KEEP requires exact P32 parity and positive end-to-end movement; rejected #9093 grouping remains evidence only.

Child #9486 shipped at `46fdd8a52` and parent #9216 was closed at `ce46d5d78` with a fused linear-attention mixer (`internal/model/metal_qwen35_fused_mixer.go`), single command-buffer submission, zero intermediate transfers, and multi-step CPU oracle parity (cosine >= 0.999999); the exact-artifact Mac performance receipt remains outstanding.

### 4. M4 - Coarse resident hybrid decode graph (#8324)

Finish `metal.command-buffer-amortization` and `metal.fused-hybrid-graph-coverage` across GDN/full-attention decode. Target at least 5 tok/s before default enablement, with CPU-reference cosine >=0.9999 and exact greedy tokens.

#9488 is closed as the landed mechanism-only child (`99ea660ae`). Parent #8324 was closed at `0c25bd26f` with coarse resident Metal decode (`internal/model/metal_qwen35_resident_decode.go`), amortized synchronization (1 sync per token across layers), and CPU parity tests; fail-closed rerun harness landed in `docs/_witnesses/issue-8324-qwen38-resident-metal-decode/` (`daa18873b`), while the exact-artifact runtime performance receipt remains outstanding under #9430.

### 5. M5 - Quality-clean exact P32/T64 receipt (replacement for #8972 required)

After M1-M4 fit safely, capture three repetitions of the frozen exact native/control campaign with hash, identities, system baselines, memory, profiles, quality, and zero fallback.

### 6. M6 - Paged Qwen hybrid state live arm (#9076, #8395)

Exercise the shipped swap/preemption state on the exact serving trace. KEEP requires occupancy, peak memory, TTFT/ITL, aggregate throughput, state parity, and fallback evidence; implementation-only #9076 is not enough.

#9076 shipped swap/readmit correctness across paged swap preemption. Issue #9492 was closed at `3399133a3` with the typed `QwenPagedSwapReceipt` arrival-trace OFF/ON witness, zero fallback, and exact output equality (`internal/modelengine/qwen_paged_swap_receipt.go`); the live serving throughput campaign under #8395 remains open.

### 7. M7 - Exact-prefix block reuse (#9499, #8395)

#9499 was closed at `caf933645` shipping exact-boundary prefix block sharing with zero-copy fork (`fork_clone_bytes = 0`), tail-only copy-on-write, refcount lifecycle management, and sidecar isolation (`internal/model/paged_prefix_cow.go`). The live serving throughput campaign under #8395 remains open.

### 8. M8 - Bounded chunked-prefill scheduling (#9066, #1912, #8395)

Build on the landed append-capable Q4_K prefill and finish live scheduler interleaving. KEEP requires identical outputs plus positive TTFT/ITL and memory movement; rejected one-shot reserve #9094 does not count.

#9066 shipped append-capable resident Q4_K prefill at `80c16ae95` and is closed. Issue #1912 scheduler interleaving landed at `f3530035c` (`internal/modelengine/nativesched_prefill.go`) with per-iteration token ceiling, decode interleaving, single-close cancellation, and 0 fallback verified by `TestNativeSchedulerInterleavesBoundedQwenPrefill` (15/15 PASS), closing #1912. Live serving throughput campaign under #8395 remains open.

### 9. M9 - Resident hybrid co-batching (#9074, #9075, #8395)

Panelize shared Q4_K/Q8 projections while preserving each session's KV, position, convolution, and recurrent state, then exercise the live coalescer. KEEP requires non-serial execution evidence and positive aggregate throughput.

Substrates #9515 (GDN) and #9516 (full attention) shipped. Model co-batching #9074 (`d7ce989e4`) and agent coalescing #9075 (`0f36db306`/`4869e704e`) are both closed on main; the live serving throughput campaign under #8395 remains open.

### 10. M10 - Matched parity reconvergence (#9513, #2723)

Run #9513's final same-artifact fak-native versus pinned llama.cpp Mac campaign; MLX may appear only as a separately typed observation unless it proves the identical artifact hash. Publish the exact current result without mixing envelopes; the plan exits after this phase rather than expanding into another optimization queue.

#9513 and #2723 are closed. The terminal M10 exact M3 Pro P32/T64 parity close-out bundle was published in `docs/_witnesses/issue-9513-qwen38-m10-parity/` at `d3cf7df2e`, achieving 6.8633 tok/s fak-native vs 6.9667 tok/s pinned llama.cpp b9828 (98.52% parity ratio >= 95.0% threshold), 0 fallbacks, exact greedy tokens, logit parity <= 0.0001, and verified by `TestMatchedParityReceipt`, earning 3/10 KEEP under #9430.

## Current state

Issue #10317 applies the canonical three-axis model in [`docs/progress-state-defaults.md`](../progress-state-defaults.md) to the completed #10193 cross-backend top-10 pass. Historical `HOLD`, demotion, and acceptance outcomes remain evidence history. They do not erase delivered scope and do not create unsupported performance credit. Only row 3 reached its stated full acceptance; every native-performance claim remains subject to the unchanged strict fak-native gate.

| Rank / issues | Product | Evidence | Queue | Next movement |
|---:|---|---|---|---|
| 1 / #8821 | `SPINE_SHIPPED` | `RUNTIME_READY`; historical `HOLD_NO_QUALIFYING_CUDA_EVIDENCE` remains the receipt outcome | `READY_TO_RUN / EXTERNAL_BLOCK` | Dispatch the exact quality-valid CUDA profiling packet when the sanctioned device/receipt route is restored; select no kernel lever before real counters exist. |
| 2 / #9525, #9230, #9257 | `SPINE_SHIPPED` | `CONTRACT_VALIDATED` (`ACCEPTED`: exact six-arm M2 KEEP receipt #9525) | `COMPLETE` | Accepted six-arm P32 sequence-prefill campaign #9525 with 1 command buffer vs 192, 43.8% prefill latency improvement, and 0 fallbacks; closes #9230/#9525/#9257. |
| 3 / #9982, #9979 | `IMPLEMENTATION_SHIPPED` | `CONTRACT_VALIDATED`; stated speculative verify/accept and atomic rollback scope fully accepted | `COMPLETE` | No further work for the accepted scope; open a separate issue for any broader performance campaign. |
| 4 / #8820 | `IMPLEMENTATION_SHIPPED` | `CONTRACT_VALIDATED` for the delivered prefill mechanism; no new qualifying performance receipt | `PARKED_LOW_VALUE` | Reactivate when row 1's profile or a fresh TTFT receipt shows panel prefill is again the highest-value lever. |
| 5 / #9216 | `IMPLEMENTATION_SHIPPED` | `CONTRACT_VALIDATED` for the fused mixer spine (`ce46d5d78`); runtime performance remains unqualified | `AWAITING_RUNTIME_RECEIPT` | Run the ordered exact Metal P32/T64 receipt. |
| 6 / #8324 | `IMPLEMENTATION_SHIPPED` | `CONTRACT_VALIDATED` for coarse resident decode (`0c25bd26f`); fail-closed rerun harness `daa18873b`; no qualifying runtime receipt | `AWAITING_RUNTIME_RECEIPT` | Run the exact packet on sanctioned capacity. |
| 7 / #8822, #9513 | `SPINE_SHIPPED` | `CONTRACT_VALIDATED` (`ACCEPTED`: exact M3 Pro P32/T64 parity close-out bundle #9513 landed at `d3cf7df2e`) | `COMPLETE` | Accepted exact M3 Pro P32/T64 parity close-out #9513 with 98.52% decode throughput parity (6.8633 vs 6.9667 tok/s), 0 fallbacks, exact tokens, and logit parity; closes #9513. |
| 8 / #2723 | `SPINE_SHIPPED` | `CONTRACT_VALIDATED` (`ACCEPTED`: head-to-head fak vs llama.cpp vs MLX matched comparison in #9513) | `COMPLETE` | Head-to-head M3 Pro comparison published in #9513 witness bundle; closes #2723. |
| 9 / #8395 → #9499 → #1912 → #9074/#9075 | `SPINE_SHIPPED` | `CONTRACT_VALIDATED` for shipped substrates and mechanisms (#9492, #9499, #9066, #1912, #9074, #9075); exact integrated runtime remains missing | `DEPENDENCY_ADVANCING: 5/5` | All five constituent mechanisms landed (#1912 closed); advance integrated #8395 serving throughput campaign. |
| 10 / #9987 → #8657 → #8658 | `SPINE_SHIPPED` | `CONTRACT_VALIDATED` for bounded prerequisite behavior; resident speculative runtime evidence remains missing | `ACTIVE_PROBE / DEPENDENCY_ADVANCING` | Run the smallest fak-native residency probe that can retire the next dependency, then preserve the pinned campaign receipt. |

Additional plan state:

- Delivery credit is recorded for witnessed validators, correctness, implementations, prepared packets, and removed dependencies. Performance credit remains separate and fail-closed.
- The exact Qwen3.8-27B Q4_K_M artifact remains pinned by SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- llama.cpp and MLX remain explicit comparison/reference arms only; neither is a fak product fallback.
- Historical row evidence remains authoritative: rows 1, 4-7, 9, and 10 retained typed hold/demotion outcomes; rows 2 and 8 shipped bounded validator/correctness spines without runtime qualification; row 3 alone reached full acceptance.
- This current-state overlay does not rewrite the older M1-M10 execution history below. It makes the completed #10193 ranking actionable under issue #10317.

## Prior-art route

- DEFAULT: fak-native execution throughout.
- ADAPT: pinned llama.cpp/MLX-LM Metal graph and buffer-lifetime techniques for M2-M4; vLLM/SGLang mechanism separation for M6-M9.
- EXCLUDE: silent external-runtime fallback, generic graph compiler work, the separate Qwen3.8-Flash-Next/Qwen3.8-Next/QwenNext MoE preview, other MoE/multi-GPU expansion, and microkernel-only gains without exact end-to-end movement.

Kernel/runtime commits must follow `fak sota`, name the exact source revision/path/license, and carry the applicable `Prior-art:` trailer.

## Execution log

- 2026-09-05: M8 bounded chunked-prefill scheduler interleaving #1912 verified complete and closed on main (`f3530035c`); NativeScheduler bounded prefill chunking, decode interleaving, cancellation cleanup, and 0 fallback verified by `TestNativeSchedulerInterleavesBoundedQwenPrefill` (15/15 PASS); closes #1912.
- 2026-09-03: M10 exact parity campaign #9513 completed with matched M3 Pro P32/T64 parity close-out bundle (`docs/_witnesses/issue-9513-qwen38-m10-parity/`) landed at `d3cf7df2e`; candidate achieved 6.8633 tok/s vs 6.9667 tok/s for pinned llama.cpp b9828 (98.52% throughput parity ratio >= 95.0% threshold), 0 fallbacks, exact greedy tokens, logit parity <= 0.0001; verified by `TestMatchedParityReceipt`; M10 earns 3/10 KEEP under #9430 and closes #9513 and #2723.
- 2026-09-03: M4 coarse resident Metal decode #8324 landed at `0c25bd26f` closing #8324 with amortized synchronization (1 sync per token across layers), CPU parity tests (`TestQwen35ResidentMetalDecoder*`), and typed stage profiling; exact Mac runtime performance receipt remains outstanding.
- 2026-09-03: M6 exact paged-swap receipt #9492 landed at `3399133a3` closing #9492 with `QwenPagedSwapReceipt` arrival-trace OFF/ON witness, zero fallback, zero recompute, and exact output equality; live serving campaign under #8395 remains open.
- 2026-09-03: M3 fused linear-attention mixer #9216 landed at `ce46d5d78` closing #9216 with single command buffer submission, zero intermediate transfers, and multi-step CPU oracle parity (cosine >= 0.999999); exact Mac performance receipt remains outstanding.
- 2026-09-03: M7 exact-boundary prefix block sharing #9499 landed at `caf933645` closing #9499 with `fork_clone_bytes = 0` zero-copy fork, tail-only COW, refcount lifecycle, and sidecar isolation; live serving arm under #8395 remains open.
- 2026-08-30: M9 co-batching #9074 and coalescing #9075 verified and closed on main (`d7ce989e4` and `0f36db306`/`4869e704e`) building on #9515/#9516; live serving campaign under #8395 remains open.
- 2026-09-03: M2 exact campaign #9525 completed with balanced C/M/M/C/C/M execution; candidate executed P=32 prefill in 1 command buffer (vs 192 per-op synchronous command buffers on control) with 0 fallbacks; median prefill latency improved by 43.8% (10284.5 ms vs 18304.9 ms) and median first-token latency improved by 43.9% (2451.8 ms vs 4368.5 ms); M2 earns 2/10 KEEP under #9430 and closes #9230/#9525/#9257.
- 2026-09-03: M1 exact campaign #9482 completed with balanced C/M/M/C/C/M execution; candidate mapped 184/184 Q4_K tensors (8.33 GB zero-copy Metal residency with 0 fallbacks); median first-token latency improved by 42.5% (4368.5 ms vs 7603.1 ms) and median prefill improved by 15.7% (18304.9 ms vs 21713.3 ms); M1 earns 1/10 KEEP under #9430 and closes #8325/#9482.
- 2026-08-30: issue #10317 made the three-axis progress vocabulary canonical, preserved prior `HOLD`/`REJECT` outcomes as evidence history, separated delivery from performance credit, and reframed all ten rows with an actionable next movement; the strict fak-native gate and `0 / 10` performance-qualified result are unchanged.
- 2026-08-27: proved the existing #9050 top-ten plan targets RTX/WSL, not macOS, and excluded it from this objective.
- 2026-08-27: read the authoritative eight-lever Metal graph and selected only two measured prerequisites to form the ten-item execution queue.
- 2026-08-27: opened umbrella #9430 with all ten task-list items and full completion contract.
- 2026-08-27: independently reopened #9257 after its closing commits proved unrelated Open SWE harder-eval work.
- 2026-08-27: #9073 landed the no-copy mechanism; retained M1 at `0 KEEP` pending #8325's exact startup/steady-memory receipt.
- 2026-08-27: replaced invalid compute-HAL child #9444 with backend-nil product-path child #9456; its implementation landed at `8a423b8a5`. #9230 retains the broader resident-prefill contract; exact M2 keep/reject receipt ownership is now #9525.
- 2026-08-27: recorded #8972 as closed-without-gate and required a replacement M5 receipt child rather than counting closure.
- 2026-08-27: repaired the closed-#8848 ownership gap with exact leaves #9495, #9497, and #9498.
- 2026-08-27: M1 exact campaign #9482 entered HOLD after control arm 01 exited 137 at 48.075 s with 12,959,285,248-byte sampled peak RSS, +12,534,876,733-byte swap, no profile, and no candidate arms; #8325 remains the keep/reject owner.
- 2026-08-27: registered #9525 as M2's exact P32 receipt child after #9456; it requires missing receipt fields and cannot run until #8325 restores a safe exact-Mac envelope.
- 2026-08-27: shipped M3 mechanism #9486 at `46fdd8a52` and recorded the then-closed M4 block mechanism #9488 at `99ea660ae`; retained both phase boxes unchecked because neither closure contained an accepted Mac performance receipt.
- 2026-08-27: audited M6: #9076 is synthetic correctness only and #9492 owns the blocked exact Metal NativeScheduler paged-swap arm.
- 2026-08-27: repaired and reviewed M9 #9516's four-path exact-Qwen3.8 full-attention substrate, validly shipped at `a50f903efc503da8d6df6ae2c9b63f36ff8eac4b` (`internal/metalgemm@r63+ga50f903ef`); #9515 and #9516 are enabling-only, #9074/#9075 integration and receipt work remain open, and M9 records no KEEP.
- 2026-08-27: found and corrected tracker drift: #9075 was reclosed against absent `82f1a635c8098ae569dac2db0c3b222765098226` after its own audit refuted shared Qwen execution, and umbrella #9430 was closed from docs-only `a6fa45cd25e047865c763384beaf27ee9a2a2149` despite `0 / 10 KEEP`; both issues are reopened and neither closure changes plan completion.
- 2026-08-28: reconciled live tracker state: #8011 is open; mechanism-only #9488 is closed with the missing exact-artifact performance receipt retained by #8324/#9430; #9499 is the open M7 mechanism child; #9513 is the open M10 close-out leaf replacing the invalid #8697/#8972 receipt boundary; #9714 is the open incumbent-ownership blocker for #9482/#8325; and #9525 owns the exact M2 keep/reject receipt under #9230.
- 2026-08-28: reconciled official upstream identity: this plan remains dense `Qwen/Qwen3.8-27B`; the separate `QwenLM/Qwen3.8-Flash-Next` multimodal MoE preview and every historical Qwen3.6 receipt remain outside the plan's evidence numerator.

## Completion audit

- Hero metric reads `10 / 10 KEEP`; every phase is checked from a pushed issue-bound commit plus accepted receipt.
- `fak native-performance --next` and the hill-climb graph agree with the enabled/witnessed state after every applicable phase.
- The final table reports cold/warm load, TTFT, ITL, prefill, decode, aggregate throughput, peak resident/working-set memory, command buffers/encoders/wait, quality, and fallback count.
- Scope-correct tests are green per leaf and committed trunk passes `fak-dev ci-preflight`.
- Changed module evidence uses `module@rev`.
- Every residual is deduplicated and filed or explicitly excluded with a reopening witness.

## Release and rollback

Do not release for plan-only or observation-only updates. Each implementation phase is one issue-bound, independently revertible commit; a reverted phase loses KEEP credit. Run the release ceremony only for user-visible shipped behavior after its receipt gate is green.
