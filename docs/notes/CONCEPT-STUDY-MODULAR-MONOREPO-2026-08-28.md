---
title: "study-repo: modular/modular full monorepo to fak (2026-08-28)"
description: "Pinned full-tree, full-history, forge-census, and FAK self-query study of Modular's compiler, MAX serving/model runtime, kernels, AsyncRT, Cache, and Support surfaces; 87 candidates, 68 dispatchable FAK issues."
---

# study-repo: modular/modular full monorepo to fak (2026-08-28)

This pass closes the broad-monorepo gap left by the narrower [Mojo 1.0 study](CONCEPT-STUDY-MOJO-2026-08-21.md). It pins [modular/modular](https://github.com/modular/modular) at **`1c9fd2e03331f77d3a1034127cb3700b7fa43c02`** (2026-08-28, MAX `26.6.0.dev2026082807` / Mojo `1.1.0.dev2026082807`), inventories the non-vendored tree, expands to the full reachable Git/tag history, captures every public forge census endpoint through the cutoff, and maps load-bearing mechanisms to exact FAK seams.

The outcome is **87 witnessed candidates** (**ABSENT 38; PARTIAL 42; PRESENT 7**; **DEFAULT 33; EXCLUDE 8; OPTIONAL-MODULE 12; RECIPE 1; WATCH 33**). **68** survived PRESENT/exact-owner deduplication and are live as [#9910](https://github.com/anthony-chaudhary/fak/issues/9910)-[#9977](https://github.com/anthony-chaudhary/fak/issues/9977) under parent [#9900](https://github.com/anthony-chaudhary/fak/issues/9900); **19** remain recorded-only. All 68 live bodies passed `fak dispatch issue-smallness-lint` and the strict issue contract, and read-back proved 68 unique markers, 68 receipt references, and the intended labels/milestone.

Durable receipts:

- source-to-decision: `study_a55850a776986e47ab91c858765a98e3cc4ce918086bb8ef380d5aa4f2325189`;
- source-to-outcome superseding receipt: `study_96c38d851458c92a9dea84574cff09471a9d13bf9d4a050eb076c5bb8b62416f`;
- forge index checksum: `sha256:a0752d13d096c945d50198126de5083a4c3878f00c84fe26bfc404322ba0f40e`.

## Why this is a different study from #8446

The prior note pinned `577b6b839efa11d750cdf264f1094954cc7d5b25` on 2026-08-21 and studied Mojo primarily as a language/toolchain. The current pin is **309 commits later** and includes the serving, scheduler, KV, speculative-decoding, model-architecture, kernel, compiler-pass, and failure-history surfaces that pass intentionally did not cover. The full clone now exposes **54,075 reachable commits and 18 release tags**, so candidate history is not limited to the initial shallow acquisition window.

## Inventory denominator

The committed machine map is [`docs/research/inventory/modular-modular.json`](../research/inventory/modular-modular.json). It walked **10688 files**, **1577 directories**, **1448362 text lines**, and classified **2987 runtime**, **4593 test**, and **759 documentation** files.

| subsystem | files | text lines | runtime | tests | docs |
|---|---:|---:|---:|---:|---:|
| `max` | 6003 | 879193 | 1897 | 2612 | 236 |
| `KGEN` | 2312 | 298327 | 609 | 1478 | 39 |
| `mojo` | 1358 | 123730 | 26 | 427 | 191 |
| `Support` | 403 | 49103 | 299 | 37 | 20 |
| `docs` | 249 | 67610 | 0 | 0 | 249 |
| `bazel` | 186 | 8038 | 54 | 17 | 5 |
| `AsyncRT` | 86 | 15523 | 63 | 11 | 6 |
| `Cache` | 21 | 3253 | 11 | 5 | 4 |
| `.` | 21 | 1327 | 4 | 0 | 6 |
| `.github` | 17 | 833 | 13 | 2 | 1 |
| `Init` | 13 | 1070 | 6 | 4 | 0 |
| `Licenses` | 4 | 13 | 0 | 0 | 1 |

The public forge capture is complete through `2026-08-28T23:59:59Z`: **7,071 normalized records** - 3,716 issues, 2,667 pull requests, 523 discussions, 151 labels, 14 releases, and no milestone rows - with page checksums and cross-endpoint non-atomicity evidence in [`modular-modular-forge-receipt.json`](../research/inventory/modular-modular-forge-receipt.json). Keyword-linked review covered 132 attention/KV, 251 GEMM/quantization, 1,485 serving/scheduling, 1,477 compiler/KGEN, and 119 MoE/speculation records; source-specific candidates also used file history and exact issue/PR read-back rather than keyword counts as proof.

## Overall design lessons

1. **The monorepo is deliberately layered.** The root map names KGEN as the compiler, `max/kernels` as the accelerator library, `max/python/max/serve` as the server, and `max/python/max/pipelines` as Python graph/model orchestration (`README.md:34-40@1c9fd2e03331f77d3a1034127cb3700b7fa43c02`). The transferable idea is not to rewrite FAK in Mojo; it is to keep serving policy, graph planning, runtime state, and device kernels separable while binding them through witnessed contracts.
2. **Optimization starts at explicit ownership boundaries.** Serving admission owns queue depth, request lifetime, batching budgets, and KV-transfer readiness; model code owns persistent recurrent state and draft/target layout; kernels own shape/dtype/device eligibility. FAK should copy that ownership clarity, not MAX's language/runtime stack.
3. **Fast paths keep a reference path and a shape gate.** The strongest kernel candidates pair device-specific scheduling/fusion with source tests covering tails, page boundaries, invalid indices, scale extremes, or aliasing. Every filed FAK kernel leaf therefore retains the existing path as oracle/fallback and forbids a gain claim without matched setup/transfer/workspace accounting.
4. **Graph compilation is a pipeline, not a capability bit.** KGEN's pass ordering, fixpoint behavior, conservative escape fences, and stable graph digests expose the delta between FAK's current graph-capture capability and an optimizer. [#9966](https://github.com/anthony-chaudhary/fak/issues/9966) is the minimal no-op pass-pipeline spine; the canonicalizer and symbol DCE follow before the more invasive SSA/interprocedural transforms.
5. **Persistent state is capacity, not incidental allocation.** GDN/hybrid state is preallocated by request slot, admitted by exact bytes per slot, and rolled back by indexed device state. This becomes [#9956](https://github.com/anthony-chaudhary/fak/issues/9956)-[#9959](https://github.com/anthony-chaudhary/fak/issues/9959), not a general allocator rewrite.
6. **Negative history is first-class optimization evidence.** Reverted host-callback elision and residual-recovery experiments become regression gates ([#9924](https://github.com/anthony-chaudhary/fak/issues/9924), [#9926](https://github.com/anthony-chaudhary/fak/issues/9926)); open proposals remain WATCH until merged code and FAK operating-envelope evidence exist.

## Highest-impact correctness findings

- **Qwen MTP concatenation order is currently reversed in FAK.** Modular's pinned `qwen3_5/mtp.py:181-185` and selector test require `[embedding, hidden]`; FAK `internal/model/qwen35_mtp_forward.go:102-107,133-178` and its tiny test pin `[hidden, embedding]`. [#9960](https://github.com/anthony-chaudhary/fak/issues/9960) carries the failing selector witness.
- **Qwen MTP pre-fusion bypasses the family `(1+w)` norm rule.** FAK derives `NormGain1p` for normal Qwen paths but calls plain `rmsnorm` at the MTP seam. [#9961](https://github.com/anthony-chaudhary/fak/issues/9961) requires a nonzero-weight oracle that distinguishes the two semantics.
- **Live Qwen vision decode lacks a three-axis M-RoPE cursor.** The current oracle intentionally collapses to text-only; [#9965](https://github.com/anthony-chaudhary/fak/issues/9965) binds image+text prefill and multi-request decode to an independent position oracle and typed refusal when the cursor is absent.
- **Grammar masks need a support-monotonicity invariant.** [#9925](https://github.com/anthony-chaudhary/fak/issues/9925) proves masking never makes a previously impossible padded-vocabulary token selectable.

## Optimization portfolio by layer

- **Serving / KV / speculative control - #9910-#9926 (17):** transaction cassettes, privacy-bounded schema telemetry, incremental UTF-8, composable budgets, decode TTL, KV-onload cordons and handles, cross-tier recency, TP-aware transfer, componentized memory, IndexK precision, acceptance calibration, and three negative-regression gates.
- **Core kernels - #9927-#9943 (17):** attention sinks/window tile skips, sparse/paged/quantized attention operands, FP8 MLA, fused RoPE/QKV/KV store, ragged batching, deterministic split-K, Apple FP4 routes, fused SwiGLU/RMSNorm, codec admission, and wide-vocabulary top-k/top-p.
- **Compiler/runtime/cache support - #9944-#9955 (12):** work donation/deadline scheduling, write-through/result diagnostics/inspectability, schema-approved key normalization, bounded telemetry labels, lazy logging, microspan scope, Welch advisory significance, toolchain-salted identity, and CPU cache topology.
- **Model architecture - #9956-#9965 (10):** GDN state capacity/rollback, capture-safe ragged offsets, two MTP correctness fixes, BF16 draft/quantized-head composition, shared-expert overlap, sparse MTP index reuse, and M-RoPE.
- **KGEN graph optimizer - #9966-#9977 (12):** deterministic pass spine, lifetime/escape buffer reuse, safe inlining, SROA, Mem2Reg, bounded SCCP, pure-branch canonicalization, function dedup, dead argument/result removal, symbol DCE, argument promotion, and budgeted loop unrolling.

## Full candidate matrix

`PRESENT` means the exact axis already exists; `PARTIAL` means a narrower FAK mechanism exists but not the candidate contract; `ABSENT` means the three-layer self-query found no on-axis implementation. A live issue is an experiment/development leaf, not proof that the candidate is faster or production-ready.

| track | row | candidate | FAK status | portfolio | route |
|---|---:|---|---|---|---|
| max-serving | 1 | Stream-count request bodies; do not trust Content-Length | PRESENT | EXCLUDE | recorded only |
| max-serving | 2 | One OpenAI-shaped error envelope for every served failure | PRESENT | EXCLUDE | recorded only |
| max-serving | 3 | Full HTTP transaction cassette, not only admitted-result replay | PARTIAL | OPTIONAL-MODULE | [#9910](https://github.com/anthony-chaudhary/fak/issues/9910) |
| max-serving | 4 | Normalize tool-call argument wire shape before adjudication | PARTIAL | DEFAULT | recorded only |
| max-serving | 5 | Privacy-bounded tool-schema conformance telemetry | PARTIAL | DEFAULT | [#9911](https://github.com/anthony-chaudhary/fak/issues/9911) |
| max-serving | 6 | Buffer split UTF-8 during incremental detokenization | ABSENT | DEFAULT | [#9912](https://github.com/anthony-chaudhary/fak/issues/9912) |
| max-serving | 7 | Bounded subprocess reap: join, TERM grace, then KILL grace | PRESENT | EXCLUDE | recorded only |
| max-serving | 8 | Bound scheduler pending depth so API ingress can shed load | PRESENT | EXCLUDE | recorded only |
| max-serving | 9 | Compose several independent batch budgets through one status fold | PARTIAL | OPTIONAL-MODULE | [#9913](https://github.com/anthony-chaudhary/fak/issues/9913) |
| max-serving | 10 | Per-request decode TTL independent of stall timeout | ABSENT | DEFAULT | [#9914](https://github.com/anthony-chaudhary/fak/issues/9914) |
| max-serving | 11 | Cordon decode until asynchronous KV onload is safe | PARTIAL | DEFAULT | [#9915](https://github.com/anthony-chaudhary/fak/issues/9915) |
| max-serving | 12 | Transfer handle reports direction and exact touched device blocks | PARTIAL | DEFAULT | [#9916](https://github.com/anthony-chaudhary/fak/issues/9916) |
| max-serving | 13 | Refresh external-tier recency on device-tier hits | ABSENT | DEFAULT | [#9917](https://github.com/anthony-chaudhary/fak/issues/9917) |
| max-serving | 14 | Attribute prefix hits to the tier that actually served them | PRESENT | EXCLUDE | recorded only |
| max-serving | 15 | Topology-derived UCX defaults with operator override precedence | ABSENT | DEFAULT | [#9918](https://github.com/anthony-chaudhary/fak/issues/9918) |
| max-serving | 16 | Resolve KV transfer strategy explicitly across TP changes | ABSENT | OPTIONAL-MODULE | [#9919](https://github.com/anthony-chaudhary/fak/issues/9919) |
| max-serving | 17 | Shared huge-block pool across heterogeneous KV leaves (Jenga) | PARTIAL | WATCH | [#9920](https://github.com/anthony-chaudhary/fak/issues/9920) |
| max-serving | 18 | Plan weights, activations, and signal buffers separately | PARTIAL | DEFAULT | [#9921](https://github.com/anthony-chaudhary/fak/issues/9921) |
| max-serving | 19 | Separate sparse IndexK precision from the main KV-cache dtype | ABSENT | DEFAULT | [#9922](https://github.com/anthony-chaudhary/fak/issues/9922) |
| max-serving | 20 | Explicit speculative acceptance portfolio plus synthetic calibration | PARTIAL | DEFAULT | [#9923](https://github.com/anthony-chaudhary/fak/issues/9923) |
| max-serving | 21 | Snapshot and restore grammar state around speculative draft walks | PRESENT | EXCLUDE | recorded only |
| max-serving | 22 | Negative knowledge: never skip the nominal host callback without an end-to-end OOM witness | PARTIAL | EXCLUDE | [#9924](https://github.com/anthony-chaudhary/fak/issues/9924) |
| max-serving | 23 | Negative knowledge: grammar masks must never widen model support | PARTIAL | DEFAULT | [#9925](https://github.com/anthony-chaudhary/fak/issues/9925) |
| max-serving | 24 | Negative knowledge: residual recovery can increase runaway truncation | PARTIAL | EXCLUDE | [#9926](https://github.com/anthony-chaudhary/fak/issues/9926) |
| kernels-core | 1 | Sink-aware CPU online softmax | ABSENT | WATCH | [#9927](https://github.com/anthony-chaudhary/fak/issues/9927) |
| kernels-core | 2 | Sliding-window mask bounds with whole-tile skip | ABSENT | WATCH | [#9928](https://github.com/anthony-chaudhary/fak/issues/9928) |
| kernels-core | 3 | Page-contained KV TMA sub-tiles | PARTIAL | WATCH | recorded only |
| kernels-core | 4 | Sparse logical-to-physical KV index remap with -1 preservation | ABSENT | WATCH | [#9929](https://github.com/anthony-chaudhary/fak/issues/9929) |
| kernels-core | 5 | Quantization-granularity-aware paged attention operand | PARTIAL | WATCH | [#9930](https://github.com/anthony-chaudhary/fak/issues/9930) |
| kernels-core | 6 | SM100 tensor-core FP8 sparse index scorer | ABSENT | WATCH | [#9931](https://github.com/anthony-chaudhary/fak/issues/9931) |
| kernels-core | 7 | Sparse MLA decode with all-FP8 KV and staged FP8->BF16 conversion | PARTIAL | WATCH | recorded only |
| kernels-core | 8 | Per-token-scale FP8 MLA prefill | ABSENT | WATCH | [#9932](https://github.com/anthony-chaudhary/fak/issues/9932) |
| kernels-core | 9 | AMD MI355X interleaved MHA prefill with lazy rescale | ABSENT | WATCH | recorded only |
| kernels-core | 10 | Fused RoPE + Q/K/V split + paged KV store | ABSENT | WATCH | [#9933](https://github.com/anthony-chaudhary/fak/issues/9933) |
| kernels-core | 11 | Ragged continuous-batching QKV matmul with scale granularity | PARTIAL | WATCH | [#9934](https://github.com/anthony-chaudhary/fak/issues/9934) |
| kernels-core | 12 | Persistent output-tile scheduler | ABSENT | WATCH | recorded only |
| kernels-core | 13 | Deterministic vs atomic split-K as an explicit contract | ABSENT | WATCH | [#9935](https://github.com/anthony-chaudhary/fak/issues/9935) |
| kernels-core | 14 | AMD warp-specialized producer/consumer ring-buffer GEMM | ABSENT | WATCH | recorded only |
| kernels-core | 15 | MI355X small-M weight-streaming matmul | PARTIAL | WATCH | recorded only |
| kernels-core | 16 | Apple M5 M=1 W4A16 NVFP4 GEMV | PARTIAL | WATCH | [#9936](https://github.com/anthony-chaudhary/fak/issues/9936) |
| kernels-core | 17 | Apple M5 cooperative-SMEM W4A16 prefill GEMM with crossover routing | PARTIAL | WATCH | [#9937](https://github.com/anthony-chaudhary/fak/issues/9937) |
| kernels-core | 18 | SM100 fused GEMM + SwiGLU three-phase epilogue | ABSENT | WATCH | [#9938](https://github.com/anthony-chaudhary/fak/issues/9938) |
| kernels-core | 19 | Active-expert-only grouped block-scaled GEMM with fused quantized SwiGLU | PARTIAL | WATCH | recorded only |
| kernels-core | 20 | CPU Q4_K/Q6_K 256-element packed block matmul | PRESENT | WATCH | recorded only |
| kernels-core | 21 | Per-channel grouped symmetric 4-bit codec | PARTIAL | WATCH | [#9939](https://github.com/anthony-chaudhary/fak/issues/9939) |
| kernels-core | 22 | Warp-per-row / warp-tiled RMSNorm dispatch | PARTIAL | WATCH | [#9940](https://github.com/anthony-chaudhary/fak/issues/9940) |
| kernels-core | 23 | Fused RMSNorm + residual add | ABSENT | WATCH | [#9941](https://github.com/anthony-chaudhary/fak/issues/9941) |
| kernels-core | 24 | Persistent bitonic top-k for k ~= N | ABSENT | WATCH | [#9942](https://github.com/anthony-chaudhary/fak/issues/9942) |
| kernels-core | 25 | Cluster-launched top-k/top-p sampling for very wide vocabularies | ABSENT | WATCH | [#9943](https://github.com/anthony-chaudhary/fak/issues/9943) |
| compiler-runtime | 1 | perf(modelengine): let blocked native drains donate scheduler work | ABSENT | OPTIONAL-MODULE | [#9944](https://github.com/anthony-chaudhary/fak/issues/9944) |
| compiler-runtime | 2 | feat(modelengine): carry session affinity into an execution-worker receipt | PARTIAL | OPTIONAL-MODULE | recorded only |
| compiler-runtime | 3 | perf(modelengine): benchmark a shared deadline heap for waiting native lanes | PARTIAL | WATCH | [#9945](https://github.com/anthony-chaudhary/fak/issues/9945) |
| compiler-runtime | 4 | feat(cache): promote an L2/L3 restore-on-access hit into the resident tier | PARTIAL | OPTIONAL-MODULE | recorded only |
| compiler-runtime | 5 | feat(vdso): add an opt-in write-through result-store chain | ABSENT | OPTIONAL-MODULE | [#9946](https://github.com/anthony-chaudhary/fak/issues/9946) |
| compiler-runtime | 6 | feat(vdso): normalize schema-declared nonsemantic path fields before hashing | PARTIAL | OPTIONAL-MODULE | [#9947](https://github.com/anthony-chaudhary/fak/issues/9947) |
| compiler-runtime | 7 | feat(vdso): preserve replay-safe producer diagnostics with cached payloads | ABSENT | OPTIONAL-MODULE | [#9948](https://github.com/anthony-chaudhary/fak/issues/9948) |
| compiler-runtime | 8 | feat(vdso): add read-only cache inspection and a guarded fixture put | ABSENT | RECIPE | [#9949](https://github.com/anthony-chaudhary/fak/issues/9949) |
| compiler-runtime | 9 | feat(cacheobs): attribute cache latency to a bounded pipeline-phase label | PARTIAL | DEFAULT | [#9950](https://github.com/anthony-chaudhary/fak/issues/9950) |
| compiler-runtime | 10 | feat(observability): add lazy structured KV logging at a kernel event seam | PARTIAL | OPTIONAL-MODULE | [#9951](https://github.com/anthony-chaudhary/fak/issues/9951) |
| compiler-runtime | 11 | refactor(metrics): expose one locked JSONL append primitive for telemetry producers | PARTIAL | DEFAULT | recorded only |
| compiler-runtime | 12 | feat(metrics): add a scope helper for paired microspan duration records | PARTIAL | OPTIONAL-MODULE | [#9952](https://github.com/anthony-chaudhary/fak/issues/9952) |
| compiler-runtime | 13 | feat(modelbench): report Welch significance beside median gain | PARTIAL | DEFAULT | [#9953](https://github.com/anthony-chaudhary/fak/issues/9953) |
| compiler-runtime | 14 | feat(vdso): salt tool-dependent cache identity with binary and toolchain witnesses | PARTIAL | DEFAULT | [#9954](https://github.com/anthony-chaudhary/fak/issues/9954) |
| compiler-runtime | 15 | feat(compute): report host CPU cache topology in the native execution envelope | ABSENT | DEFAULT | [#9955](https://github.com/anthony-chaudhary/fak/issues/9955) |
| kgen-models | M1 | Preallocate request-indexed GDN state pools instead of allocating per session | PARTIAL | DEFAULT | [#9956](https://github.com/anthony-chaudhary/fak/issues/9956) |
| kgen-models | M2 | Price recurrent state as bytes_per_slot x max_slots before admission | ABSENT | DEFAULT | [#9957](https://github.com/anthony-chaudhary/fak/issues/9957) |
| kgen-models | M3 | Roll recurrent state back entirely on device by shadowing live slots and replaying accepted rows | PARTIAL | DEFAULT | [#9958](https://github.com/anthony-chaudhary/fak/issues/9958) |
| kgen-models | M4 | Compute ragged prompt+draft offsets on the host to keep CUDA graph capture sync-free | ABSENT | DEFAULT | [#9959](https://github.com/anthony-chaudhary/fak/issues/9959) |
| kgen-models | M5 | Fix Qwen3.5 MTP fusion layout to embedding-first | PARTIAL | DEFAULT | [#9960](https://github.com/anthony-chaudhary/fak/issues/9960) |
| kgen-models | M6 | Apply Qwen's (1 + w) norm convention inside the MTP pre-fusion path | PARTIAL | DEFAULT | [#9961](https://github.com/anthony-chaudhary/fak/issues/9961) |
| kgen-models | M7 | Keep the MTP draft body BF16 while sharing a quantized target LM head | PARTIAL | DEFAULT | [#9962](https://github.com/anthony-chaudhary/fak/issues/9962) |
| kgen-models | M8 | Overlap an unfused shared expert with routed expert dispatch/combine on a side stream | ABSENT | OPTIONAL-MODULE | [#9963](https://github.com/anthony-chaudhary/fak/issues/9963) |
| kgen-models | M9 | Fuse logical routing, hot-expert remap, and logical-load accounting in one EPLB graph op | PARTIAL | DEFAULT | recorded only |
| kgen-models | M10 | Reuse one sparse top-k list across folded MTP query positions with an explicit shape gate | PARTIAL | WATCH | [#9964](https://github.com/anthony-chaudhary/fak/issues/9964) |
| kgen-models | M11 | Carry true three-axis M-RoPE positions from vision prefill into every decode row | ABSENT | DEFAULT | [#9965](https://github.com/anthony-chaudhary/fak/issues/9965) |
| kgen-models | K1 | Add a deterministic native graph-pass pipeline spine | PARTIAL | DEFAULT | [#9966](https://github.com/anthony-chaudhary/fak/issues/9966) |
| kgen-models | K2 | Reuse transient allocations by lifetime and escape, not merely by forward reset | PARTIAL | DEFAULT | [#9967](https://github.com/anthony-chaudhary/fak/issues/9967) |
| kgen-models | K3 | Inline graph functions with SCC and non-call-reference safety | ABSENT | WATCH | [#9968](https://github.com/anthony-chaudhary/fak/issues/9968) |
| kgen-models | K4 | Scalar-replace aggregate temporaries while preserving graph debug/provenance bindings | ABSENT | DEFAULT | [#9969](https://github.com/anthony-chaudhary/fak/issues/9969) |
| kgen-models | K5 | Promote graph-local mutable slots to region-carried SSA values | ABSENT | DEFAULT | [#9970](https://github.com/anthony-chaudhary/fak/issues/9970) |
| kgen-models | K6 | Add bounded region-aware sparse conditional constant propagation | ABSENT | DEFAULT | [#9971](https://github.com/anthony-chaudhary/fak/issues/9971) |
| kgen-models | K7 | Canonicalize pure structured branches and propagate branch facts | PARTIAL | DEFAULT | [#9972](https://github.com/anthony-chaudhary/fak/issues/9972) |
| kgen-models | K8 | Deduplicate structurally equivalent graph functions bottom-up | ABSENT | WATCH | [#9973](https://github.com/anthony-chaudhary/fak/issues/9973) |
| kgen-models | K9 | Remove dead function arguments/results with external and indirect-call fences | ABSENT | WATCH | [#9974](https://github.com/anthony-chaudhary/fak/issues/9974) |
| kgen-models | K10 | Eliminate unreachable graph symbols from exported roots and non-call references | ABSENT | DEFAULT | [#9975](https://github.com/anthony-chaudhary/fak/issues/9975) |
| kgen-models | K11 | Promote small nonescaping by-reference graph arguments to values/results | ABSENT | WATCH | [#9976](https://github.com/anthony-chaudhary/fak/issues/9976) |
| kgen-models | K12 | Unroll only statically bounded graph loops under an explicit budget | ABSENT | WATCH | [#9977](https://github.com/anthony-chaudhary/fak/issues/9977) |

## License and integration boundary

The repository root uses **Apache-2.0 with LLVM exceptions**, while `Licenses/LICENSE` carries a separate MAX Community License. Every candidate was routed by the cited file's own header or inherited root license; the source/test files independently sampled across the delegated packets resolved to the Apache-with-LLVM-exception boundary, with a root-governed test fixture as the sole headerless sample. Recommendations are predominantly **ADAPT** or **INSPIRE** because Mojo/Python/C++ interfaces and FAK's Go/native ownership differ. No source bytes were vendored, and nothing here authorizes a Modular runtime fallback.

## PRESENT and exact-owner exclusions

Nineteen rows were not re-filed. Representative exclusions include FAK's existing request body limit, error envelope, bounded shutdown, ingress shedding, tier-attributed cache metrics, grammar snapshot/restore, CPU Q4_K/Q6_K block math, existing speculative/KV owners, restore-on-access #1469, shared telemetry append owners, EPLB #3886, and DFlash/MTP parents #3197/#3078/#9819. Their mechanisms and source anchors remain in the candidate matrix and source receipt so a later refresh can detect a genuine delta without manufacturing a duplicate.

## Honest limits

- The inventory intentionally skips `.git` plus generated/target/vendor trees named in the machine map; it indexes the maintained source boundary, not copied tool output or third-party vendor payloads.
- The forge receipt proves complete pagination over issues, pulls, discussions, releases, labels, and milestones at one cutoff. Comments, reviews, review comments, timelines/events, and every linked commit were not exhaustively traversed for all 7,071 parents; candidate-bearing threads received targeted follow-up.
- The repository is now full-history and tag-complete for reachable Git objects, but the source pin is still a snapshot. Refresh on any named source-path change, a material FAK-seam change, or by **2026-09-11**.
- Hardware candidates are hypotheses until a sanctioned node produces a FAK-native receipt naming the executed kernel, matched model/dtype/shape/batch/context/clocks, quality oracle, setup/transfer/workspace costs, and three steady trials. No local-hardware absence is a terminal blocker.
- File-level license classification is engineering provenance, not legal advice.

## Reproduction

    fak study search --limit 5 "modular/modular exhaustive monorepo borrow study"
    fak dev study-forge validate --receipt <full-corpus.json>
    fak dev study-monitor --inventory-check --registry docs/research/monitored-repositories.json
    fak-dev issue contract --from-issues <68-live-issue-export.json> --strict-born-routed --strict-model-tier --strict-project-work --strict-scale --strict-witness --json

The committed standalone forge receipt is consumed by `study-monitor`; `study-forge validate` validates the full corpus object retained during capture. After the Modular row passed, `study-monitor` exposed an unrelated pre-existing registry schema defect at `repositories[33].inventory.source_classes[0]` (unsupported `paper_source`); the global monitor cannot be claimed green until that separate defect is repaired.
