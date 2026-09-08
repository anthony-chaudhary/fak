<!-- fak-compute-key: vulkan-q4k-dense-mlp-fusion -->
# perf(compute): test fused Q4_K dense-MLP Vulkan dispatches

```routing
lane: compute
paths: ["docs/tickets/strix-performance/TICKET-vulkan-q4k-dense-mlp-fusion.md", "internal/compute/shaders/rmsnorm_q4k_matmul2.comp", "internal/compute/shaders/swiglu_q4k_matmul_add.comp", "internal/compute/build_vulkan.ps1", "internal/compute/vulkan_backend.h", "internal/compute/vulkan_shim.cpp", "internal/compute/vulkan.go", "internal/compute/vulkan_q4k_fusion.go", "internal/compute/vulkan_q4k_fusion_test.go", "internal/computebuild/vulkan.go", "internal/computebuild/computebuild_test.go"]
expected_steps: 8
```

Parent: [#11572](https://github.com/anthony-chaudhary/fak/issues/11572). Related evidence: [#11919](https://github.com/anthony-chaudhary/fak/issues/11919), [#12054](https://github.com/anthony-chaudhary/fak/issues/12054), [#12064](https://github.com/anthony-chaudhary/fak/issues/12064), and [#12175](https://github.com/anthony-chaudhary/fak/issues/12175).

GitHub: [#12207](https://github.com/anthony-chaudhary/fak/issues/12207).

## Parent context

#11572 owns the Vulkan performance campaign; the linked issues supply the Q4_K baseline, Q8 fusion precedent, and negative controls.

## Why this is next

The model-facing fusion calls and parity oracles already exist, leaving one isolated dispatch experiment rather than a graph redesign.

## Value

- **For:** the dense Qwen3.8 Vulkan Q4_K decode path.
- **Problem:** model code calls two fusion interfaces, but Q4_K executes RMSNorm + two projections and SwiGLU + down projection + residual as three dispatches each.
- **Better because:** two separately compiled, opt-in shaders can test whether eliminating four dispatch/barrier boundaries helps without changing the default or fallback.
- Centrality: Core
- P1: advanced - exercises the native dense decode critical path.
- P2: preserved - fak owns shader, dispatch, parity, and rollback.
- P3: advanced - reuses existing Q4_K packing and Q8 fusion plumbing.
- P4: preserved - software evidence cannot claim physical speedup.

## Current state

Classification: **PARTIAL**. Dense model execution already invokes `RMSNormMatMul2` and `SwiGLUMatMulAddInPlace`; Q8 has single-dispatch Vulkan shaders. The Q4_K branches in `internal/compute/vulkan.go` instead compose three dispatches. Q4_K parity tests and source-bound Qwen3.8 receipt validation exist, but no Q4_K fusion pipeline or selector does.

Coordination status (2026-09-08): a software-ICD prototype reached exact scalar-arm parity and one-dispatch counters, but was not landed because #12178 concurrently changed the same Vulkan host/shim seams. Re-audit those interfaces from current trunk before implementation; the prototype result is non-canonical and makes no physical claim.

#12064 is composition over Q2_K, #12175 is a standalone Wave32 GEMV, and #12054 is Q8 subgroup precedent; none is this experiment. Upstream RDNA3 evidence also warns that activation requantization can lose badly, so this candidate keeps f32 activations and directly decodes Q4_K weights.

## Core through-line

Resident f32 activation + Q4_K weights -> explicit default-off selector -> one fused RMSNorm/two-projection or SwiGLU/down/add dispatch -> resident outputs -> primitive parity and dispatch witness -> source-bound full-model KEEP/REJECT receipt -> retain or reject without weakening the unfused Q4_K fallback.

## Working spine

Q4_K dense-MLP inputs -> selected fused or unchanged composed Vulkan path -> resident output -> parity/counter witness -> durable receipt.

## Scope

Independently write two P=1 Q4_K shaders, optional pipeline creation and a pure selector, registry entries, and device-free plus Vulkan-skipping parity/counter tests. The default and explicit scalar override remain unfused.

## Gold-plating boundary

- No graph scheduler, command graph, model-loader, quantization-format, prefill, P>1, Wave32, or activation-requantization work.
- No replacement of existing Q4_K kernels, Q8 fusion, or scalar fallback.
- No default enablement, appliance access, tokens/s claim, or physical result in the software leaf.

## Prior art and license ledger

Observed 2026-09-08. Authoritative public source: `ggml-org/llama.cpp@5d806aa2575e01e126651fd69ab1ab6cefff861d` (commit event `2026-09-08T13:01:03Z`): [`mul_mat_vec.comp`](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp), [`rms_norm.comp`](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp), and [MIT license](https://github.com/ggml-org/llama.cpp/blob/5d806aa2575e01e126651fd69ab1ab6cefff861d/LICENSE). Khronos contracts: [compute shaders](https://docs.vulkan.org/spec/latest/chapters/dispatch.html) and [synchronization](https://docs.vulkan.org/guide/latest/synchronization_examples.html).

Disposition: **INSPIRE-ONLY / OPTIONAL-MODULE**. MIT design inspiration is legally usable, but this work copies no upstream source, comments, identifiers, or test vectors; implementation derives from the public Q4_K format and in-tree Apache-2.0 Q8 fusion architecture. Refresh if the pinned sources, Vulkan compiler, or target driver changes.

## Done condition

- [ ] [SW-VERIFIED] Both shaders compile for Vulkan 1.2, validate as SPIR-V, and appear in the portable shader registry.
- [ ] [SW-VERIFIED] Candidate selection requires `P=1`, both optional pipelines, and explicit opt-in; unset/false/unknown/P>1 selects unchanged Q4_K composition, and an explicit scalar override wins.
- [ ] [SW-VERIFIED] Crafted fixtures cover Q4_K metadata, both nibbles, tails, and `in={256,512,768}` with finite outputs.
- [ ] [SW-VERIFIED] Each primitive preserves exact greedy argmax, relative L2 `<=1e-4`, and cosine `>=0.99999` versus unfused Q4_K, with dispatch counters proving `3 -> 1` only in the candidate arm.
- [ ] [HW-WITNESSED] A clean committed revision binds source/archive/binary/shader/compiler/driver/device hashes and greedy token IDs, finite logits, CPU-model parity, fallback state, and raw post-warm samples.
- [ ] [HW-WITNESSED] KEEP requires exact greedy token parity, cosine `>=0.99999`, no fallback/validation errors, and at least 10% median decode improvement on both agreed Qwen3.8 shapes; otherwise record REJECT and retain default-off.

## Witness

The commands and binary criteria below are the reproducible software witness.

## Verifiable Witness

```powershell
pwsh internal/compute/build_vulkan.ps1 shaders
go test ./internal/computebuild -run TestVulkanShadersCompleteness -count=1
go test ./internal/compute -run 'TestVulkanQ4K(Fusion|RMSNormMatMul2|SwiGLUMatMulAdd)' -count=1
go vet ./internal/compute ./internal/computebuild
```

Physical witness is deliberately deferred. It must use the existing source-bound Qwen3.8 decode-receipt contract on the committed candidate and scalar control; no dirty-tree receipt can satisfy the gate.

## Done condition / witness

The software candidate is complete when the compilation, registry, selector, fallback, parity, and counter checks above pass. The issue remains open until the source-bound physical gate records KEEP or REJECT.

## Acceptance gate

All `[SW-VERIFIED]` checks must pass before landing. Default enablement additionally requires every `[HW-WITNESSED]` check and the two-shape 10% gate.

## Closure binding

The software commit uses `Ref #12207` and `(fak compute)`; it must not close the issue before the canonical committed-source physical decision.

## File:Line seams

- `internal/compute/vulkan.go:1329` and `:1501` - current Q4_K three-dispatch fallthrough.
- `internal/compute/vulkan_shim.cpp:1249` and `:1289` - Q8 fused pipeline precedent.
- `internal/compute/vulkan_q4k_test.go:269` and `:313` - current primitive parity oracles.

## Blast radius and fallback

Only `internal/compute` and `internal/computebuild` change. Model semantics, other backends/formats, and private appliance lanes are unaffected. Pipeline-build failure, absent opt-in, P>1, parity failure, or failed physical KEEP uses the existing unfused Q4_K path.

## Likely files

The routed paths above are exhaustive.

## Lane

`compute`; one public software experiment under #11572.

## Completion standard

experiment

## Work estimate

Estimate: 5 points.

## Overall completion contribution

Contribution: 5/100 points against the provisional #11572 campaign denominator.

## Definition of done

Every `[SW-VERIFIED]` criterion passes on committed source, the default and P>1 fallbacks remain unchanged, and the issue stays open for a separately witnessed physical KEEP or REJECT.
