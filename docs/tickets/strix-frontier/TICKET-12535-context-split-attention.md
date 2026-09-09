# Context-split Vulkan decode attention

Tracking: [fak #12535](https://github.com/anthony-chaudhary/fak/issues/12535).
Observed 2026-09-09 against public source `2cb0d145d`.

## Current state

The control shader in `internal/compute/shaders/attention.comp` assigns one
workgroup to each query head and walks all 256-position context tiles serially.
`internal/compute/vulkan_shim.cpp` dispatches only the query-head workgroups.
This limits available parallelism during long-context, low-batch decode.

## Core through-line

Explicit opt-in -> independent `(head, context tile)` workgroups -> partial
online-softmax `(maximum, normalizer, weighted values)` -> deterministic merge
in increasing tile order -> output parity against the scalar CPU oracle.

The process-start selector is `FAK_VULKAN_ATTENTION_CONTEXT_SPLIT=1`. Other
values retain the control. Candidate scratch uses
`heads * max(1, ceil(context/256)) * (head_dimension+2) * sizeof(float)` bytes,
subject to integer, workgroup-count and device allocation limits. Its lifetime
extends through the ordered merge, including execution in a command batch.

## Gold-plating boundary

Keep the existing control as the default. No autotuner, permanent scratch pool,
grouped-query tile-sharing change, alternative runtime, receipt promotion,
model-quality claim, or performance claim belongs to this implementation leaf.
The evidence contract belongs to #12599; matched performance promotion belongs
to #12098. GPU parity on another device does not establish Halo performance.

## Prior art and placement

The published [Flash-Decoding mechanism](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
(2023-10-12, inspected 2026-09-09) splits the KV sequence and combines rescaled
partial softmax results. This change independently implements that mathematical
mechanism within fak's existing Vulkan backend; it copies no upstream source.
Disposition: optional candidate pending target-device evidence. Revisit when
the source-bound physical comparison is available.

Self-inspection found the serial attention control in public `fak`; private
serving policy does not replace this compute shader. The boundary tool maps
`internal/compute/shaders/attention.comp` to Gate 4, public compute kernels.

## Done condition

- [x] The explicit candidate splits context into 256-position workgroups.
- [x] Partial states merge in deterministic increasing tile order.
- [x] The control remains the default and keeps its output semantics.
- [x] Device-required parity covers non-divisible tile counts, finite output,
      maximum absolute error and cosine similarity against the CPU oracle.
- [x] Scratch remains alive through both immediate and batched dispatches.
- [x] A candidate refusal raises a backend error instead of returning an
      unwritten tensor or silently selecting the control.
- [x] Source/shader build, focused tests, boundary and leak checks pass.

## Witness

Set `FAK_VULKAN_REQUIRE_DEVICE=1` in the executing process and run:

```text
go test -tags vulkan ./internal/compute -run '^TestVulkanDecodeAttentionContextSplitParity$' -count=1 -v
```

A missing device or skipped test is not a pass. This ticket records numerical
delivery only; it awards no throughput or default-promotion credit.

## Validation record (2026-09-09)

The device-required test executed on a local AMD Radeon RX 7600 using the
Windows Vulkan backend, Vulkan SDK 1.4.350.0 and GCC-compatible cgo toolchain.
Shader compilation, the C++ static library and the Go Vulkan build succeeded.
The test launched its own process with the candidate and dispatch-profile
selectors, so the caller needed only the required-device flag.

All six numerical cases passed: contexts 257, 511 and 513; MHA, GQA and MQA;
head dimensions 32 and 256; immediate execution and a command batch. Every
call observed two attention dispatches. Maximum absolute error was at most
`8.04663e-7`, below the existing `1e-2` tolerance, and cosine similarity was
`1.00000000`. A 1025-wide head was rejected before dispatch and raised the
backend error instead of returning an output tensor.

The selector-unset control tests `TestVulkanAttentionApprox` and
`TestVulkanAttentionFlashShapes` passed through context 8192. The complete
non-cgo compute package tests, compute vet with and without Vulkan,
environment-configuration lint tests, companion boundary tests and staged
five-gate/leak audits passed.

This is a software correctness witness executed on a different physical GPU.
No Halo deployment, end-to-end model qualification, throughput comparison or
promotion is established. Target-device measurements remain with #12599 and
#12098.

## Likely files

- `internal/compute/shaders/attention.comp`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/vulkan_ops_vulkan.go` (propagates backend failure)
- `internal/compute/spirv/attention.spv` (generated, ignored build output)
- `internal/compute/vulkan_qwen35_sequence_test.go`

## Lane

```routing
lane: compute
paths: internal/compute/shaders/attention.comp, internal/compute/vulkan_shim.cpp, internal/compute/vulkan_ops_vulkan.go, internal/compute/vulkan_qwen35_sequence_test.go
expected_steps: 3
```
