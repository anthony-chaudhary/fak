# fix(compute): the Vulkan Q2_K panel matmul (P>1) matches the CPU oracle

## Parent context

Split out of TICKET-03d (#13776), the Vulkan-tagged suite reds, under parent #13761, serving #13668.
TICKET-03d forbids kernel changes, so this real kernel bug is tracked here.

<!-- fak-model-key: vulkan-q2k-panel-matmul-shared-race -->

```routing
lane: compute/vulkan
paths: ["internal/compute/shaders/q2k_matmul.comp", "internal/compute/vulkan_qwen35_sequence_test.go", "docs/tickets/waived-verification-2026-10-09/TICKET-03e-q2k-panel-matmul-shared-race.md"]
expected_steps: 2
```

Process cause: verification-gap

## Scope class

single-package

## Current state

The red is the q2_k subtest of `TestVulkanQwen35SequenceQuantizedPanelsMatchCPU`. Evidence:

- [HW-WITNESSED] Strix Halo (RADV STRIX_HALO) at 42dc8adb624: q2_k max_abs_delta is 4.5.
- [HW-WITNESSED] Local AMD Radeon RX 7600 (AMD proprietary driver) at f2be2792ab2: max_abs_delta 4.5499, with
  `token=0 row=0 got=3.0519106 want=-1.4979788`. The f32, q8_0 and q4_k panels in the same test pass.

Suspected cause, from reading the source and not yet confirmed by a fix:

- When P>1, `fvk_q2k_matmul_f32` (`vulkan_shim.cpp`) dispatches `K_Q2K_MATMUL`. When P=1, it uses `q2k_matvec`,
  whose parity tests pass.
- In `shaders/q2k_matmul.comp`, `plainMatMul()` has each invocation fill the workgroup-`shared float fusedX[256]`
  with its own token's input slice, with no barrier. Invocations in one workgroup map to different
  `token = globalIdx / outDim`, so they overwrite each other's input before `q2kDot` reads it.
- The fused paths (`fusedSwiGLUAdd`, `fusedRMSNormMatMul2`) share one input per workgroup and use barriers.
  That matches their passing tests.

## Problem

Prefill and batched Q2_K projections on Vulkan return wrong values whenever P>1.

## Why this is next

It is the only real kernel red left in the Vulkan-tagged compute suite after TICKET-03d's test-isolation fixes.
Q2_K is the pinned DeepSeek-V4.1-Flash quantization.

## Working spine

1. Make `plainMatMul` read X directly, or into invocation-private storage, instead of using unsynchronized shared memory.
2. Rerun the panel test and the Q2_K matvec/fused tests on silicon.

## Core through-line

The Q2_K panel output equals the CPU scalar oracle within the existing 2e-3 + 2e-4*|want| bound.

## Gold-plating boundary

No tolerance loosening, and no performance retuning beyond what the fix needs.

## Done condition / witness

- [ ] [HW-WITNESSED] `TestVulkanQwen35SequenceQuantizedPanelsMatchCPU/q2_k` passes on a registered Vulkan device.
- [ ] The Q2_K matvec and fused tests stay green.

## Witness

`./compute.test -test.run 'TestVulkanQwen35SequenceQuantizedPanelsMatchCPU|TestVulkanQ2K' -test.v`, with
`FAK_VULKAN_SPIRV` set and `FAK_VULKAN_REQUIRE_DEVICE=1`.

## Definition of done

Both boxes are ticked, and the resolving commit cites this issue.

## Acceptance gate

The CPU `internal/compute` tests and the Vulkan-tagged suite stay green.

## Closure binding

The resolving commit cites this issue and carries the `compute` leaf in its subject.

## Likely files

- `internal/compute/shaders/q2k_matmul.comp`: `plainMatMul`.

## Lane

compute/vulkan

## Blast radius and affected lanes

One shader. It covers Q2_K prefill and batched projections.

## Problem frame

- Centrality: Core (a Q2_K serving-path correctness bug)
- P1 Context: preserved - no change to context, KV or prompt handling.
- P2 Net value: advanced - Q2_K prefill returns correct values on Vulkan.
- P3 Adaptation: N/A - no external mechanism is involved.
- P4 Operations: preserved - no operational surface changes.

## Expected steps

2

## Work estimate

Estimate: 1 points

## Overall completion contribution

Contribution: 1/9 points toward parent #13761.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13807
