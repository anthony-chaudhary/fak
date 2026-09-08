# fix(compute): bind Vulkan GDN recurrence to production CPU semantics

<!-- fak-compute-key: vulkan-qwen35-gdn-production-parity -->

```routing
lane: internal/compute
paths: ["internal/model/vulkan_qwen35_gdn_production_parity_test.go"]
expected_steps: 3
```

## Current state

Commit `fb3ba5f1d` corrected the Vulkan Gated-DeltaNet recurrence shader for
#12026 and added compute-package oracle coverage. The public issue remains open
because its acceptance also requires a physical comparison against the actual
production CPU model path with carried state, multiple tokens, and logits.

## Core through-line

Run identical synthetic weights, nonzero convolution/recurrent state, and three
sequential inputs through `Session.linearAttnStep` and the real Vulkan
`Qwen35GDNDecode` path. Compare hidden output, both persistent states, vocabulary
logits, and greedy token choice.

## Gold-plating boundary

Do not change the recurrence, relax the established parity floor, add a fallback,
or claim a performance improvement. This leaf adds the missing independent
production-path witness only.

## Done condition / witness

- [ ] A Vulkan-tagged model test compiles without an overlay on the sanctioned host.
- [ ] The physical Radeon backend executes three sequential steps from nonzero state.
- [ ] Hidden output, convolution state, recurrent state, and logits satisfy the declared parity bounds.
- [ ] Greedy token choice matches the production CPU path at every step.
- [ ] The pre-fix shader fails the identical binary and input panel.

## Verifiable Witness

```bash
FAK_VULKAN_GDN_REQUIRED=1 FAK_VULKAN_SPIRV=<spirv-dir> \
  ./model.test -test.run '^TestVulkanQwen35GDNMatchesProductionCPUMultiStep$' -test.v
```

## File:Line seams

- `internal/model/qwen35.go` (`Session.linearAttnStep`)
- `internal/compute/vulkan_qwen35_gdn.go` (`Qwen35GDNDecode`)
- `internal/compute/shaders/qwen35_gdn_recurrent.comp` (`main`)

## Blast radius and affected lanes

- Affected: Vulkan-tagged Qwen3.5/3.8 GDN correctness tests.
- Unaffected: production runtime code and non-Vulkan builds.
- Fallback: none; required physical runs fail when the Vulkan backend is absent.

## Likely files

- `internal/model/vulkan_qwen35_gdn_production_parity_test.go`

## Expected steps

3

## Lane

internal/compute

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points.

## Completion standard

production

## Target operating envelope

acceptance pass rate: = 100 percent

## Witnessed operating envelope

acceptance pass rate: = 0 percent pending physical witness
