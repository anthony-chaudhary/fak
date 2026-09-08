# fix(compute): disambiguate duplicate Vulkan Qwen35 recovery tests

<!-- fak-compute-key: vulkan-qwen35-duplicate-recovery-test -->

```routing
lane: internal/compute
paths: ["internal/compute/vulkan_test.go"]
expected_steps: 2
```

## Current state

`internal/compute/vulkan_test.go` declares `TestVulkanQwen35_ResidencyAndErrorRecovery`
twice, at lines 1619 and 1938. Go rejects the package before any `vulkan`-tagged
test can execute. The first function exercises host-only residency validation; the
second requires a physical Vulkan device and checks transient-buffer recovery.

## Why this is next

This duplicate declaration blocked the physical #11803 shader witness before the
candidate test could execute. Restoring tagged compilation makes every Vulkan kernel
regression discoverable again.

## Parent context

Follow-up discovered while verifying #11803.

## Problem Frame

- Centrality: Core - correctness gate for the Vulkan compute backend.
- P1 Context: advanced - native shader regressions cannot be witnessed while the test binary fails to compile.
- P2 Net value: preserved - retain both distinct test bodies and restore tagged package compilation.
- P3 Adaptation: preserved - rename only the host-only validation test; neither test body changes.
- P4 Operations: preserved - no production code or device behavior changes.

## Core through-line

Give the host-only residency validation test a unique descriptive name while retaining
the physical-device recovery test under its existing name. Prove that the tagged test
binary compiles and lists both contracts.

## Working spine

`internal/compute/vulkan_test.go` unique test identifiers -> tagged test compilation ->
test discovery -> host-only residency validation execution.

## Gold-plating boundary

Do not merge, rewrite, weaken, or delete either test body. Do not change Vulkan runtime
code, shader code, device selection, or error semantics.

## Done condition / witness

- [x] `go test -c -tags vulkan ./internal/compute` compiles on the sanctioned Vulkan build host.
- [x] The compiled binary lists both `TestVulkanQwen35ResidencyValidationWithoutDevice` and `TestVulkanQwen35_ResidencyAndErrorRecovery`.
- [x] The host-only validation test passes.

## Definition of Done

- [x] Both distinct recovery contracts have unique Go test identifiers.
- [x] The `vulkan`-tagged compute test binary compiles without an overlay.
- [x] Test discovery lists both recovery contracts and the host-only contract passes.

## Acceptance gate

The tagged compute test binary compiles without an overlay, lists both unique test
identifiers, and the host-only validation test exits zero.

## Closure binding

The resolving commit cites the created public issue and carries `(fak compute)`.

## Verifiable Witness

```bash
go test -c -tags vulkan ./internal/compute
./compute.test -test.list 'TestVulkanQwen35.*Residency'
./compute.test -test.run '^TestVulkanQwen35ResidencyValidationWithoutDevice$' -test.v
```

## Witness

Witnessed on the physical Strix Vulkan build host with `vulkan_test.go` source blob
`95bc8b8186c180f59a5d5836a9ae541c2994358f`. The tagged test binary compiled without a Go overlay (SHA-256
`ae515dab7ae0a10538835e4e689b38a528f8307cd58e7fd8a7d5f2c080a3e21b`), test
discovery printed both identifiers, and
`TestVulkanQwen35ResidencyValidationWithoutDevice` passed.

## File:Line seams

- `internal/compute/vulkan_test.go:1619` (`TestVulkanQwen35_ResidencyAndErrorRecovery`, host-only validation)
- `internal/compute/vulkan_test.go:1938` (`TestVulkanQwen35_ResidencyAndErrorRecovery`, physical-device recovery)

## Blast radius and affected lanes

- Affected: `internal/compute` tests built with `vulkan`.
- Unaffected: production compute code and all non-Vulkan builds.
- Fallback: the prior #11803 hardware witness used a build-only overlay that renamed the second duplicate; no production source was altered by that overlay.

## Likely files

- `internal/compute/vulkan_test.go`

## Expected steps

2

## Lane

internal/compute

## Work estimate

Estimate: 1 point.

## Overall completion contribution

Contribution: 1/1 point.

## Completion standard

production

## Target operating envelope

acceptance pass rate: = 100 percent

## Witnessed operating envelope

acceptance pass rate: = 100 percent
