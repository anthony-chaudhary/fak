<!-- qwen-composability-key: device-memory-profile-v1 -->
# Device-specific scratch planning

Issue: https://github.com/anthony-chaudhary/fak/issues/12440

```routing
lane: internal/compute
paths: internal/compute/device_memory_profile.go, internal/compute/device_memory_profile_test.go, internal/compute/rocm_arch.go, tools/concept_disambiguation_scorecard.data/rows-device-memory-profile.json, docs/tickets/halo-memory-profile/TICKET-12440.md
expected_steps: 5
```

## Current state

The explicit device-memory profile and profile-taking planning API are introduced.
Existing ROCm and Vulkan planner entry points have not yet migrated to that API,
so the behavioral fix and issue closure remain pending a consumer commit.

Priority: P1. Centrality: Core. This is the bounded prerequisite for device-aware
sparse attention policy under #12185, within #10193's native Qwen path.

## Core through-line

Detected backend/device facts -> explicit memory profile -> existing ROCm QSA
and Vulkan scratch planners -> distinct known-device and unknown-device decisions.
Keep explicit supported Halo behavior; unknown capability must not become a
Halo default. An optional backend profile takes precedence over an architecture
hint. Allocation accounting and cache eligibility are software policy decisions.

## Gold-plating boundary

No QSA kernels, MTP implementation, new hardware tuning, performance targets,
or private fleet policy. Go slice allocation and modeled cache fit do not prove
physical address alignment, GPU residency, or measured performance.

## Done condition

- [x] The explicit profile and profile-taking planning API are available.
- [ ] Existing planners consume explicit cache, alignment, and subgroup facts.
- [ ] gfx1151 and another known device produce profile-specific planner decisions.
- [ ] Unknown Vulkan devices do not inherit gfx1151 or its cache capacity.

Closure requires a resolving commit landed with boundary and public leak audits;
the issue closure and landing receipt bind the final commit identity.

## Witness

API-stage witness: `go test -short ./internal/compute -run TestDeviceMemoryProfileDrivesScratchPolicy -count=1 -v`.
This establishes only the software profile contract. Consumer behavior and
[HW-WITNESSED] evidence are not claimed by this stage.

## Likely files

- `internal/compute/device_memory_profile.go`: explicit device facts and policy helpers.
- `internal/compute/rocm_arch.go`: `TuneQSASparseGather`.
- `internal/compute/vulkan_kv.go`: `NewVulkanKVScratchpad` and cache eligibility.
- `internal/compute/device_memory_profile_test.go`: independent acceptance witness.
- `internal/compute/device_memory_profile_vulkan_test.go`: tagged consumer integration witness.

## Lane

Public `internal/compute`; no private imports. Other live kernel work remains
independent. The unknown-device path must refuse unsupported cache planning
or explicitly report that capability as unavailable.
