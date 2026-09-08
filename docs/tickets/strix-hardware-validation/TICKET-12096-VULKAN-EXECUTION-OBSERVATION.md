<!-- fak-qwen38-key: vulkan-executed-backend-observation -->
# feat(compute): expose executed Vulkan identity and counter deltas

GitHub: #12213. Parent issues: #12096 and #12097. Follows #12211.

```routing
lane: qwen38-vulkan-execution-observation
paths: ["internal/compute/physical_observation.go", "internal/compute/physical_observation_test.go", "internal/compute/vulkan_device_debug.go", "internal/compute/vulkan_backend.c", "internal/compute/vulkan_backend.h", "internal/rawdecode/executor.go", "internal/rawdecode/executor_test.go"]
expected_steps: 8
```

## Parent context

Child prerequisite of #12096 and #12097. It follows the running source/binary
observer in #12211 and does not close either parent.

## Current state

The real raw-decode executor reports selected backend name/class/capabilities,
but it has no importable contract for identity or execution-scoped counters.
The Vulkan backend already owns cumulative transfer, dispatch, Q4_K stage, and
tensor-home snapshots, yet raw decode cannot distinguish an unsupported
observer from an observed zero and cannot bind device/runtime identity.

## Working spine

The compute backend instance used by raw decode is the authoritative execution
seam. Its existing cumulative snapshots can be taken immediately before and
after one serialized decode, while the Vulkan shim already owns the selected
physical-device properties. A small optional compute interface can expose a
typed identity and snapshot without accepting caller labels.

## Why this is next

After #12211, executed device/runtime/fallback identity and dispatch/transfer/
tensor-home evidence are the largest remaining runner-owned gap. Capturing them
at the actual backend prevents request flags or environment variables from
masquerading as execution evidence.

## Core through-line

Add an optional compute-backend observation interface whose production Vulkan
implementation derives device/API/driver/runtime identity from the selected
backend and returns cumulative dispatch, transfer, Q4_K stage/fallback, tensor
home, and device-memory snapshots. Raw decode captures before/after snapshots
around the real execution and emits a validated delta only when both snapshots
are supported, identity-stable, and monotonic.

Unsupported fields remain unavailable through explicit presence state; a zero
value is never treated as an observation. The contract accepts no identity or
counter values from the raw-decode request.

## Gold-plating boundary

- No appliance, SSH, service, or physical benchmark.
- No synthetic counters, process-global reset during shared execution, or
  environment-variable identity.
- No source archive, tokenizer/template/tensor-inventory, host firmware,
  clocks, power, or thermal collection.
- No fanout receipt promotion until the complete #12096 identity envelope is
  present.

## Done condition

- [ ] Compute exposes an optional fail-closed executed-backend identity and
  cumulative snapshot contract.
- [ ] Vulkan derives identity from the selected physical backend and reuses its
  existing runtime counters.
- [ ] Raw decode reports only monotonic execution deltas captured around the
  backend it actually invoked.
- [ ] Missing support, identity drift, reset/wrap, or partial snapshots remain
  unavailable and cannot be zero-filled.
- [ ] Device-free fakes prove caller request labels cannot forge identity or
  counters and unsupported backends remain unavailable.
- [ ] Focused tests, vet, exact validation, and leak audit pass.

## Done condition / witness

Done condition: the real executor can carry a sealed backend-owned observation
without manufacturing absent values.

Witness: focused device-free tests prove monotonic deltas and fail-closed
unsupported/mismatch behavior.

## Definition of done

All software criteria pass; physical fanout remains unavailable until #12096's
remaining source-archive/model/host identity contract and trusted-host smoke.

## Witness

```text
go test ./internal/compute ./internal/rawdecode -run 'Test.*(PhysicalObservation|RawDecode.*Observation)' -count=1
go vet ./internal/compute ./internal/rawdecode
fak validate --mine internal/compute/physical_observation.go --mine internal/compute/physical_observation_test.go --mine internal/compute/vulkan_device_debug.go --mine internal/compute/vulkan_backend.c --mine internal/compute/vulkan_backend.h --mine internal/rawdecode/executor.go --mine internal/rawdecode/executor_test.go
```

The witness is device-free and earns no `[HW-WITNESSED]` status.

## Acceptance gate

All focused tests/vet and exact-path validation pass, and review finds no
request/environment fields copied into the observation.

## Closure binding

The resolving commit cites this child issue and carries `(fak compute)` or
`(fak rawdecode)`. A documentation-only ticket commit does not close it.

## Likely files

- `internal/compute/physical_observation.go` and focused test
- `internal/compute/vulkan_device_debug.go`
- `internal/compute/vulkan_backend.c` and `.h`
- `internal/rawdecode/executor.go` and focused test

## Lane

`qwen38-vulkan-execution-observation`; two packages, expected steps: 8.

## Verifiable witness details

- Repro: the current raw-decode `Execution` type explicitly has no device,
  memory, or counter fields despite the backend owning debug snapshots.
- Exact seams: `internal/rawdecode/executor.go:101-130` and
  `internal/compute/vulkan_device_debug.go:45-125`.
- Blast radius: compute observation and raw-decode orchestration only.
- Fallback: omit the optional observation; #12096/#12097 continue to refuse
  physical receipt bytes.

## Classification

- Portfolio tier: 2 (serving); prerequisite for tier 1 fanout evidence.
- Centrality: Core
- P1 Context: advanced - binds the backend that performed real work.
- P2 Net value: preserved - unavailable telemetry stays unavailable.
- P3 Adaptation: preserved - reuses current backend counters.
- P4 Operations: advanced - mismatch and counter reset fail closed.

## Work unit

leaf

## Expected steps

8

## Work estimate

Estimate: 3 points.

## Overall completion contribution

Contribution: 3/8 points toward the complete physical observation envelope;
zero points toward the trusted-host witness.

## Completion standard

production

## Target operating envelope

- unsupported or mismatched observations admitted: = 0 percent

## Witnessed operating envelope

- unsupported or mismatched observations admitted: = 0 percent
