<!-- fak-qwen38-key: backend-observation-receipt-integration -->
# feat(qwen38campaign): carry observed backend evidence into physical receipt attempts

GitHub: #12232. Parent issues: #12096 and #12097. Follows #12213 and #12211.

```routing
lane: qwen38-backend-observation-integration
paths: ["internal/qwen38campaign/fanout_contract.go", "internal/qwen38campaign/subagent_fanout.go", "internal/qwen38campaign/subagent_fanout_test.go", "cmd/modelbench/raw_decode.go", "cmd/modelbench/raw_decode_test.go"]
expected_steps: 7
```

## Current state

#12213 makes `internal/rawdecode` return a backend-owned, per-execution Vulkan
identity and monotonic counter delta. The B=1 fanout adapter and modelbench
canonical receipt attempt do not consume that observation, so real execution
evidence is dropped before either scorecard boundary.

Both boundaries correctly remain unavailable when source archive, full model
identity, host identity, peak memory, or complete canonical transfer counters
are missing. This leaf must preserve that fail-closed state.

## Working spine

Map the already-sealed `rawdecode.Run.BackendExecution` observation into the
fanout run receipt and the canonical raw-decode attempt. The fanout schema
records the complete observed identity, dispatch/transfer/tensor-home deltas,
fallback delta, closing tensor-home gauges, and the optional closing device
memory gauge. Modelbench maps only fields with exact canonical meaning:
executed device/runtime/fallback and complete dispatch/tensor-home values.

## Core through-line

`rawdecode.Execution -> backend-owned observation -> non-creditable physical
receipt attempt`. Missing, mismatched, incomplete, reset, or underflowed
observations fail before fanout promotion. Canonical promotion stays
`UNAVAILABLE` until the remaining independently observed identity and memory
envelope exists.

## Gold-plating boundary

- No appliance, SSH, service mutation, or physical benchmark.
- No caller or environment labels as provenance.
- No conversion of H2D/D2H byte deltas into DRAM traffic or peak memory.
- No zero-filled transfer counts, D2D byte totals, process memory, device peak,
  source archive, tokenizer, template, or tensor-inventory identity.
- No performance or hardware claim and no parent closure.

## Done condition

- [ ] The B=1 rawdecode physical adapter requires a sealed backend observation
  and carries it unchanged into each physical run metric.
- [ ] Backend/device/driver/runtime are non-empty, selected backend matches, and
  observed fallback is zero before fanout promotion.
- [ ] Missing, unsupported, reset/underflowed, mismatched, or incomplete
  observations cannot emit a fanout receipt.
- [ ] Modelbench maps only semantically exact observed identity/counter fields
  and keeps the canonical receipt attempt `UNAVAILABLE` while other required
  observations are absent.
- [ ] Device-free tests prove no DRAM, MALL, peak-memory, or unavailable transfer
  fields are synthesized.
- [ ] Focused tests, vet, exact-path validation, and public leak audit pass.

## Done condition / witness

Done condition: sealed backend execution evidence reaches both physical receipt
attempts without making either incomplete envelope creditable.

Witness: device-free fakes exercise the real adapter mapping and fail-closed
validation; no hardware is accessed.

## Witness

```text
go test ./internal/qwen38campaign ./cmd/modelbench -run 'Test.*(RawDecode.*Backend|SubagentFanout.*Backend)' -count=1
go vet ./internal/qwen38campaign ./cmd/modelbench
fak validate --mine internal/qwen38campaign/fanout_contract.go --mine internal/qwen38campaign/subagent_fanout.go --mine internal/qwen38campaign/subagent_fanout_test.go --mine cmd/modelbench/raw_decode.go --mine cmd/modelbench/raw_decode_test.go
```

The witness is device-free and earns no `[HW-WITNESSED]` status.

## Acceptance gate

Focused tests and vet are green; exact review shows every promoted value derives
from `Run.BackendExecution`, and incomplete attempts emit no canonical receipt.

## Closure binding

The resolving implementation commit cites this child issue and carries the
`(fak qwen38campaign)` leaf trailer. The physical parent issues remain open.

## Likely files

- `internal/qwen38campaign/fanout_contract.go` physical observation contract
- `internal/qwen38campaign/subagent_fanout.go` rawdecode mapping
- `internal/qwen38campaign/subagent_fanout_test.go` device-free regression
- `cmd/modelbench/raw_decode.go` canonical partial mapping
- `cmd/modelbench/raw_decode_test.go` unavailable/absence regression

## Lane

`qwen38-backend-observation-integration`; two packages, expected steps: 7.

## Verifiable witness details

- Repro: a rawdecode run with a non-nil `BackendExecution` currently produces a
  fanout result whose counter source remains `UNAVAILABLE`, and modelbench drops
  the observation entirely.
- Exact seams: `internal/qwen38campaign/subagent_fanout.go:675-735` and
  `cmd/modelbench/raw_decode.go:231-306`.
- Blast radius: qwen38 physical fanout and modelbench physical receipt attempts;
  simulation and compute collection are unchanged.
- Fallback: missing observation returns no physical fanout receipt; modelbench
  continues returning a scrubbed `UNAVAILABLE` attempt.

## Classification

- Portfolio tier: 1 (all-in-one fanout evidence) supported by tier 2 serving.
- Centrality: Core.
- P1 Context: advanced - preserves actual executed-backend evidence.
- P2 Net value: preserved - incomplete data never earns comparison credit.
- P3 Adaptation: preserved - reuses #12213 rather than adding instrumentation.
- P4 Operations: advanced - missing and mismatched observations fail closed.

## Work unit

leaf

## Expected steps

7

## Work estimate

Estimate: 3 points.

## Overall completion contribution

Contribution: one integration step toward #12096/#12097; zero points toward the
trusted-host physical witness.

## Completion standard

production

## Target operating envelope

- incomplete backend observations promoted: = 0 percent

## Witnessed operating envelope

- incomplete backend observations promoted: = 0 percent
