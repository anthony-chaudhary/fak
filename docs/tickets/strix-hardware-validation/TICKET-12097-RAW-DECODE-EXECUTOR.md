<!-- fak-qwen38-key: rawdecode-importable-executor -->
# refactor(rawdecode): expose the real modelbench executor to physical fanout

GitHub: #12193. Parent: #12097. Prerequisite: #12096.

```routing
lane: qwen38-rawdecode-executor
paths: ["internal/rawdecode/**", "cmd/modelbench/main.go", "cmd/modelbench/raw_decode.go", "cmd/modelbench/raw_decode_test.go"]
expected_steps: 8
```

## Parent context

Child prerequisite of #12097. It reuses the fail-closed physical receipt
boundary from #12096 and does not close either parent.

## Why this is next

The #12097 CLI boundary is now safe, but no importable implementation can reach
the real raw model path. This extraction is the smallest dependency that lets a
later runner call existing engine code without duplicating it or inventing
identity.

## Current state

`cmd/fak bench subagent --simulated=false` correctly fails closed after
`b0459b090`; it has no production `PhysicalRunner`. The old
`ProductPhysicalRunner` at `internal/qwen38campaign/subagent_fanout.go:629-888`
manufactures tokens, logits, timings, and identity and therefore cannot be
reattached.

The real greedy fak-native decode is implemented in the non-importable
`package main` at `cmd/modelbench/raw_decode.go:306-690`. Its load and backend
selection are also command-owned at `cmd/modelbench/main.go:51-91` and
`cmd/modelbench/main.go:200-230`. The #12096 promotion boundary is importable at
`internal/compute/qwen38_vulkan_decode_receipt.go:172-287`, but moving partial
values through that builder does not execute a model and correctly remains
`UNAVAILABLE`.

Today #12097 cannot reach the real engine without either copying the raw-decode
loop into `internal/qwen38campaign` or trusting caller-supplied identity. Both
would violate the single-engine and fail-closed evidence contracts.

## Working spine

The shipped spine already has three sound pieces: `ggufload` opens the selected
artifact, `model.Session` plus the registered `compute.Backend` performs real
prefill/decode, and `BuildQwen38VulkanDecodeReceipt` rejects incomplete physical
evidence. This leaf only moves the command-local orchestration behind an
importable boundary while retaining those implementations and tests.

## Classification

- Portfolio tier: 2 (serving); directly enables tier 1 physical fanout evidence.
- Centrality: Core
- P1 Context: advanced - makes the already-shipped raw fak-native decode reusable by the
  product fanout runner.
- P2 Net value: preserved - observed execution stays separate from modeled fanout.
- P3 Adaptation: preserved - one executor remains authoritative; no second benchmark
  framework or external inference fallback is introduced.
- P4 Operations: advanced - artifact/backend/token inputs are explicit and mismatches fail
  before execution.

## Core through-line

Extract the existing path without changing its math:

`explicit artifact + token request -> ggufload/model/compute engine -> observed tokens and timings -> Qwen38VulkanRawDecodeResult`.

Create an importable `internal/rawdecode` executor whose configuration names the
artifact path, expected artifact SHA-256, model name, backend, prompt token IDs,
context limit, generated-token limit, repetitions, EOS policy, and CPU
verification policy. The executor must hash the opened artifact itself and
reject an expected-digest mismatch before model execution. It must use the
registered fak-native backend and the existing raw-decode loop, return only
values observed by that loop, and leave source/device/counter fields absent
until their runner-owned collectors exist.

Keep `cmd/modelbench` as a thin CLI adapter to this executor. Delete or delegate
the command-local loop so there is exactly one implementation. A later bounded
#12097 leaf may import this package, attach runner-owned provenance collectors,
and map a fully promoted #12096 receipt into `PhysicalTrialResult`.

## Gold-plating boundary

- No appliance access, service restart, tuning, or physical throughput claim.
- No qwen38campaign runner attachment in this leaf.
- No new cache, scheduler, multi-node, or comparison framework.
- No external `llama.cpp` subprocess or fallback.
- No environment-variable or caller-provided source, binary, device, timing, or
  counter identity.
- No promotion of an incomplete #12096 raw result to an AVAILABLE receipt.

## Done condition

- [ ] `internal/rawdecode` owns the one real raw-decode execution loop and model
  load/backend selection needed by both command and product consumers.
- [ ] The executor accepts an explicit artifact selector, hashes the opened file,
  and rejects missing or mismatched identity before execution.
- [ ] Generated tokens, finite-logit state, CPU parity, and per-run timings come
  only from actual session calls; unavailable identity and counters stay absent.
- [ ] `cmd/modelbench` delegates to the extracted executor with no duplicate
  token-generation loop.
- [ ] A device-free regression proves the production adapter calls the executor
  seam exactly once and that an incomplete observation cannot become a physical
  receipt.
- [ ] Focused tests and vet pass for both affected packages.

## Done condition / witness

The command-local implementation has been replaced by one importable executor,
the artifact selector is verified by the executor from opened bytes, every
reported token/timing/parity field originates in a real session call, incomplete
#12096 identity stays unpromoted, and the focused witness commands in the next
section pass.

## Definition of done

- [ ] One importable executor owns the real raw-decode loop.
- [ ] Modelbench delegates to it without a duplicate implementation.
- [ ] Artifact mismatch and incomplete physical evidence fail closed.
- [ ] Focused tests, vet, and prospective exact-path validation pass.

## Witness

```text
go test ./internal/rawdecode ./cmd/modelbench -run 'Test.*(RawDecodeExecutor|RawDecode).*' -count=1
go vet ./internal/rawdecode ./cmd/modelbench
fak validate --mine internal/rawdecode/executor.go --mine internal/rawdecode/executor_test.go --mine cmd/modelbench/main.go --mine cmd/modelbench/raw_decode.go --mine cmd/modelbench/raw_decode_test.go
```

The device-free witness proves routing, fail-closed validation, and preservation
of observed fields only. It earns no `[HW-WITNESSED]` status.

## Acceptance gate

All focused tests, both package vets, and the exact-path prospective validator
pass. Review confirms `cmd/modelbench` and the later consumer share one executor,
and no incomplete observation promotes through #12096.

## Closure binding

The resolving commit cites this child issue and carries `(fak rawdecode)`. Its
issue update records the exact paths and green device-free witnesses and states
that no hardware execution occurred.

## Work unit

leaf

## Expected steps

8

## Work estimate

Estimate: 3 points.

## Overall completion contribution

Contribution: 3/8 points toward the importable-runner prerequisite; zero points
toward the hardware witness required by #12097.

## Completion standard

production

## Target operating envelope

incomplete physical receipt promotion rate: = 0 percent

## Witnessed operating envelope

incomplete physical receipt promotion rate: = 0 percent

## Likely files

- `internal/rawdecode/executor.go` and `internal/rawdecode/executor_test.go`
- `cmd/modelbench/main.go`
- `cmd/modelbench/raw_decode.go` and `cmd/modelbench/raw_decode_test.go`

## Lane

`qwen38-rawdecode-executor`; two Go packages, expected steps: 8.

## Verifiable witness details

- Repro: `go test ./cmd/fak -run '^TestRunBenchSubagentPhysicalModeRequiresAttachedRunner$' -count=1`
  is green because no real runner can currently be constructed.
- Exact blocked seam: `cmd/fak/bench_subagent.go:11-19` accepts an explicit
  `PhysicalRunner`, while the real implementation is trapped behind
  `cmd/modelbench/raw_decode.go:306-690`.
- Blast radius: `cmd/modelbench` and new `internal/rawdecode` only; serving,
  gateway, and other accelerator lanes remain unaffected.
- Fallback: retain the landed #12097 nil-runner guard. Physical mode continues
  to fail with zero receipt bytes until this child and the remaining #12096
  provenance collectors are complete.
