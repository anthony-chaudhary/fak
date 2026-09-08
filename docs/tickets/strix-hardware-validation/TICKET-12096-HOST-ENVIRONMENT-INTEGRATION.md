<!-- fak-qwen38-key: rawdecode-observed-host-environment-integration -->
# feat(rawdecode): bind observed RADV host identity into canonical receipts

GitHub: #12309. Parent: #12096. Depends on #12281 and follows #12232.

```routing
lane: rawdecode-vulkan-host-receipt
paths: ["internal/rawdecode/executor.go", "internal/rawdecode/executor_test.go", "cmd/modelbench/raw_decode.go", "cmd/modelbench/raw_decode_test.go", "docs/tickets/strix-hardware-validation/TICKET-12096-HOST-ENVIRONMENT-INTEGRATION.md"]
expected_steps: 7
```

## Current state

#12281 provides a strict observed Linux/RADV host tuple, and #12232 carries
backend-owned Vulkan execution identity per raw-decode repetition. The real
rawdecode path does not retain the host tuple, so modelbench cannot populate
the canonical host/device fields or prove they describe the device that ran.

## Why this is next

#12096 cannot promote an exact physical receipt until observed host and device
identity survives the real execution boundary and binds to the device that ran.

## Parent context

#12096; depends on the #12281 producer and follows #12232 integration.

## Working spine

Observe one host tuple beside each real repetition, retain it with that run,
cross-bind its device name and PCI vendor/device IDs to the backend-owned
Vulkan identity, then map only a complete, stable tuple into the canonical
receipt attempt.

## Core through-line

`#12281 observer -> rawdecode.Run -> exact per-run backend/PCI binding ->
modelbench canonical Host/Device`. Missing, ambiguous, mismatched, or drifting
evidence leaves the receipt `UNAVAILABLE` and emits no canonical receipt bytes.

## Gold-plating boundary

No hardware access, benchmark, service/SSH action, static hardware defaults,
environment or argv provenance, opaque-driver-string inference, Windows host
observer, Mesa inference, kernel tuning, or performance claim. Source/archive,
memory, counters, and other incomplete #12096 fields remain unchanged.

## Concrete repro witness

`go test ./internal/rawdecode ./cmd/modelbench -run 'Test.*RawDecode.*HostEnvironment' -count=1`
currently has no per-run host observation or canonical mapping to exercise.

## Exact seams and blast radius

- `internal/rawdecode/executor.go` (`dependencies`, `Run`, `executeLoaded`) owns
  injected per-repetition observation and exact binding to `BackendExecution`.
- `cmd/modelbench/raw_decode.go` (`rawDecodePhysicalReceipt`) owns stable-run
  validation and canonical Host/Device mapping.
- Only `internal/rawdecode` and `cmd/modelbench` change. Other compute and
  serving lanes remain unaffected.

## Quarantined fallback

The raw decode result remains useful as a non-canonical report, but absent or
invalid host evidence is never filled from defaults. Canonical construction
continues to fail closed with `UNAVAILABLE`, `credit_eligible=false`, and a nil
receipt.

## Definition of done

- [x] Production rawdecode invokes only the #12281 observer for each Vulkan run.
- [x] Device-free tests inject the observer and perform no live sysfs or
  `vulkaninfo` reads.
- [x] Host tuple device name and exact sealed PCI IDs bind to each run's
  backend-owned Vulkan identity.
- [x] Missing, ambiguous, mismatched, or per-run drifting tuples cannot populate
  canonical Host/Device fields or emit receipt bytes.
- [x] A complete stable tuple maps OS, arch, kernel, device, Mesa, Vulkan, and
  firmware exactly while the still-incomplete parent receipt remains unavailable.
- [x] Focused/full affected tests, vet, exact-path and leak checks pass.

## Done condition / witness

The four-file software leaf is complete when injected fixtures prove exact
mapping and every fail-closed class without hardware access. This grants zero
physical-performance credit and does not close #12096.

## Acceptance gate

Focused and full affected tests plus vet are green, and independent review
confirms every mapped host/device value is observed and per-run cross-bound.

## Closure binding

The signed resolving commit cites this child with `Ref #<child>` and carries
`(fak rawdecode)`; #12096 remains open for the complete physical envelope.

## Verifiable Witness

```text
go test ./internal/rawdecode ./cmd/modelbench -run 'Test.*RawDecode.*HostEnvironment' -count=1
go test ./internal/rawdecode ./cmd/modelbench -count=1
go vet ./internal/rawdecode ./cmd/modelbench
```

## Likely files

- `internal/rawdecode/executor.go`
- `internal/rawdecode/executor_test.go`
- `cmd/modelbench/raw_decode.go`
- `cmd/modelbench/raw_decode_test.go`
- `docs/tickets/strix-hardware-validation/TICKET-12096-HOST-ENVIRONMENT-INTEGRATION.md`

## Lane

`rawdecode-vulkan-host-receipt`; two Go packages, five paths, seven steps.

## Classification

- Portfolio tier: 2 (serving); enables tier 1 measurement.
- Centrality: Core receipt prerequisite.
- P1: advanced - creates an exact physical host/device identity chain.
- P2: preserved - software-only or incomplete evidence receives zero credit.
- P3: preserved - reuses #12281 and #12232 rather than adding observers.
- P4: advanced - rejects missing, mismatched, ambiguous, and drifting identity.

## For / Problem / Today / Better because / Witness

- **For:** maintainers evaluating the canonical Strix Halo B=1 comparison.
- **Problem:** the receipt attempt drops strict host evidence before promotion.
- **Today:** backend execution identity and host identity are separate facts.
- **Better because:** every mapped field is observed and cross-bound per run.
- **Witness:** deterministic injected fixtures reject all invalid bindings and
  preserve an unavailable, zero-byte receipt.

## Work unit

leaf

## Expected steps

7

## Work estimate

Estimate: 1 point.

## Overall completion contribution

Contribution: 1/8 points toward #12096; zero physical credit.

## Completion standard

development

## Target operating envelope

- incomplete host observations promoted: = 0 percent

## Witnessed operating envelope

- incomplete host observations promoted: = 0 percent
