# feat(ggufload): read and validate .rco-allocation.txt sidecars against the parsed tensor inventory (fak ggufload)

GitHub issue: [#14093](https://github.com/anthony-chaudhary/fak/issues/14093).

Process cause: none

```routing
lane: gsq-rco-followons
paths: ["internal/ggufload/companion.go", "internal/ggufload/allocation_sidecar.go", "internal/ggufload/allocation_sidecar_test.go", "cmd/fak-dev/"]
expected_steps: 4
```

## Current state

ISTA-DASLab ships a per-tensor allocation sidecar (`.rco-allocation.txt`) alongside every mixed-precision GGUF build of Qwen3.8-Flash-Next: quant type per tensor + quant-type histogram + target whole-file bpw + fixed/BF16-held tensors. Four builds, 1,223 tensors each, pinned HF revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`; manifests are public artifacts of that HF repo (four files ~39 KB each, fetchable at the pin). Format: a `# RCO per-tensor GGUF allocation` header block (repository, GGUF name, target whole-file bpw, GGUF version, metadata-entry count, tensor count, quant-type counts) followed by `<GGUF tensor name>: <GGML quantization/storage type>` rows that include the fixed non-RCO tensors.

Public `fak` has no reader for it. Real seams:

- `internal/ggufload/companion.go` owns companion-file handling today: `DetectDraftKind`/`ExtractQuantTag`/`RankCompanionDrafts`/`DiscoverCompanion`/`DiscoverDirectoryCompanions` pair `.gguf` draft sidecars next to a target model, with sentinel errors `ErrNoCompanionFound`/`ErrAmbiguousDraftMechanism` and a `knownQuantTags` tag-to-bits table. Text sidecars are out of its scope.
- `internal/ggufload/artifact_quant.go` owns recipe classification: `ClassifyArtifactQuant`/`ClassifyTensorQuant` over `[]TensorInfo`, `ArtifactQuant{Name, Inventory, Q4KResident, Recipe}` with the UD-* recipe families.
- `internal/ggufload/gguf.go:109-132`: `TensorType` constants recognize every encoding used by any of the four builds (F32/F16/BF16, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ1_M, IQ2_S/XXS/XS, IQ3_S/XXS, IQ4_NL/XS).

Baseline: ABSENT — `rg -n -i 'rco-allocation|allocation_sidecar' internal/ggufload/` returns no match (exit 1, verified at study time).

## Problem frame

- Centrality: Enabling — mixed-precision artifacts become auditable without parsing the GGUF; kernel-coverage and quant-floor checks consume plain text instead of a 66–84 GB shard.
- P1 Context: advanced — extends the shipped companion-discovery and recipe-classification seams; no loader-behavior change.
- P2 Net value: advanced — fail-closed probe for qwen38-flash-next bringup; a mislabeled or drifted mixed-precision artifact is caught by reading a ~39 KB text file, not by opening the shard.
- P3 Adaptation: preserved — adopts an Apache-2.0-provenance upstream convention; reuses the companion.go discovery pattern and the artifact_quant.go tensor-inventory input.
- P4 Operations: preserved — read-only probe verb; fail-closed (non-zero exit on any mismatch); no mutation of artifact or loader.

## Core through-line

Step 1: parser — new `internal/ggufload/allocation_sidecar.go` decodes the header block and the `<tensor>: <TYPE>` rows into a typed `AllocationSidecar`. Step 2: validator — `ValidateAllocation(sidecar, []TensorInfo)` checks every tensor has exactly one declared type, declared types are loader-admissible `TensorType`s, fixed tensors match the sensitivity-floor set, target bpw vs actual bytes within tolerance, and the declared histogram matches observed counts; each mismatch class gets its own sentinel error. Step 3: hermetic fixture tests generated from the four cached manifest texts. Step 4: fail-closed probe verb in `cmd/fak-dev/` (name TENTATIVE: `fak-dev artifact allocation-check`).

## Scope

- In:
  - Parser: header fields (repository, GGUF name, target whole-file bpw, GGUF version, metadata-entry count, tensor count, quant-type counts) plus one row per tensor; duplicate and malformed rows fail closed.
  - Validator mismatch classes (one sentinel error each, matched via `errors.Is`): missing sidecar, unknown tensor (declared but not in the inventory), undeclared tensor (in the inventory but not declared), type mismatch (declared differs from the actual `TensorType`, including non-loader-admissible declared types), histogram drift (declared quant-type counts differ from observed), bpw overrun (actual bytes vs target whole-file bpw outside tolerance).
  - Fixed-tensor floor: fixed/BF16-held tensors must match the sensitivity-floor set (hc_* highway path, ffn_gate_inp*, PLE top-level, SSM parameters, attention norms) per the manifest's fixed-tensor rows.
  - Verb: `fak-dev artifact allocation-check` (TENTATIVE) — takes a pinned GGUF path plus a sidecar path (or directory), parses, validates, prints a machine-readable report; non-zero exit on any mismatch.
  - Tests: hermetic fixtures generated from the four cached manifest texts; one test per mismatch class asserting the sentinel via `errors.Is`.
- Out:
  - No shard download, no GGUF byte-range reads beyond the existing `OpenWeights` path.
  - No recipe-classification semantics change in `artifact_quant.go`; no per-backend kernel-coverage implementation (that gap is a separate lane).
  - No RCO allocator port (study row R4); no weight-tiering/residency policy (row R2).

## Gold-plating boundary

One parser, one validator, one verb. No sidecar authoring tooling, no multi-shard stitching, no tolerance knobs beyond a single documented default.

## Metric moved

- Metric: auditability/debug cost of mixed-precision artifacts — kernel-coverage and quant-floor checks move from parse-the-GGUF to read-a-text-file.
- Baseline: no sidecar reader exists (ABSENT; `rg -n -i 'rco-allocation|allocation_sidecar' internal/ggufload/` -> no match, exit 1, verified at study time).
- Direction: down (debug cost); auditability up.
- Reader: `fak-run fak-dev artifact allocation-check --json` (verb name TENTATIVE).

## Dependencies

- Blocked by: nothing.
- Blocks: qwen38-flash-next bringup fail-closed probes (named, not started yet); study row R4 port-evaluation acceptance (histogram match against a shipped `.rco-allocation.txt`).

## Parent context

Denominator parent: #3613 (GSQ-RCO follow-on campaign tracker, lane
gsq-rco-followons).

Convention adoption from the upstream release: HF repo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (GSQ arXiv 2604.18556, RCO arXiv 2605.00649). Campaign lane `gsq-rco-followons`. Consumer: qwen38-flash-next bringup fail-closed probes — not started yet.

## Why this is next

Host-cheap and shard-free: the four manifest texts (public, fetchable at the pinned HF revision) already exercise the full parser surface, so the whole ticket witnesses on this host without a 66–84 GB download. Convention adoption lands before the bringup probes need it and before any GSQ/RCO tooling port-evaluation, whose acceptance is a histogram match against a shipped sidecar.

## Evidence

- [SW-VERIFIED] `internal/ggufload/companion.go`: companion-file handling lives here — `DiscoverDirectoryCompanions` walks the target's directory for `.gguf` candidates only; sentinel errors `ErrNoCompanionFound`/`ErrAmbiguousDraftMechanism`; `knownQuantTags` maps canonical tags to representative bits.
- [SW-VERIFIED] `internal/ggufload/artifact_quant.go`: `ClassifyArtifactQuant`/`ClassifyTensorQuant` classify `[]TensorInfo` into `ArtifactQuant` (UD-* recipe families); the inventory input the validator will consume.
- [SW-VERIFIED] `internal/ggufload/gguf.go:109-132`: `TensorType` constants cover every encoding used by any of the four builds.
- [SW-VERIFIED] HF repo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09`: ships `tensor-allocation/{Q2_0,IQ2_XS,IQ3_XXS,IQ3_S}.txt`; header block plus 1,223 `<tensor>: <TYPE>` rows each; e.g. IQ3_S declares `output.weight: Q6_K`, `blk.0.ffn_down_exps.weight: IQ4_NL`, fixed floor `blk.0.hc_attn_norm.weight: F32` (manifests fetched 2026-10-28).
- [SW-VERIFIED] ABSENT: `rg -n -i 'rco-allocation|allocation_sidecar' internal/ggufload/` -> no match, exit 1.
- [source-only] Upstream provenance: GSQ (arXiv 2604.18556) and RCO (arXiv 2605.00649) papers behind the builds; Apache-2.0 tooling licenses at the pinned tooling main SHAs.

## Next checkable step

Write the parser plus one fixture test: generate a fixture from the fetched `IQ3_S.txt` manifest, parse it, assert tensor count 1,223 and the header fields (target bpw 3.50, quant-type counts) decode. Command: `go test ./internal/ggufload/ -run TestAllocationSidecarParse -count=1`.

## Definition of done

- [ ] [SW] Parser decodes all four cached manifest fixtures (header + 1,223 rows each) into the typed `AllocationSidecar`; malformed and duplicate rows fail closed.
- [ ] [SW] Validator returns a distinct sentinel error per mismatch class (missing sidecar, unknown tensor, undeclared tensor, type mismatch, histogram drift, bpw overrun); tests assert each via `errors.Is`, never error-prose matching (per the anti-brittle-tests rule: closed-contract asserts, sentinel/`errors.Is`, never error prose or a flat `%+v`).
- [ ] [SW] Declared types restricted to loader-admissible `TensorType`s; fixed tensors match the sensitivity-floor set.
- [ ] [SW] `go test ./internal/ggufload/... -count=1` green; new tests declare `// fak-test:runtime` and pass `fak-run fak-testruntime check --staged`; gofmt/vet clean.
- [ ] [SW] Verb wired: `fak-dev artifact allocation-check` (TENTATIVE name) exits 0 on a clean fixture pair and non-zero on each injected mismatch class.

## Working spine

Hermetic fixture tests first — fixtures generated from the four cached manifest texts, shard-1 coverage only; the shard-2 n-gram table (`per_layer_token_embd`) is absent by construction and stays out of scope (named limit, not a defect). Then verb wiring. A live smoke against a real pinned GGUF happens when a shard is available — out of scope for this ticket; everything above witnesses host-side without downloading any shard.

## Acceptance gate

Package suite green, the verb demonstrates exit 0 on a clean fixture pair and non-zero on each injected mismatch class, and the closure commit carries the citations below.

## Witness

- Envelope at filing: `go test -run ZZZ_NoTestMatches -count=1 ./internal/ggufload ./cmd/fak-dev` -> ok/ok (both test binaries compile, 2026-10-28)
- `go test ./internal/ggufload/... -count=1`
- `fak-run fak-dev artifact allocation-check --fixture <dir> --json` (clean pair => exit 0; each injected mismatch class => exit non-zero with the sentinel class named)
- `rg -n -i 'rco-allocation|allocation_sidecar' internal/ggufload/` (reader now present; flips ABSENT -> present)

## Likely files

- internal/ggufload/companion.go
- internal/ggufload/allocation_sidecar.go (new)
- internal/ggufload/allocation_sidecar_test.go (new)
- cmd/fak-dev/ (verb wiring; exact file TENTATIVE)

## Closure binding

The closing commit cites this ticket, its GitHub mirror, and `(fak ggufload)`.

## Work estimate

Estimate: 2 points (parser + validator + hermetic fixture tests + verb wiring).

## Overall completion contribution

Contribution: 2/8 points of the gsq-rco follow-on set (ESTIMATED; siblings tracked under lane `gsq-rco-followons`, denominator parent #3613).

## Completion standard

Production.

## Target operating envelope

- test packages: >= 2 packages

## Witnessed operating envelope

- test packages: 2 packages

github_issue: anthony-chaudhary/fak#14093

github_issue: anthony-chaudhary/fak#14093
