<!-- fak-compute-key: alloc-ceiling-and-test-only-enable-gates -->
```routing
lane: compute
paths: ["internal/compute/backend_alloc_ceiling_test.go", "internal/model/test_only_enable_fields_test.go"]
expected_steps: 6
```

# test(compute): gate the hot-path copy class with a per-dtype allocation ceiling and a test-only-enable detector

GitHub: [#13876](https://github.com/anthony-chaudhary/fak/issues/13876)

Process cause: verification-gap

## Value

- **For:** every native backend and the model step code; this is the process control for the sweep's bug class.
- **Problem:** the 2026-10-16 forensics found two failure mechanisms that no gate catches.
  1. **Unmeasured allocation.** `12dc1386da` (2026-06-23) added cpu-ref K-quant MatMul with only correctness witnesses; every call copied the whole weight until `19eab47217` (2026-10-16). The sibling fix `3f9a0e3d7` (2026-10-12, found by `perf record` on hardware) pinned the helper, not its callers, so 12 sibling sites survived 4 more days. `make ci` has no allocation target; `BenchmarkCPURefMatMulGEMV` is dense f32 and ungated.
  2. **Built, tested, never enabled.** `gatherOps.indexerKeyBudgetBytes` is written only in `v41_indexer_score_device_test.go`, so the resident scorer never runs in production (TICKET-01). Tests prove the path works; nothing proves production selects it.
- **Better because:** a table-driven ceiling over every supported weight dtype catches the copy class at review time, including future dtypes; a detector for gating fields whose only writers are `_test.go` files catches the presence-vs-invokability class.
- **Witness:** both tests in the default `go test` set; the ceiling red against `12dc1386da`-style code, the detector red on today's `indexerKeyBudgetBytes`.

## Problem frame

- Centrality: Core
- P1 advanced: advanced - prevents regressions on the native decode path.
- P2 preserved: preserved - no runtime behavior change.
- P3 advanced: advanced - generalizes `TestCPURefMatMulKQuantReadsWeightInPlace` (`19eab47217`) to every dtype and backend.
- P4 advanced: advanced - turns a hardware-profile discovery into a pre-merge software gate.

## Why this is next

Each instance in this sweep was found late, by hand, on hardware. Without a gate, the next reference path or test-only knob repeats the cycle.

## Current state

Classification: **PARTIAL**. A K-quant (Q4_K/Q5_K/Q6_K) cpu-ref ceiling exists from `19eab47217`; Q2_K, Q3_K, IQ2_XXS, Q2_0, raw IQ, the Vulkan upload path, and any test-only-enable detection are absent.

## Core through-line

Supported dtype list -> one table-driven `AllocsPerRun` ceiling per backend MatMul and upload (bytes < weight bytes) -> red on any per-call weight copy. Separately: AST scan of non-test packages for struct fields that gate a fast path -> writers outside `_test.go` must exist or be allowlisted with a reason.

## Working spine

`go test ./internal/compute/ ./internal/model/` runs both gates; a seeded regression (re-adding a copy, or a test-only budget) turns them red.

## Scope

The two gate tests and an allowlist file with reasons.

## Gold-plating boundary

- No benchstat CI, no timing thresholds (allocation counts are deterministic; timings are not).
- The detector starts as an allowlisted report of fields named `*Budget*`, `*Enable*`, `*Enabled` read in a guard; it does not try to prove arbitrary reachability.

## Done condition

- [ ] Ceiling covers every dtype the cpu-ref and Vulkan backends accept.
- [ ] Detector flags `indexerKeyBudgetBytes` until TICKET-01 lands, via an allowlist entry that names TICKET-01.
- [ ] A seeded copy and a seeded test-only field each turn the gates red (verified once, recorded in the commit).
- [ ] Tests declare their runtime class.

## Definition of done

Both gates in the default test set, documented in the sweep README, with the allowlist owned by tickets.

## Witness

`go test ./internal/compute/ -run 'AllocCeiling' -count=1 && go test ./internal/model/ -run 'TestOnlyEnable' -count=1`

## Likely files

- `internal/compute/backend_alloc_ceiling_test.go` (new)
- `internal/model/test_only_enable_fields_test.go` (new)

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Both gates red on a seeded regression and green on trunk with an explicit allowlist.

## Closure binding

Close with the landing commit and both test names.

## Work estimate

Estimate: 3 points (small) - about 1 day.

## Overall completion contribution

3/26 points of the hot-path copy sweep (process control that keeps the other fixes fixed).

## Completion standard

development

github_issue: anthony-chaudhary/fak#13876
