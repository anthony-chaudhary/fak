<!-- fak-compute-key: cpuref-per-call-scratch-and-rope -->
```routing
lane: compute
paths: ["internal/compute/cpuref.go", "internal/compute/cpuref_matmul_test.go"]
expected_steps: 5
```

# perf(compute): reuse cpu-ref attention, Q8 activation and RoPE scratch instead of rebuilding it every call

GitHub: [#13877](https://github.com/anthony-chaudhary/fak/issues/13877)

Process cause: verification-gap

## Value

- **For:** the cpu-ref backend: the `--backend cpu` path, the CPU auto fallback, V4.1 prefix reuse (cpu-ref is its only qualified backend), and every parity test that uses it as the reference.
- **Problem:** after `19eab47217` and `300a373aed` removed the per-call weight copy, three per-call costs remain in `internal/compute/cpuref.go`:
  - `Attention` allocates `scores := make([]float32, nPos)` inside the per-query-head loop (:469).
  - Q8_0 `MatMul` and `BatchedMatMul` quantize the activation into fresh slices on every call via `quantizeVecQ8` (:207, :293, :697). The existing `PrepareMatMul`/`plan.Run` workspace path is not used.
  - `ropeRow` (:652-660) recomputes the cos/sin table with `math.Pow`/`Cos`/`Sin` on every RoPE call, and Q and K at the same position each recompute it.
- **Today:** ESTIMATED on a 28-layer, 32-head, 4K-context model: about 14 MB and 900 allocs/token from attention scores, about 1 MB and 400 allocs/token from Q8 scratch, and about 11K transcendentals plus 112 allocs/token from RoPE. Unmeasured.
- **Better because:** each is a reusable buffer or a cache keyed on (theta, headDim, pos); the math is unchanged.
- **Witness:** `AllocsPerRun` ceilings for `Attention`, Q8_0 `MatMul` and `RoPE`, alongside the per-dtype ceiling in `TestCPURefMatMulKQuantReadsWeightInPlace`.

## Problem frame

- Centrality: Enabling (fak-native CPU decode and fast, allocation-stable reference parity)
- P1 advanced: advanced - lowers per-token allocation on the CPU native path.
- P2 preserved: preserved - cpu-ref stays the bit-exact reference.
- P3 advanced: advanced - reuses the existing `plan.Run` workspace pattern.
- P4 preserved: preserved - bit-identical outputs are required; no speed claim without a benchmark delta.

## Why this is next

These are the in-file leftovers after the cpu-ref whole-weight copy fix. The session that owns `cpuref.go` handed them off on 2026-10-16.

## Current state

Classification: **PARTIAL**. K-quant/IQ MatMul now reads weights in place with an allocation ceiling. Attention, Q8_0 activation and RoPE still allocate or recompute on every call.

## Core through-line

Per-backend reusable scratch -> one `scores` buffer per Attention call reused across heads -> Q8 activation quantized into workspace -> RoPE table cached per (theta, headDim, pos) -> outputs identical to the parent.

## Working spine

cpu-ref decode step -> Attention / MatMul Q8_0 / RoPE -> allocation ceilings green -> parity green.

## Scope

The three per-call costs above, in `cpuref.go`.

## Gold-plating boundary

- No SIMD, threading, or kernel rewrite.
- No change to the reference math or rounding.

## Done condition

- [ ] Attention reuses one scores buffer across heads.
- [ ] Q8_0 MatMul/BatchedMatMul quantize into reused workspace.
- [ ] RoPE cos/sin cached and shared by Q and K at the same position.
- [ ] `AllocsPerRun` ceilings for all three, red on the parent.
- [ ] Bit-identical outputs against the parent.

## Definition of done

Ceilings and parity green in the default `go test ./internal/compute/` set; before/after allocs recorded in the commit.

## Witness

`go test ./internal/compute/ -run 'CPURef.*(Attention|Q8|RoPE|InPlace)' -count=1`

## Likely files

- `internal/compute/cpuref.go`
- `internal/compute/cpuref_matmul_test.go`

## Parent context

Parent campaign: #12640. Sweep record: `docs/tickets/hot-path-copy-sweep/README.md`.

## Acceptance gate

Ceilings red on the parent and green on the candidate; bit-identical parity.

## Closure binding

Close with the landing commit and the ceiling test names.

## Work estimate

Estimate: 2 points (small) - about half a day.

## Overall completion contribution

2/28 points of the hot-path copy sweep.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13877
