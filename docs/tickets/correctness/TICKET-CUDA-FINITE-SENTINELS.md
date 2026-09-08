# fix(compute): replace finite CUDA reduction sentinels with true bounds

<!-- fak-compute-key: cuda-finite-reduction-sentinels -->

```routing
lane: internal/compute
paths: ["internal/compute/cuda_kernels.cu", "internal/compute/cuda_flash_test.go", "internal/compute/cuda_argmax_test.go", "internal/compute/cuda_finite_sentinel_test.go"]
expected_steps: 7
```

## Current state

Four CUDA reductions initialize a maximum with `-1e30f`: the retained naive
attention baseline, decode flash attention, DSA sparse attention, and argmax.
IEEE-754 float32 values below `-1e30` are finite and valid. Attention then keeps
an empty normalizer and can return zeros; argmax can retain index zero even when
a later element is larger.

A physical sm89 witness reached this state through the decode flash ABI using
finite Q/K values whose dot products were below the sentinel: expected norm
`1.92133251`, CUDA norm `0`, cosine `0`, and maximum absolute error `0.125`.
The neighboring split-KV path was fixed under #11100, leaving these independent
production and baseline reductions exposed.

## Problem Frame

- Centrality: Core compute correctness; these reductions choose attention probabilities and output tokens.
- P1 Context: Advanced; ordinary random tests do not cross the assumed score floor, so the defect hides behind otherwise stable softmax algebra.
- P2 Net value: advanced - one shared invariant removes silent zero vectors and incorrect token selection without changing supported APIs.
- P3 Adaptation: preserved - change only empty/max initializers and add focused extreme-finite regressions.
- P4 Operations: Preserved; no allocation, launch geometry, scheduling, or transfer behavior changes.

## Core through-line

Replace every CUDA maximum accumulator that semantically means “no finite value
seen yet” with true negative infinity, then prove attention and argmax accept
finite inputs below `-1e30` on a physical CUDA device.

## Why this is next

The #11100 physical regression proved the same invalid assumption in split-KV
attention. Its independent sequential flash baseline also returned a zero vector,
so the remaining production exposure is witnessed rather than hypothetical.

## Parent context

Follow-up discovered while physically verifying #11100.

## Working spine

Extreme finite Q/K or logits -> CUDA reduction initializer -> online softmax or
argmax update -> correct nonzero attention output or maximum index -> device receipt.

## Gold-plating boundary

Do not redesign attention tiling, change reduction order, tune occupancy, alter
NaN policy, or sweep Vulkan/Metal shaders in this leaf. Do not conflate masked
`-1e300` double values with empty float32 reduction state.

## Done condition

- [x] Naive attention, decode flash attention, DSA sparse attention, and CUDA argmax use true negative infinity for empty/max state.
- [x] A physical CUDA attention witness with finite scores below `-1e30` matches an independent CPU oracle at cosine at least 0.99999 and maximum absolute delta at most `1e-5`.
- [x] A CUDA argmax witness over all-finite values below `-1e30` returns the actual maximum index.
- [x] Source contracts prevent reintroduction of finite reduction sentinels.
- [x] Focused tests, full compute tests, vet, sm80/sm89/sm90 compilation, boundary, and leak audits pass.

## Definition of Done

- [x] All four CUDA maximum initializers represent a true empty bound.
- [x] Attention and argmax edge cases execute correctly on a physical CUDA device.
- [x] Default-CI source contracts and the full compute package remain green.

## Acceptance gate

The physical attention and argmax edge witnesses pass, all three supported CUDA
architectures compile, and the focused/full package checks exit zero.

## Closure binding

The DCO-signed resolving commit cites the created public issue and carries
`(fak compute)`.

## Witness

```bash
go test ./internal/compute -run 'CUDA.*(Attention|Argmax).*Extreme|FiniteReductionSentinel' -count=1
go test ./internal/compute -count=1
go vet ./internal/compute/...
```

Physical CUDA execution must additionally invoke the real exported attention
and argmax ABIs with the edge inputs above; source inspection alone is not a
device witness.

Witnessed on an RTX 4070 Laptop GPU (sm89): flash, naive attention, and DSA each
returned cosine `1` and maximum absolute delta `0` against independent uniform-
softmax oracles for finite scores below `-1e30`; CUDA argmax returned index `1`
for `[-3e30, -2e30, -2.5e30, -4e30]`. Whole-translation-unit NVCC compilation
also passed for sm80 and sm90. The full compute package passed in `117.164s`.

## Likely files

- `internal/compute/cuda_kernels.cu:2057` (`k_attention` block maximum)
- `internal/compute/cuda_kernels.cu:2142` (`k_flash_attention` online softmax)
- `internal/compute/cuda_kernels.cu:2571` (`k_dsa_sparse_attend` online softmax)
- `internal/compute/cuda_kernels.cu:2740` (`k_argmax` maximum)
- `internal/compute/cuda_flash_test.go`
- `internal/compute/cuda_argmax_test.go`
- `internal/compute/cuda_finite_sentinel_test.go`

## Blast radius and affected lanes

- Affected: CUDA attention and argmax inside `internal/compute`.
- Unaffected: CPU, Vulkan, Metal, model loading, serving APIs, and kernel launch geometry.
- Fallback: existing normal-range paths remain unchanged; unsupported CUDA builds continue to exclude tagged device tests.

## Lane

internal/compute

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points.

## Completion standard

production

## Target operating envelope

physical CUDA extreme-finite attention and argmax pass rate: = 100 percent

## Witnessed operating envelope

physical CUDA extreme-finite attention and argmax pass rate: = 100 percent
