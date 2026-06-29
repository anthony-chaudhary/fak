---
title: "sm80 CUDA 13 issue #973 closure witness"
description: "A scrubbed A100-class sm_80 CUDA 13.0 witness for the two issue #973 residuals: CUDA graph decode parity and Q4_K batched matmul."
date: 2026-06-29
---

# sm80 CUDA 13 issue #973 closure witness

Issue #973 tracked two residual failures from the first A100-class sm_80 CUDA 13.0
weightless `internal/compute` witness map:

- `TestCUDAGraphDecodeParity`
- `TestCUDAQ4KBatchedMatMulApproxMatchesRef`

The separate f32 op-level failures from that run stay tracked by #972.

## Reached gate

On an A100-class sm_80 GPU node with CUDA 13.0, the targeted #973 gate passed:

```sh
FAK_CUDA_ARCH=sm_80 \
CUDA_VISIBLE_DEVICES=0 \
CUDA_HOME=/usr/local/cuda-13.0 \
FAK_CUDA_GRAPH=1 \
go test -tags cuda -count=1 -v \
  -run '^TestCUDAGraphDecodeParity$|^TestCUDAQ4KBatchedMatMulApproxMatchesRef$' \
  ./internal/compute/
```

Recorded results:

| Witness | Result | Recorded detail |
|---|---:|---|
| `TestCUDAGraphDecodeParity` | PASS | graph-replayed == eager argmax-exact; logit cosine >= 0.999; `device=cuda tier=sm_80 class=approx` |
| `TestCUDAQ4KBatchedMatMulApproxMatchesRef` | PASS | cosine `1.00000000`, maxAbs `1.22e-03`, gate `0.9950` |

The package run exited green:

```text
PASS
ok github.com/anthony-chaudhary/fak/internal/compute 1.261s
```

## Commit scope

The GPU-tested commit was `f8610f4cccf958e5c61ff3012dab88f01f396f66`.
At closure time, current `origin/main` was
`0da1f38a72306ed262d6b837ad5221f18ce1b730`; there were no changes between those
commits in `internal/compute`, `go.mod`, or `go.sum`:

```sh
git diff --name-only f8610f4cccf958e5c61ff3012dab88f01f396f66..0da1f38a72306ed262d6b837ad5221f18ce1b730 -- internal/compute go.mod go.sum
```

That makes the targeted package witness source-equivalent to current trunk for the two
#973 tests.

## Not claimed

This note does not claim a fresh green run of the full original
`go test -tags cuda -run 'CUDA|HALDevice'` umbrella. It closes only the two residuals
owned by #973; #972 remains the tracker for the f32 shape failures from the same first
sm_80 map.
