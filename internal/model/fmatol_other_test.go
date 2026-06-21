//go:build !amd64

package model

// fmaCrossPathTol bounds the divergence allowed between two algebraically-identical f32 code
// paths in the "bit-exact" rungs on arches with mandatory hardware FMA (arm64, ppc64, etc.).
//
// On those arches the gc compiler auto-fuses a*b+c into a single-rounded FMADD — and it may
// fuse two algebraically-identical expressions DIFFERENTLY depending on each one's syntactic
// shape, so the same math computed by two distinct code paths (e.g. the proven decode vs the
// profiler twin, or a repositioned K vs a freshly recomputed RoPE) can diverge by a few ULP.
// Measured on Apple M3 Pro: the reposition invariant drifts ≤1e-6, and the profiler twin
// drifts ≤3.1e-5 accumulated over 24 decode steps with ZERO argmax mismatches — i.e. the
// divergence is strictly sub-token (greedy/argmax outputs stay identical; the functional
// rungs in these tests pass exactly on every arch). Only the byte-exact equality is
// arch-specific. 1e-4 sits ~3× above the observed FMA noise floor and far below any
// divergence a real algorithmic bug would produce (which flips tokens / moves logits by ≫1).
const fmaCrossPathTol = 1e-4
