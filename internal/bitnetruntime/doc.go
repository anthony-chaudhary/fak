// Package bitnetruntime admits Microsoft's BitNet runtime (bitnet.cpp) as an
// EXTERNAL DELEGATE, and never as something fak owns.
//
// The distinction is the whole point. fak ships no ternary kernel, no BitNet
// build, and no artifact format; the most this package can conclude is that a
// named external runtime, on this host, could execute a given artifact — a
// delegation, not a capability. So its success outcome is literally
// "delegate", and the strongest claim it licenses is "a named runtime owns this
// artifact's realization". It never licenses a statement about what the
// artifact IS (that is internal/bitnetmeta's), about how the artifact was
// produced, or about how fast it runs: a capability probe cannot measure
// hardware.
//
// Three independent axes decide a delegation, and collapsing any of them is the
// failure this package exists to prevent:
//
//   - the RUNTIME — which kernels a specific build actually carries. bitnet.cpp
//     is a compile-time selection, so a version banner is evidence of a version
//     and of nothing else. A build that does not report its kernels abstains;
//     the "usual" kernel set is never assumed from a recent version.
//   - the ARTIFACT's kernel — bitnet.cpp weights are packed FOR a kernel
//     (i2_s, tl1, tl2), so an artifact packed for one is not servable by
//     another. A kernel this contract has never heard of abstains rather than
//     falling back to a familiar one.
//   - the HOST — each kernel dispatches on a specific CPU feature per
//     architecture (avx2 on amd64, neon on arm64), and tl1/tl2 are single-
//     architecture kernels. A runtime that carries a kernel the host cannot
//     dispatch is an explicit unsupported, not a silent downgrade to i2_s.
//
// Model-family checking is deliberately narrow for the same reason: bitnet.cpp
// serves the trained-ternary ("1.58-bit") family, so a binary or uniform int2/
// int4 artifact gets a named unsupported rather than being waved through as
// "low-bit". Every input lands on one of four typed outcomes — delegate,
// unsupported, abstain, refuse — and nothing silently falls back.
//
// This package is pure. It performs no process execution itself: the caller
// injects a Prober that collects the delegate's own probe output, so the same
// contract is driven by a real bitnet.cpp in cmd/fak and by a fake runtime in
// the tests. It defines no fak-owned artifact format and requires no
// conversion.
//
// Tier: primitive (1) - see internal/architest. Stdlib-only.
package bitnetruntime
