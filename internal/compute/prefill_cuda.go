//go:build cuda

package compute

// prefill_cuda.go — the CUDA half of the prefill graph-capture wiring (issue #9). It is
// compiled ONLY under -tags cuda and does nothing but pin the contract: the CUDA backend,
// which already exposes GraphBegin/GraphEndLaunch/GraphReset (cuda.go), MUST satisfy the
// PrefillGraphCapturer seam declared in the always-compiled prefill.go. If a future edit
// changes either side's signature, this assertion fails the CUDA build instead of silently
// falling back to eager prefill on the device. On a non-CUDA build this file is absent, so
// the default artifact stays pure-Go and compiles without it (the guarded-stub rule).
var _ PrefillGraphCapturer = (*cudaBackend)(nil)
