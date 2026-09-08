//go:build amd64

// Package deltanetavx512 owns the Go assembly implementation of the canonical
// 128x128 Gated-DeltaNet head transition. Keeping the assembly in a leaf package
// lets compute use cgo-backed device implementations (such as Vulkan) in the same
// binary; the Go toolchain rejects cgo and Go assembly in one package.
package deltanetavx512

// Step fuses one canonical 128x128 Gated-DeltaNet head transition. The caller
// must check AVX-512F plus OS ZMM-state support and validate all buffer lengths.
func Step(st, qn, kn, vh, od, kvmem, delta *float32, bt, gate float32) {
	deltaNetStep(st, qn, kn, vh, od, kvmem, delta, bt, gate)
}

//go:noescape
func deltaNetStep(st, qn, kn, vh, od, kvmem, delta *float32, bt, gate float32)
