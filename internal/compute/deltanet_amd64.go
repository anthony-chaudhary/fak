//go:build amd64

package compute

import syscpu "golang.org/x/sys/cpu"

// hasDeltaNetSIMD reports actual executable support. x/sys/cpu includes the OSXSAVE/XCR0
// check, so HasAVX512F is false when the processor has ZMM registers but the OS cannot save them.
func hasDeltaNetSIMD() bool {
	return syscpu.X86.HasAVX512F
}

func tryDeltaNetSIMD(st, qn, kn, vh []float32, bt, g float32, od, kvmem, delta []float32) bool {
	const dim = 128
	if !hasDeltaNetSIMD() || len(st) < dim*dim || len(qn) != dim || len(kn) < dim ||
		len(vh) != dim || len(od) < dim || len(kvmem) < dim || len(delta) < dim {
		return false
	}
	deltaNetStepAVX512(
		&st[0], &qn[0], &kn[0], &vh[0], &od[0], &kvmem[0], &delta[0], bt, g,
	)
	return true
}

// deltaNetStepAVX512 fuses one canonical 128x128 Gated-DeltaNet head transition.
//
//go:noescape
func deltaNetStepAVX512(st, qn, kn, vh, od, kvmem, delta *float32, bt, gate float32)
