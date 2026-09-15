package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// q4kHostCopyReleasable decides whether a resident Q4_K tensor's host packed bytes may be dropped
// after its device upload, so the model holds a SINGLE residency (device only) instead of two.
//
// It is the model-side half of compute.Q4KSingleResidencyRelease: that pure predicate answers the
// tier + knob question, and this adds the ROUTING precondition that makes the release provable. All
// four gates must hold:
//
//   - envEnabled: the operator turned on FAK_Q4K_FREE_CPU. Off by default — the CPU Q4_K
//     GEMV/GEMM fallbacks read qt.raw and panic legibly on nil (requireRawCPU, #1067), so the
//     default must keep the host copy.
//   - halQ4KRouted: the session stages eligible Q4_K weights as RAW resident device tensors and
//     the device Backend.MatMul serves every q4_k matmul (useHALQ4KWeights). Without this the CPU
//     fallback can still be reached and a nil raw is a correctness bug, not a memory saving.
//   - resident: qt.lazy == nil, i.e. a permanent host-resident tensor — not a streamed lazy range
//     whose bytes were never a resident copy to begin with.
//   - uploadOK: the device Upload returned a non-nil tensor. A failed upload leaves the device copy
//     NON-authoritative, so the host bytes must stay.
//
// The tier check ("integrated:", a unified-memory APU) is the reason the release is worth doing at
// all: on UMA the host copy and the device copy draw on the same physical pool, so keeping both
// doubles the footprint. On a discrete device the host copy is separate DRAM and a free CPU
// fallback, so the release never fires.
func q4kHostCopyReleasable(s *Session, qt *q4kTensor, uploadOK bool) bool {
	if s == nil || qt == nil || !uploadOK {
		return false
	}
	if !compute.Q4KSingleResidencyEnvEnabled() {
		return false
	}
	if !s.useHALQ4KWeights() {
		return false
	}
	if qt.lazy != nil {
		return false
	}
	return compute.Q4KSingleResidencyRelease(s.Backend.Tier(), true)
}
