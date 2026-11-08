package compute

import "github.com/anthony-chaudhary/fak/internal/fp4meta"

// nvfp4CUBLASLtPolicy delegates architecture/runtime policy to the neutral
// metadata layer so capability validation and dispatch share one contract.
func nvfp4CUBLASLtPolicy(sm, libraryVersion int) (eligible, skipAlgo66 bool) {
	return fp4meta.CUBLASLtNVFP4Policy(sm, libraryVersion)
}

// NVFP4CUBLASLtEligible exports the eligibility half of nvfp4CUBLASLtPolicy so
// capability adjudicators (fp4meta.CUDACapabilities) gate on the same floors
// the CUDA W4A4 dispatch enforces.
func NVFP4CUBLASLtEligible(sm, libraryVersion int) bool {
	eligible, _ := nvfp4CUBLASLtPolicy(sm, libraryVersion)
	return eligible
}
