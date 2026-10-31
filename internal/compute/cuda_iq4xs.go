//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

// iq4xsLaunchesForTest reports the IQ4_XS MMVQ and dequant-split GEMM launch counts
// (cuda/iq4xs_mmvq.cuh), so the parity witness can tell which arm produced an output.
func (c *cudaBackend) iq4xsLaunchesForTest() (mmvq, dequantGEMM uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_iq4xs_mmvq_launch_count()), uint64(C.fcuda_iq4xs_dequant_gemm_launch_count())
}
