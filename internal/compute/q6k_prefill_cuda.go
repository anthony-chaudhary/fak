//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"os"
)

// q6k_prefill_cuda.go is the default Q6_K prompt-panel arm seam: init resolves
// FAK_CUDA_Q6K_PREFILL once, SetCUDAQ6KPrefillArm switches it for tests, and
// BatchedMatMul offers eligible panels through q6KPrefillMMQStatus.

// cudaQ6KPrefillArmCur mirrors the C arm; cudaMu guards both.
var cudaQ6KPrefillArmCur = CUDAQ6KPrefillDequant

func initCUDAQ6KPrefillArm(sm int) {
	arm, err := resolveCUDAQ6KPrefillArm(os.Getenv(CUDAQ6KPrefillEnv), sm)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak-cuda: %v; Q6_K prefill keeps arm=%s\n", err, arm)
	}
	applyCUDAQ6KPrefillArm(arm)
	fmt.Fprintf(os.Stderr, "fak-cuda: q6k prefill arm=%s\n", arm)
}

func applyCUDAQ6KPrefillArm(arm CUDAQ6KPrefillArm) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_set_q6k_prefill_arm(C.int(cudaQ6KPrefillArmCodes[arm]))
	cudaQ6KPrefillArmCur = arm
}

// SetCUDAQ6KPrefillArm switches the Q6_K P>=8 BatchedMatMul arm at runtime. A refused arm
// returns its error and leaves the current arm in place.
func SetCUDAQ6KPrefillArm(arm CUDAQ6KPrefillArm) error {
	joinDeferredBackends()
	sm := 0
	if cudaDev != nil {
		sm = cudaBackendSM(cudaDev)
	}
	if err := CheckCUDAQ6KPrefillArm(arm, sm); err != nil {
		return err
	}
	applyCUDAQ6KPrefillArm(arm)
	return nil
}

// CUDAQ6KPrefillArmInUse reports the arm the Q6_K P>=8 BatchedMatMul offers panels to.
func CUDAQ6KPrefillArmInUse() CUDAQ6KPrefillArm {
	joinDeferredBackends()
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return cudaQ6KPrefillArmCur
}

// q6KPrefillMMQStatus offers an eligible panel to the default MMQ arm. ran=false means nothing
// was allocated or enqueued and the caller must run the legacy projection. Caller holds cudaMu.
func (c *cudaBackend) q6KPrefillMMQStatus(w Tensor, wb *cudaBuf, x, y Tensor, p int) (rc int, ran bool) {
	if !cudaQ6KPrefillMMQEligible(cudaQ6KPrefillArmCur, w.Dtype, p, cudaQ56KMMQSelected(w, p)) ||
		c.capturing || len(w.Shape) != 2 || w.Layout != RowMajor || w.Quant == nil || w.Quant.Block != 256 ||
		len(x.Shape) != 2 || x.Shape[0] != p || x.Shape[1] != w.Shape[1] || x.Dtype != F32 ||
		x.Layout != RowMajor || x.Quant != nil {
		return 0, false
	}
	rc = int(C.fcuda_q6k_prefill_mmq_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y),
		C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(p)))
	if rc == cudaQ6KPrefillMMQDeclined {
		return 0, false
	}
	return rc, true
}

func (c *cudaBackend) cudaQ6KPrefillMMQArmForTest() bool {
	return CUDAQ6KPrefillArmInUse() == CUDAQ6KPrefillMMQ
}
