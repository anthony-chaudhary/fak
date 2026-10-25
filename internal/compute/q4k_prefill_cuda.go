//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"os"
	"sync"
)

// q4k_prefill_cuda.go holds the CUDA Q4_K prefill arm seam (P-04, fak#13983): init resolves
// FAK_CUDA_Q4K_PREFILL once and logs the arm, and SetCUDAQ4KPrefillArm switches it at runtime so
// one test process can drive every arm. A refused selection latches its error and sets the C arm
// to refused, so every P >= 128 Q4_K GEMM fails with that error instead of falling back.

// cudaQ4KPrefillArmRefused is FCUDA_Q4K_PREFILL_ARM_REFUSED in cuda/quant_matmul.cuh.
const cudaQ4KPrefillArmRefused = 80003

// cudaQ4KPrefillArm and cudaQ4KPrefillArmErr mirror the C arm; cudaQ4KPrefillArmMu guards them
// so a GEMM status can be mapped to its refusal with or without cudaMu held.
var (
	cudaQ4KPrefillArmMu  sync.Mutex
	cudaQ4KPrefillArm    = CUDAQ4KPrefillCompensated
	cudaQ4KPrefillArmErr error
)

func initCUDAQ4KPrefillArm(sm int) {
	arm, err := resolveCUDAQ4KPrefillArm(os.Getenv(CUDAQ4KPrefillEnv), sm)
	applyCUDAQ4KPrefillArm(arm, err)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak-cuda: %v; Q4_K prompt panels of 128+ tokens are refused\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "fak-cuda: q4k prefill arm=%s\n", arm)
}

func applyCUDAQ4KPrefillArm(arm CUDAQ4KPrefillArm, err error) {
	code := cudaQ4KPrefillArmRefusedCode
	if err == nil {
		code = cudaQ4KPrefillArms[arm].code
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_set_q4k_prefill_arm(C.int(code))
	cudaQ4KPrefillArmMu.Lock()
	cudaQ4KPrefillArm, cudaQ4KPrefillArmErr = arm, err
	cudaQ4KPrefillArmMu.Unlock()
}

// SetCUDAQ4KPrefillArm switches the P >= 128 Q4_K GEMM arm at runtime. A refused arm returns
// its typed error and leaves the current arm in place.
func SetCUDAQ4KPrefillArm(arm CUDAQ4KPrefillArm) error {
	sm := 0
	if cudaDev != nil {
		sm, _ = cudaNVFP4SM(cudaDev.tier)
	}
	if err := CheckCUDAQ4KPrefillArm(arm, sm); err != nil {
		return err
	}
	applyCUDAQ4KPrefillArm(arm, nil)
	fmt.Fprintf(os.Stderr, "fak-cuda: q4k prefill arm=%s (runtime)\n", arm)
	return nil
}

// CUDAQ4KPrefillArmInUse reports the arm the P >= 128 Q4_K GEMM runs, or the latched refusal.
func CUDAQ4KPrefillArmInUse() (CUDAQ4KPrefillArm, error) {
	cudaQ4KPrefillArmMu.Lock()
	defer cudaQ4KPrefillArmMu.Unlock()
	return cudaQ4KPrefillArm, cudaQ4KPrefillArmErr
}

// cudaQ4KPrefillRefusal is the typed error a FCUDA_Q4K_PREFILL_ARM_REFUSED status carries.
func cudaQ4KPrefillRefusal() error {
	arm, err := CUDAQ4KPrefillArmInUse()
	if err != nil {
		return err
	}
	return &CUDAQ4KPrefillArmError{Arm: string(arm), Err: ErrCUDAQ4KPrefillArmUnimplemented}
}
