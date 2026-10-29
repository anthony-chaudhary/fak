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

// Runtime toolkit/shape/capture/heuristic refusal, not an alternative-arm decline.
const cudaQ4KFP8LTRefused = 80005

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
	if arm == CUDAQ4KPrefillFP8LT {
		fmt.Fprintln(os.Stderr, "fak-cuda: fp8lt experimental scale=per-tensor operands=e4m3 output=f32; numerical qualification pending")
	}
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
		sm = cudaBackendSM(cudaDev)
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

// cudaBackendSM is the device compute capability from its tier ("sm_89" -> 89), 0 if unknown.
func cudaBackendSM(c *cudaBackend) int {
	sm, _ := cudaNVFP4SM(c.tier)
	return sm
}

// q4kMMQLaunchesForTest reports completed mmq-arm launches, so the parity witness can refuse a
// run where the compensated route produced the output.
func (c *cudaBackend) q4kMMQLaunchesForTest() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_q4k_mmq_launch_count())
}

// q4kFP8LTLaunchesForTest counts accepted enqueues, not completed device work.
func (c *cudaBackend) q4kFP8LTLaunchesForTest() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_q4k_fp8lt_launch_count())
}

// cudaQ4KScratch is the scratch one Q4_K GEMM of rows rows allocates on the arm in use. The mmq
// arm allocates its q8_1 activation (rows padded to 512 values, plus one 128-column tile). Its
// stream-k fixup buffer depends on the SM count, so a miss there reports the activation bytes.
func cudaQ4KScratch(out, in, rows int) cudaGEMMScratch {
	if arm, err := CUDAQ4KPrefillArmInUse(); rows >= 128 && err == nil && arm == CUDAQ4KPrefillFP8LT {
		padded := (rows + 15) / 16 * 16
		bytes := out*in + padded*in + 16 // E4M3 W/X and device scales, no f32 weight
		if padded != rows {
			bytes += padded * out * F32.Bytes()
		}
		return cudaGEMMScratch{main: bytes}
	}
	if arm, err := CUDAQ4KPrefillArmInUse(); rows >= 128 && err == nil && arm == CUDAQ4KPrefillMMQ {
		const q81Block, q81Values = 144, 128 // block_q8_1_mmq
		padded := (in + 511) / 512 * 512
		return cudaGEMMScratch{main: rows*padded/q81Values*q81Block + 128*q81Block}
	}
	if rows < 4 && CUDAQ4KDecodeArmInUse() == CUDAQ4KDecodeMMVQ {
		return cudaGEMMScratch{main: rows * (in / 32) * 36} // block_q8_1 per 32 values
	}
	return cudaKQuantScratch(out, in, rows)
}

// cudaQ4KDecodeArm mirrors the C P < 4 Q4_K GEMV arm (fak#8635); cudaQ4KPrefillArmMu guards it.
var cudaQ4KDecodeArm = CUDAQ4KDecodeF32

// initCUDAQ4KDecodeArm resolves FAK_CUDA_Q4K_DECODE once. An unknown value keeps the f32 arm and
// says so on stderr: decode has no refusal path a whole serve could survive.
func initCUDAQ4KDecodeArm() {
	arm, err := ParseCUDAQ4KDecodeArm(os.Getenv(CUDAQ4KDecodeEnv))
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak-cuda: %v; Q4_K decode keeps arm=%s\n", err, CUDAQ4KDecodeF32)
		arm = CUDAQ4KDecodeF32
	}
	applyCUDAQ4KDecodeArm(arm)
	if arm != CUDAQ4KDecodeF32 {
		fmt.Fprintf(os.Stderr, "fak-cuda: q4k decode arm=%s\n", arm)
	}
}

func applyCUDAQ4KDecodeArm(arm CUDAQ4KDecodeArm) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_set_q4k_decode_arm(C.int(cudaQ4KDecodeArmCodes[arm]))
	cudaQ4KPrefillArmMu.Lock()
	cudaQ4KDecodeArm = arm
	cudaQ4KPrefillArmMu.Unlock()
}

// SetCUDAQ4KDecodeArm switches the P < 4 Q4_K GEMV arm at runtime. An unknown arm returns
// ErrCUDAQ4KDecodeArmUnknown and leaves the current arm in place.
func SetCUDAQ4KDecodeArm(arm CUDAQ4KDecodeArm) error {
	if err := CheckCUDAQ4KDecodeArm(arm); err != nil {
		return err
	}
	applyCUDAQ4KDecodeArm(arm)
	return nil
}

// CUDAQ4KDecodeArmInUse reports the arm the P < 4 Q4_K GEMV runs.
func CUDAQ4KDecodeArmInUse() CUDAQ4KDecodeArm {
	cudaQ4KPrefillArmMu.Lock()
	defer cudaQ4KPrefillArmMu.Unlock()
	return cudaQ4KDecodeArm
}

// q4kMMVQLaunchesForTest reports mmvq-arm GEMV launches, so the parity witness can refuse a run
// where the f32 arm produced the output.
func (c *cudaBackend) q4kMMVQLaunchesForTest() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_q4k_mmvq_launch_count())
}
