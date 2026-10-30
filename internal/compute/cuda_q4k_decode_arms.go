package compute

import (
	"errors"
	"fmt"
)

// CUDAQ4KDecodeArm names the kernel the CUDA Q4_K GEMV runs for decode and panels of fewer than
// four tokens (fak#8635). Prompt panels keep their own arms (CUDAQ4KPrefillArm).
type CUDAQ4KDecodeArm string

const (
	// CUDAQ4KDecodeF32 dequantizes each weight in the GEMV tile and dots it with the f32
	// activation (k_q4k_gemm). It is the escape hatch (FAK_CUDA_Q4K_DECODE=f32).
	CUDAQ4KDecodeF32 CUDAQ4KDecodeArm = "f32"
	// CUDAQ4KDecodeMMVQ is llama.cpp's Q4_K MMVQ (cuda/q4k_mmvq.cuh): the activation is quantized
	// to q8_1 per 32 values and the dot runs on DP4A.
	CUDAQ4KDecodeMMVQ CUDAQ4KDecodeArm = "mmvq"
	// CUDAQ4KDecodeDefault is the arm an unset FAK_CUDA_Q4K_DECODE selects. mmvq is the default
	// on a same-card ABBA decode receipt with greedy-token parity against f32 (raider RTX 5090
	// Laptop, Qwen3.8-27B Q4_K_M; fak-private docs/benchmarks/raider-5090/cuda-decode-defaults-20261029).
	CUDAQ4KDecodeDefault = CUDAQ4KDecodeMMVQ
)

// CUDAQ4KDecodeEnv selects the arm once at CUDA init; SetCUDAQ4KDecodeArm changes it later.
const CUDAQ4KDecodeEnv = "FAK_CUDA_Q4K_DECODE"

// ErrCUDAQ4KDecodeArmUnknown is the refusal of a FAK_CUDA_Q4K_DECODE value no arm answers to.
var ErrCUDAQ4KDecodeArmUnknown = errors.New("unknown CUDA Q4_K decode arm")

// cudaQ4KDecodeArmCodes maps each arm to its fcuda_set_q4k_decode_arm code.
var cudaQ4KDecodeArmCodes = map[CUDAQ4KDecodeArm]int{CUDAQ4KDecodeF32: 0, CUDAQ4KDecodeMMVQ: 1}

// ParseCUDAQ4KDecodeArm reads a FAK_CUDA_Q4K_DECODE value. Empty selects CUDAQ4KDecodeDefault.
func ParseCUDAQ4KDecodeArm(s string) (CUDAQ4KDecodeArm, error) {
	if s == "" {
		return CUDAQ4KDecodeDefault, nil
	}
	arm := CUDAQ4KDecodeArm(s)
	if err := CheckCUDAQ4KDecodeArm(arm); err != nil {
		return "", err
	}
	return arm, nil
}

// CheckCUDAQ4KDecodeArm refuses an arm no decode kernel answers to, the empty arm included.
func CheckCUDAQ4KDecodeArm(arm CUDAQ4KDecodeArm) error {
	if _, ok := cudaQ4KDecodeArmCodes[arm]; !ok {
		return fmt.Errorf("%s=%q: %w", CUDAQ4KDecodeEnv, arm, ErrCUDAQ4KDecodeArmUnknown)
	}
	return nil
}
