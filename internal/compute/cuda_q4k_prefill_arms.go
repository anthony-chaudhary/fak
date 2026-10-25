package compute

import (
	"errors"
	"fmt"
)

// CUDAQ4KPrefillArm names the kernel the CUDA Q4_K GEMM runs for a prompt panel of 128 or more
// tokens (P-04, fak#13983). Decode (P < 4) and small panels keep their own kernels.
type CUDAQ4KPrefillArm string

const (
	// CUDAQ4KPrefillCompensated dequantizes the weight to an f32 transient and runs three
	// compensated FP16 cuBLAS GEMMs. It is the default.
	CUDAQ4KPrefillCompensated CUDAQ4KPrefillArm = "compensated"
	// CUDAQ4KPrefillMMQ is llama.cpp's int8 tensor-core Q4_K MMQ; it never expands the weight.
	CUDAQ4KPrefillMMQ CUDAQ4KPrefillArm = "mmq"
	// CUDAQ4KPrefillFP8LT dequantizes to an FP8 transient and runs one cuBLASLt FP8 GEMM.
	CUDAQ4KPrefillFP8LT CUDAQ4KPrefillArm = "fp8lt"
)

// CUDAQ4KPrefillEnv selects the arm once at CUDA init; SetCUDAQ4KPrefillArm changes it later.
const CUDAQ4KPrefillEnv = "FAK_CUDA_Q4K_PREFILL"

var (
	ErrCUDAQ4KPrefillArmUnknown       = errors.New("unknown CUDA Q4_K prefill arm")
	ErrCUDAQ4KPrefillArmUnimplemented = errors.New("CUDA Q4_K prefill arm is not implemented")
	ErrCUDAQ4KPrefillArmArch          = errors.New("CUDA Q4_K prefill arm needs a newer GPU arch")
)

// CUDAQ4KPrefillArmError is the typed refusal of an arm. Err is one of the sentinels above. SM is
// the device's compute capability (89 for sm_89), 0 when no device arch was known.
type CUDAQ4KPrefillArmError struct {
	Arm string
	SM  int
	Err error
}

func (e *CUDAQ4KPrefillArmError) Error() string {
	if e.SM > 0 {
		return fmt.Sprintf("%s=%q on sm_%d: %v", CUDAQ4KPrefillEnv, e.Arm, e.SM, e.Err)
	}
	return fmt.Sprintf("%s=%q: %v", CUDAQ4KPrefillEnv, e.Arm, e.Err)
}

func (e *CUDAQ4KPrefillArmError) Unwrap() error { return e.Err }

// cudaQ4KPrefillArmSpec is one arm's C code (fcuda_set_q4k_prefill_arm), its lowest compute
// capability, and whether the CUDA backend can run it yet.
type cudaQ4KPrefillArmSpec struct {
	code        int
	minSM       int
	implemented bool
}

var cudaQ4KPrefillArms = map[CUDAQ4KPrefillArm]cudaQ4KPrefillArmSpec{
	CUDAQ4KPrefillCompensated: {code: 0, implemented: true},
	CUDAQ4KPrefillMMQ:         {code: 1},
	CUDAQ4KPrefillFP8LT:       {code: 2, minSM: 89}, // FP8 tensor cores start at Ada
}

// cudaQ4KPrefillArmRefusedCode is the C arm code that makes every P >= 128 Q4_K GEMM return
// FCUDA_Q4K_PREFILL_ARM_REFUSED, so an invalid selection never falls back silently.
const cudaQ4KPrefillArmRefusedCode = -1

// ParseCUDAQ4KPrefillArm reads a FAK_CUDA_Q4K_PREFILL value. Empty selects the default.
func ParseCUDAQ4KPrefillArm(s string) (CUDAQ4KPrefillArm, error) {
	if s == "" {
		return CUDAQ4KPrefillCompensated, nil
	}
	arm := CUDAQ4KPrefillArm(s)
	if _, ok := cudaQ4KPrefillArms[arm]; !ok {
		return "", &CUDAQ4KPrefillArmError{Arm: s, Err: ErrCUDAQ4KPrefillArmUnknown}
	}
	return arm, nil
}

// CheckCUDAQ4KPrefillArm refuses an arm the device at compute capability sm cannot run. sm <= 0
// skips the arch floor (no device arch known).
func CheckCUDAQ4KPrefillArm(arm CUDAQ4KPrefillArm, sm int) error {
	spec, ok := cudaQ4KPrefillArms[arm]
	switch {
	case !ok:
		return &CUDAQ4KPrefillArmError{Arm: string(arm), SM: sm, Err: ErrCUDAQ4KPrefillArmUnknown}
	case sm > 0 && sm < spec.minSM:
		return &CUDAQ4KPrefillArmError{Arm: string(arm), SM: sm, Err: ErrCUDAQ4KPrefillArmArch}
	case !spec.implemented:
		return &CUDAQ4KPrefillArmError{Arm: string(arm), SM: sm, Err: ErrCUDAQ4KPrefillArmUnimplemented}
	}
	return nil
}

// resolveCUDAQ4KPrefillArm parses and checks an env value in one step.
func resolveCUDAQ4KPrefillArm(s string, sm int) (CUDAQ4KPrefillArm, error) {
	arm, err := ParseCUDAQ4KPrefillArm(s)
	if err != nil {
		return "", err
	}
	if err := CheckCUDAQ4KPrefillArm(arm, sm); err != nil {
		return "", err
	}
	return arm, nil
}
