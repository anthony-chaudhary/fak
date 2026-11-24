package compute

import (
	"errors"
	"fmt"
)

// CUDAQ6KPrefillArm names the kernel a CUDA Q6_K BatchedMatMul of cudaQ6KMMQMinCols or more
// prompt rows runs. Decode (P < 4) and narrower panels keep their own kernels on every arm.
type CUDAQ6KPrefillArm string

const (
	// CUDAQ6KPrefillDequant is the legacy k_q6k_gemm panel: the weight is dequantized in the
	// tile and dotted in F32, re-reading every weight byte once per four prompt rows.
	CUDAQ6KPrefillDequant CUDAQ6KPrefillArm = "dequant"
	// CUDAQ6KPrefillMMQ is the int8 tensor-core Q6_K MMQ (cuda/q56k_mmq.cuh): the weight is
	// streamed once per panel tile and the activation is quantized to q8_1. A pre-enqueue
	// refusal (capture, operand ownership, device) declines to the dequant panel.
	CUDAQ6KPrefillMMQ CUDAQ6KPrefillArm = "mmq"
)

// CUDAQ6KPrefillEnv selects the arm once at CUDA init; "dequant" restores the legacy panel.
const CUDAQ6KPrefillEnv = "FAK_CUDA_Q6K_PREFILL"

// cudaQ6KMMQMinCols mirrors FCUDA_Q6K_PREFILL_MMQ_MIN_COLS: the narrowest MMQ column tile.
const cudaQ6KMMQMinCols = 8

// cudaQ6KPrefillMMQDeclined is FCUDA_Q6K_PREFILL_MMQ_DECLINED: nothing was enqueued.
const cudaQ6KPrefillMMQDeclined = 80011

// cudaQ6KMMQMinSM is the int8 mma.sync floor the MMQ family compiles real kernels for.
const cudaQ6KMMQMinSM = 80

var (
	ErrCUDAQ6KPrefillArmUnknown = errors.New("unknown CUDA Q6_K prefill arm")
	ErrCUDAQ6KPrefillArmArch    = errors.New("CUDA Q6_K prefill arm is not supported on this GPU arch")
)

var cudaQ6KPrefillArmCodes = map[CUDAQ6KPrefillArm]int{
	CUDAQ6KPrefillDequant: 0,
	CUDAQ6KPrefillMMQ:     1,
}

// DefaultCUDAQ6KPrefillArm is the arm an unset FAK_CUDA_Q6K_PREFILL selects on a device of
// compute capability sm: mmq from sm_80 up, dequant below it or when the arch is unknown.
func DefaultCUDAQ6KPrefillArm(sm int) CUDAQ6KPrefillArm {
	if sm >= cudaQ6KMMQMinSM {
		return CUDAQ6KPrefillMMQ
	}
	return CUDAQ6KPrefillDequant
}

// CheckCUDAQ6KPrefillArm refuses an unknown arm, or mmq on a known arch below sm_80.
func CheckCUDAQ6KPrefillArm(arm CUDAQ6KPrefillArm, sm int) error {
	if _, ok := cudaQ6KPrefillArmCodes[arm]; !ok {
		return fmt.Errorf("%s=%q: %w", CUDAQ6KPrefillEnv, string(arm), ErrCUDAQ6KPrefillArmUnknown)
	}
	if arm == CUDAQ6KPrefillMMQ && sm > 0 && sm < cudaQ6KMMQMinSM {
		return fmt.Errorf("%s=%q on sm_%d: %w", CUDAQ6KPrefillEnv, string(arm), sm, ErrCUDAQ6KPrefillArmArch)
	}
	return nil
}

// resolveCUDAQ6KPrefillArm always returns a runnable arm: the legacy panel has no refusal a
// whole serve could survive, so a bad value reports its error and keeps dequant.
func resolveCUDAQ6KPrefillArm(s string, sm int) (CUDAQ6KPrefillArm, error) {
	if s == "" {
		return DefaultCUDAQ6KPrefillArm(sm), nil
	}
	arm := CUDAQ6KPrefillArm(s)
	if err := CheckCUDAQ6KPrefillArm(arm, sm); err != nil {
		return CUDAQ6KPrefillDequant, err
	}
	return arm, nil
}

// cudaQ6KPrefillMMQEligible reports whether a BatchedMatMul offers the panel to the default
// MMQ arm. The P>=128 qualification route keeps its strict admission and wins when selected.
func cudaQ6KPrefillMMQEligible(arm CUDAQ6KPrefillArm, dt Dtype, p int, qualificationSelected bool) bool {
	return arm == CUDAQ6KPrefillMMQ && dt == Q6_K && p >= cudaQ6KMMQMinCols && !qualificationSelected
}

// cudaQ6KPrefillMMQScratchMiss mirrors FCUDA_SCRATCH_ALLOC_FAILED. The native arm declines a
// plain miss to the legacy panel; one that still surfaces (a sticky native terminal emptied the
// pool) is an allocation error for the request, never an execution fault that latches the device.
const cudaQ6KPrefillMMQScratchMiss = 80001

// cudaQ6KPrefillMMQScratchBytes is the q8_1 activation the default Q6_K MMQ arm allocates for
// rows prompt rows: in padded to 512 values, 144 bytes per 128 values, plus one 128-column tile.
// The Stream-K fixup depends on the SM count, so a miss there reports these bytes.
func cudaQ6KPrefillMMQScratchBytes(in, rows int) int {
	const q81Block, q81Values = 144, 128 // block_q8_1_mmq
	padded := (in + 511) / 512 * 512
	return rows*padded/q81Values*q81Block + 128*q81Block
}
