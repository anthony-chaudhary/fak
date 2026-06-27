//go:build !arm64

package model

import "os"

// quant_noasm_q4k.go — the resident-Q4_K decode dispatch for archs without an SDOT Q4_K kernel
// (everything but arm64). There is no SIMD Q4_K kernel here yet, BUT the SCALAR int8 reduction
// (q4kReduceRowScalar) still beats the f32 path: it skips the 256-f32 per-super-block dequant and
// does a compact integer multiply-add (the same win the Q5_K int8 path measured at 2.13x on a
// GLM-5.2-shaped expert, sm_80). So q4kSDOTEnabled is an opt-in here via FAK_KQ_INT8 — the GLM-5.2
// mixed-quant offloaded-expert lever (its Q4_K experts otherwise run the slow f32 dequant on amd64).
// Default OFF (the path is approximate from activation quantization): the f32 GEMV stays the default
// until a real-weights witness clears it. An AVX512 Q4_K reducer is the next phase.

var q4kInt8Default = func() bool {
	switch os.Getenv("FAK_KQ_INT8") {
	case "1", "on", "true":
		return true
	}
	return false
}()

func q4kSDOTEnabled() bool {
	if q4kSDOTForce != 0 {
		return q4kSDOTForce > 0
	}
	return q4kInt8Default
}

func q4kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q4kReduceRowScalar(row, nblk, qx, IS, SS)
}
