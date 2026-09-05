//go:build !amd64

package model

// quant_q4k_f32_noasm.go — the non-amd64 build of q4kMatRowsRangeArch. There is no AVX2 f32 Q4_K
// kernel off amd64, so this reports "not handled" and q4kMatRowsInto runs the scalar q4kMatRowsRange.
// (arm64's Q4_K acceleration is the separate int8 SDOT path, gated by FAK_KQ_INT8, unaffected here.)
func q4kMatRowsRangeArch(qt *q4kTensor, x, y []float32, lo, hi int) bool                { return false }
func q4kMatRowsRangeArchRaw(raw []byte, qt *q4kTensor, x, y []float32, lo, hi int) bool { return false }
