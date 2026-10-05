//go:build !(darwin && arm64 && cgo)

package metalgemm

import "errors"

// PrefillAttentionSupported is always false without Metal.
func PrefillAttentionSupported(headDim, nH, nKV int) bool { return false }

// PrefillAttentionTiming is the execution receipt of one PrefillAttention call.
type PrefillAttentionTiming struct {
	GPUMilliseconds  float64
	WaitMilliseconds float64
}

// PrefillAttention is unavailable without Metal.
func PrefillAttention(out, q, k, v []float32, P, kvLen, nH, nKV, headDim, window int, scale float32) (PrefillAttentionTiming, error) {
	return PrefillAttentionTiming{}, errors.New("metalgemm: prefill attention requires darwin/arm64 Metal")
}
