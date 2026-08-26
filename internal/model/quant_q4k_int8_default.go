//go:build !arm64 || (fakaccel && darwin && cgo)

package model

import "os"

// q4kInt8Default resolves FAK_KQ_INT8 once. The approximate activation-
// quantized path stays opt-in until a real-weights witness clears it.
var q4kInt8Default = envEnabled("FAK_KQ_INT8")

func envEnabled(name string) bool {
	switch os.Getenv(name) {
	case "1", "on", "true":
		return true
	default:
		return false
	}
}

// q4kSDOTEnabled reports whether the resident-Q4_K approximate int8 decode
// path is active. Architecture-specific reducers choose their own SIMD tier.
func q4kSDOTEnabled() bool {
	if q4kSDOTForce != 0 {
		return q4kSDOTForce > 0
	}
	return q4kInt8Default
}
