package ggufload

import (
	"fmt"
	"strings"
)

// ServeKVDType is the resident global KV-cache dtype selected for model serving.
type ServeKVDType uint8

const (
	ServeKVF32 ServeKVDType = iota
	ServeKVF16
	ServeKVFP8
	ServeKVNVFP4
	ServeKVAsym
)

// String returns the canonical KV dtype selector token for d.
func (d ServeKVDType) String() string {
	switch d {
	case ServeKVF16:
		return "f16"
	case ServeKVFP8:
		return "fp8"
	case ServeKVNVFP4:
		return "nvfp4"
	case ServeKVAsym:
		return "fp8k_nvfp4v"
	default:
		return "f32"
	}
}

// ParseServeKVDType parses a KV dtype selector string (empty defaults to f32).
func ParseServeKVDType(raw string) (ServeKVDType, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "fp8k_nvfp4v", "k8v4":
		return ServeKVAsym, nil
	case "nvfp4", "fp4", "e2m1":
		return ServeKVNVFP4, nil
	case "fp8", "e4m3", "e4m3fn":
		return ServeKVFP8, nil
	case "f16", "fp16":
		return ServeKVF16, nil
	case "", "f32", "fp32":
		return ServeKVF32, nil
	default:
		return ServeKVF32, fmt.Errorf("serve: KV dtype %q is not f32, f16, fp8, nvfp4, or fp8k_nvfp4v", raw)
	}
}

// ParseServeKVDTypeToken parses a non-empty KV dtype selector token.
func ParseServeKVDTypeToken(raw string) (ServeKVDType, bool) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ServeKVF32, false
	}
	dt, err := ParseServeKVDType(trimmed)
	if err != nil {
		return ServeKVF32, false
	}
	return dt, true
}
