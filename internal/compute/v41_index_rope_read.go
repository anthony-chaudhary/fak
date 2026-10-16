package compute

import "math"

// ReadV41IndexRoPEOutput reads one index-query rotation row and validates it before
// returning independent host storage. The untouched prefix retains exact input
// bits. Canonical device reads transfer their fresh slice; all other reads copy.
// Input geometry is already validated by the index-query RoPE operation. No callback
// or transport failure is retried, and no malformed row escapes.
func ReadV41IndexRoPEOutput(backend Backend, tensor Tensor, input []float32, headDim, rotaryDim int) ([]float32, bool) {
	if headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 || len(input)%headDim != 0 {
		return nil, false
	}
	var values []float32
	independent := false
	// The sealed reader has no index-specific behavior: it only establishes
	// canonical Read storage ownership while preserving wrapper Read overrides.
	if reader, ok := backend.(v41IndexerOwnedReader); ok {
		values, independent = reader.v41ReadIndexerOwned(backend, tensor)
	} else {
		values = backend.Read(tensor)
	}
	if len(values) != len(input) {
		return nil, false
	}
	for i, value := range values {
		if value != value || value > math.MaxFloat32 || value < -math.MaxFloat32 ||
			(i%headDim < headDim-rotaryDim && math.Float32bits(value) != math.Float32bits(input[i])) ||
			(i%headDim >= headDim-rotaryDim && v41IndexRoPEBF16Overflows(value)) {
			return nil, false
		}
	}
	if !independent {
		values = append([]float32(nil), values...)
	}
	return values, true
}

// Match the finite F32 threshold whose BF16 round-to-nearest-even is infinite.
func v41IndexRoPEBF16Overflows(value float32) bool {
	magnitude := math.Float32bits(value) & 0x7fffffff
	return magnitude >= 0x7f7f8000 && magnitude < 0x7f800000
}
