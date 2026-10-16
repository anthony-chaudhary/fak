package compute

// v41IndexerOwnedReader is sealed to compute. The original backend is supplied
// so a promoted method cannot grant ownership to a wrapper that overrides Read.
type v41IndexerOwnedReader interface {
	v41ReadIndexerOwned(Backend, Tensor) ([]float32, bool)
}

// ReadV41IndexerScores reads once and validates the expected row count before
// copying. A valid result is independently owned host storage and survives Free.
// Arbitrary backends retain their ordinary Read invocation and defensive copy.
// A false result means a malformed row count; transport panics are unchanged.
func ReadV41IndexerScores(backend Backend, tensor Tensor, expectedRows int) ([]float32, bool) {
	var values []float32
	independent := false
	if reader, ok := backend.(v41IndexerOwnedReader); ok {
		values, independent = reader.v41ReadIndexerOwned(backend, tensor)
	} else {
		values = backend.Read(tensor)
	}
	if len(values) != expectedRows {
		return nil, false
	}
	if !independent {
		values = append([]float32(nil), values...)
	}
	return values, true
}
