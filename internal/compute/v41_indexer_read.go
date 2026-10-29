package compute

// v41IndexerOwnedReader is sealed to compute. The original backend is supplied
// so a promoted method cannot grant ownership to a wrapper that overrides Read.
type v41IndexerOwnedReader interface {
	v41ReadIndexerOwned(Backend, Tensor) ([]float32, bool)
}

// v41IndexerCheckedReader is a compute-only consumer of checked score output.
// Implementations must decline promoted wrapper methods using original identity.
// A handled result transfers fresh host storage; errors never permit a retry.
type v41IndexerCheckedReader interface {
	v41ReadIndexerChecked(original Backend, tensor Tensor, rows int) (values []float32, handled bool, err error)
}

// ReadV41IndexerScores reads once and validates the expected row count before
// copying. A valid result is independently owned host storage and survives Free.
// Arbitrary backends retain their ordinary Read invocation and defensive copy.
// A false result means a malformed row count; transport panics are unchanged.
func ReadV41IndexerScores(backend Backend, tensor Tensor, expectedRows int) ([]float32, bool) {
	if reader, ok := backend.(v41IndexerCheckedReader); ok {
		values, handled, err := reader.v41ReadIndexerChecked(backend, tensor, expectedRows)
		if err != nil {
			// Preserve the transport failure for the model's existing panic mapper.
			panic(err)
		}
		if handled {
			if len(values) != expectedRows {
				return nil, false
			}
			return values, true
		}
	}
	return readV41OwnedF32Row(backend, tensor, expectedRows)
}

// readV41OwnedF32Row only validates shape and storage ownership. Callers retain
// their operation-specific finite checks, publication order and error mapping.
func readV41OwnedF32Row(backend Backend, tensor Tensor, expectedRows int) ([]float32, bool) {
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
