package compute

import "math"

// ReadV41MHCCollapse reads a retained mHC collapse panel into independent host
// storage. Generic reads allocate before the finite-validation/copy loop, as the
// model did; canonical device reads already own their storage and need no copy.
// A false result means a malformed or nonfinite panel. Transport panics propagate.
func ReadV41MHCCollapse(backend Backend, tensor Tensor, expectedCount int) ([]float32, bool) {
	var values []float32
	independent := false
	if reader, ok := backend.(v41IndexerOwnedReader); ok {
		values, independent = reader.v41ReadIndexerOwned(backend, tensor)
	} else {
		values = backend.Read(tensor)
	}
	if len(values) != expectedCount {
		return nil, false
	}
	flat := values
	if !independent {
		flat = make([]float32, len(values))
	}
	for i, value := range values {
		if value != value || value > math.MaxFloat32 || value < -math.MaxFloat32 {
			return nil, false
		}
		if !independent {
			flat[i] = value
		}
	}
	return flat, true
}
