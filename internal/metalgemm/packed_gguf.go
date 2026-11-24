package metalgemm

import "errors"

var (
	ErrPackedGGUFClosed = errors.New("metalgemm: packed GGUF store closed")
	ErrPackedGGUFShape  = errors.New("metalgemm: invalid packed GGUF matrix or job")
	ErrPackedGGUFType   = errors.New("metalgemm: unsupported packed GGUF type")
)

// PackedGGUFJob computes one matrix-vector product for one expert. Dense
// matrices have a single expert. All jobs in one Run are independent.
type PackedGGUFJob struct {
	Matrix *PackedGGUFMatrix
	Expert int
	X, Dst []float32
}

// PackedExpertMLP evaluates the selected experts with one shared input row.
// Optional per-expert scales are indexed by checkpoint expert ID.
type PackedExpertMLP struct {
	GateUp, Down           *PackedGGUFMatrix
	Selected               []int
	Weights                []float32
	GateUpScale, DownScale []float32
	Beta, LinearBeta       float32
}

// PackedRoutedFFN runs the router, expert selection and scaled MLP on one
// device command. The bias ranks experts but does not scale the mixture.
type PackedRoutedFFN struct {
	Router, GateUp, Down          *PackedGGUFMatrix
	Bias, GateUpScale, DownScale  []float32
	Slots                         int
	WeightScale, Beta, LinearBeta float32
}
