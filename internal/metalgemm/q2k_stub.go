//go:build !(darwin && arm64 && cgo)

// q2k_stub.go — stub implementation of Q2_K Metal operations for non-Metal builds.

package metalgemm

// Q2KWeight is an inert handle in non-Metal builds.
type Q2KWeight struct {
	Out, In int
	Nblk    int
}

// UploadQ2K is a no-op returning nil in non-Metal builds.
func UploadQ2K(raw []byte, out, in int) *Q2KWeight { return nil }

// ID returns an invalid handle in non-Metal builds.
func (w *Q2KWeight) ID() int { return -1 }

// GEMV is a no-op in non-Metal builds.
func (w *Q2KWeight) GEMV(x, y []float32) {}

// GEMM is a no-op in non-Metal builds.
func (w *Q2KWeight) GEMM(X []float32, P int, Y []float32) {}

// ResetQ2K is a no-op in non-Metal builds.
func ResetQ2K() {}
