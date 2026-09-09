//go:build !(darwin && arm64 && cgo)

package metalgemm

import "errors"

// Q8Weight is an inert handle in stub builds.
type Q8Weight struct {
	Out, In, Nblk int
}

// Q6KWeight is an inert handle in stub builds.
type Q6KWeight struct {
	Out, In int
}

// EnsureFusedSwiGLUPipeline is a stub on non-darwin/non-arm64/non-cgo builds.
func EnsureFusedSwiGLUPipeline() error {
	return errors.New("metalgemm: fused swiglu not supported on this platform")
}

// FusedDequantGEMVSwiGLU is a stub on non-darwin/non-arm64/non-cgo builds.
func FusedDequantGEMVSwiGLU(gate, up *Q4KWeight, x, inter []float32) bool {
	return false
}

// FusedDequantGEMVSwiGLUQ8 is a stub on non-darwin/non-arm64/non-cgo builds.
func FusedDequantGEMVSwiGLUQ8(gate, up *Q8Weight, x, xd []float32, inter []float32) bool {
	return false
}

// FusedMLPQ6DownFast is a stub on non-darwin/non-arm64/non-cgo builds.
func FusedMLPQ6DownFast(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32) bool {
	return false
}

// FusedMLPFast is an alias to FusedMLPQ6DownFast on non-darwin/non-arm64/non-cgo builds.
func FusedMLPFast(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32) bool {
	return false
}

// FusedSwiGLUBandwidthSavingsBytes returns the eliminated DRAM bytes for the intermediate dim.
func FusedSwiGLUBandwidthSavingsBytes(intermediateDim int) int {
	return 2 * intermediateDim * 4
}
