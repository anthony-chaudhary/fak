package compute

import "math"

// requireCUDARMSNormGeometry validates flattened row geometry before allocation
// or narrowing to the CUDA int ABI. Empty shapes retain Tensor's scalar semantics.
func requireCUDARMSNormGeometry(x, weight Tensor) (rows, width int) {
	refuse := func(reason string) {
		panic(&CUDAOpError{Op: "RMSNorm", Site: "rmsnorm-geometry", Msg: reason, Class: MemoryActivation})
	}
	if x.Dtype != F32 || weight.Dtype != F32 {
		refuse("input and weight must be F32")
	}
	// Tensor.Numel is unchecked. Validate every dimension before multiplying so
	// negative dimensions and products that wrap to plausible sizes cannot pass.
	numel := func(shape []int, limit int, operand string) int {
		n := 1
		for _, d := range shape {
			if d <= 0 || d > limit/n {
				refuse(operand + " shape has a nonpositive dimension or exceeds its supported extent")
			}
			n *= d
		}
		return n
	}
	byteLimit := math.MaxInt / F32.Bytes()
	// k_rmsnorm advances a signed int index by 256 after its last element.
	// Reserve that increment as well as bounding the width passed to C.int.
	widthLimit := min(byteLimit, math.MaxInt32-255)
	width = numel(weight.Shape, widthLimit, "weight")
	elements := numel(x.Shape, byteLimit, "input")
	if elements%width != 0 {
		refuse("input element count must be divisible by weight element count")
	}
	rows = elements / width
	// One block per row uses grid.x, not the smaller grid.y/grid.z ceiling.
	if rows > math.MaxInt32 {
		refuse("row count exceeds the CUDA int ABI and grid.x limit")
	}
	return rows, width
}
