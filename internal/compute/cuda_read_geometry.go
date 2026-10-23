package compute

import "math"

// requireCUDAReadGeometry bounds the F32 host copy by the resident allocation.
// Read does not dequantize or widen narrow weights: interpreting their storage as
// F32 would copy beyond the allocation. The caller validates ownership/liveness.
func requireCUDAReadGeometry(t Tensor, storageBytes int, class MemoryClass) (elements, nbytes int) {
	refuse := func(reason string) {
		panic(&CUDAOpError{Op: "Read", Site: "read-geometry", Msg: reason, Class: class})
	}
	if t.Dtype != F32 {
		refuse("device Read supports only F32 tensors")
	}
	if storageBytes < 0 {
		refuse("resident allocation size is negative")
	}
	elements = 1
	for _, d := range t.Shape {
		if d < 0 || (d != 0 && elements > math.MaxInt/d) {
			refuse("tensor shape is negative or overflows host allocation capacity")
		}
		elements *= d
	}
	if elements > math.MaxInt/F32.Bytes() {
		refuse("F32 byte size overflows host allocation capacity")
	}
	nbytes = elements * F32.Bytes()
	if nbytes > storageBytes {
		refuse("tensor shape exceeds resident allocation")
	}
	return elements, nbytes
}
