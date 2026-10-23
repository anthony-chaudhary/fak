package compute

import "math"

// requireCUDAMatMulGeometry runs before allocation, tracing, or narrowing to the
// CUDA int ABI. It checks logical extents only; residency and buffer validity
// remain the responsibility of the existing submit path.
func requireCUDAMatMulGeometry(op string, w, x Tensor, rows int) (out, in int) {
	refuse := func(reason string) {
		panic(&CUDAOpError{Op: op, Site: "matmul-geometry", Msg: reason, Class: MemoryActivation})
	}
	if len(w.Shape) != 2 {
		refuse("weight must have shape [out,in]")
	}
	out, in = w.Shape[0], w.Shape[1]
	if out <= 0 || in <= 0 || rows <= 0 || out > math.MaxInt32 || in > math.MaxInt32 || rows > math.MaxInt32 {
		refuse("matrix dimensions must be positive and fit the CUDA int ABI")
	}
	// devTr allocates F32 outputs with host-int arithmetic. Input and dense
	// weight byte counts must also be representable before considering Numel.
	if rows > math.MaxInt/F32.Bytes()/out || rows > math.MaxInt/F32.Bytes()/in {
		refuse("activation or output byte size overflows host allocation capacity")
	}
	if (w.Dtype == F32 || w.Dtype == F16) && out > math.MaxInt/w.Dtype.Bytes()/in {
		refuse("dense weight byte size overflows host allocation capacity")
	}
	want := rows * in
	if x.Dtype != F32 {
		refuse("activation must be F32")
	}
	// Tensor.Numel is unchecked. Bound each multiplication by the expected
	// extent so a malformed shape cannot wrap back to the required input size.
	n := 1
	for _, d := range x.Shape {
		if d <= 0 || d > want/n {
			refuse("activation shape does not match rows*in")
		}
		n *= d
	}
	if n != want {
		refuse("activation shape does not match rows*in")
	}
	// fcuda_matmul_f16 converts the activation using both P*in and
	// (P*in+255)/256 as signed int expressions before invoking cuBLAS.
	if w.Dtype == F16 && want > math.MaxInt32-255 {
		refuse("F16 activation conversion exceeds CUDA signed-int launch indexing")
	}
	return out, in
}
