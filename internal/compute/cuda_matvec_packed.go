//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "unsafe"

// MatVecPacked holds cudaMu once for the whole batch: one blocking H2D of the
// packed inputs, each GEMV into its own transient followed by a stream-ordered
// D2D into the packed output, then one blocking D2H. Per-GEMV Upload/Read
// pairs drain the stream twice per GEMV.
func (c *cudaBackend) MatVecPacked(ws []Tensor, x []float32, xOff, yOff []int, yLen int) ([]float32, error) {
	if err := CheckPackedMatVec(ws, x, xOff, yOff, yLen); err != nil {
		return nil, err
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	xt := c.uploadClassLocked(NewF32(c, []int{len(x)}, x), F32, MemoryActivation, "packed-matvec-x")
	defer c.cudaFreeLocked(xt)
	yt, yb := c.devTr([]int{yLen}, F32)
	defer c.cudaFreeLocked(yt)
	xb := xt.buf.(*cudaBuf)
	for i, w := range ws {
		in := w.Shape[1]
		xv := makeTensor(c, F32, RowMajor, []int{in}, nil, &cudaBuf{
			ptr: unsafe.Add(xb.ptr, xOff[i]*4), n: in * 4, class: xb.class, device: xb.device, view: true})
		y := c.cudaDecodeGEMVLocked(w, xv)
		out := w.Shape[0]
		C.fcuda_d2d(unsafe.Add(yb.ptr, yOff[i]*4), c.cudaBufForSubmit(y).ptr, C.size_t(out*4))
		c.cudaFreeLocked(y)
	}
	return c.readLocked(yt), nil
}
