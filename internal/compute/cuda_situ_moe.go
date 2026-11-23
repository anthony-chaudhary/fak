//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "unsafe"

// SituMoE holds cudaMu once: one blocking H2D of x and the per-pick scalars,
// the gate-up GEMVs, the situ activation, the down GEMVs and the weighted sum
// stay on g_stream, then one blocking D2H of the hidden-wide result.
func (c *cudaBackend) SituMoE(gateUp, down []Tensor, x []float32, weight, guScale, dnScale []float32, beta, linBeta float32) ([]float32, error) {
	if err := CheckSituMoE(gateUp, down, x, weight, guScale, dnScale); err != nil {
		return nil, err
	}
	align := func(n int) int { return (n + PackedMatVecAlign - 1) / PackedMatVecAlign * PackedMatVecAlign }
	picks, in := len(gateUp), gateUp[0].Shape[1]
	width, hidden, downIn := gateUp[0].Shape[0]/2, down[0].Shape[0], down[0].Shape[1]
	xSeg, pSeg := align(in), align(picks)
	host := make([]float32, xSeg+3*pSeg)
	copy(host, x)
	copy(host[xSeg:], guScale)
	copy(host[xSeg+pSeg:], dnScale)
	copy(host[xSeg+2*pSeg:], weight)
	guStride, actStride, dStride := align(2*width), align(downIn), align(hidden)

	cudaMu.Lock()
	defer cudaMu.Unlock()
	ht := c.uploadClassLocked(NewF32(c, []int{len(host)}, host), F32, MemoryActivation, "situ-moe-input")
	defer c.cudaFreeLocked(ht)
	hb := c.cudaBufForSubmit(ht)
	at := func(off int) unsafe.Pointer { return unsafe.Add(hb.ptr, off*4) }
	view := func(b *cudaBuf, off, n int) Tensor {
		return makeTensor(c, F32, RowMajor, []int{n}, nil, &cudaBuf{
			ptr: unsafe.Add(b.ptr, off*4), n: n * 4, class: b.class, device: b.device, view: true})
	}
	stage := func(ws []Tensor, src func(p int) Tensor, dst *cudaBuf, stride int) {
		for p, w := range ws {
			y := c.cudaDecodeGEMVLocked(w, src(p))
			C.fcuda_d2d(unsafe.Add(dst.ptr, p*stride*4), c.cudaBufForSubmit(y).ptr, C.size_t(w.Shape[0]*4))
			c.cudaFreeLocked(y)
		}
	}
	gut, gub := c.devTr([]int{picks * guStride}, F32)
	defer c.cudaFreeLocked(gut)
	stage(gateUp, func(int) Tensor { return view(hb, 0, in) }, gub, guStride)
	actt, actb := c.devTr([]int{picks * actStride}, F32)
	defer c.cudaFreeLocked(actt)
	C.fcuda_situ_act_f32((*C.float)(gub.ptr), C.int(guStride), (*C.float)(at(xSeg)), (*C.float)(actb.ptr),
		C.int(actStride), C.int(width), C.int(picks), C.double(beta), C.double(linBeta))
	dt, db := c.devTr([]int{picks * dStride}, F32)
	defer c.cudaFreeLocked(dt)
	stage(down, func(p int) Tensor { return view(actb, p*actStride, downIn) }, db, dStride)
	out, ob := c.devTr([]int{hidden}, F32)
	defer c.cudaFreeLocked(out)
	C.fcuda_situ_combine_f32((*C.float)(db.ptr), C.int(dStride), (*C.float)(at(xSeg+pSeg)), (*C.float)(at(xSeg+2*pSeg)),
		(*C.float)(ob.ptr), C.int(hidden), C.int(picks))
	return c.readLocked(out), nil
}
