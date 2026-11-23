//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

// cudaDecodeGEMVLocked is the GEMV the batched decode entry points (MatVecPacked,
// SituMoE) launch. NVFP4 runs the warp-per-row kernel: the generic MatMul keeps
// the one-thread-per-output serial-K kernel that is bit-exact to the CPU W4A16
// oracle, but that kernel launches out/256 blocks, about 9 for a 2304-row expert,
// and dominated decode. Every other dtype takes the MatMul dispatch. Caller holds cudaMu.
func (c *cudaBackend) cudaDecodeGEMVLocked(w, x Tensor) Tensor {
	if w.Dtype != NVFP4 {
		return c.cudaMatMulLocked(w, x)
	}
	out, in, global, scales := c.cudaNVFP4Operands(w, x, 1, false)
	wb := c.cudaBufForSubmit(w)
	y, _ := c.devTr([]int{out}, F32)
	cudaGEMMStatus("DecodeGEMV", int(C.fcuda_nvfp4_w4a16_warp_f32((*C.uint8_t)(wb.ptr), (*C.uint8_t)(scales),
		c.cf(x), c.cf(y), C.int(out), C.int(in), 1, C.float(global))), cudaGEMMScratch{})
	return y
}
