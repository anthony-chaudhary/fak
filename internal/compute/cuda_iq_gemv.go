//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "strconv"

// cudaIQ2GEMVMaxTokens is the CUDA grid.y ceiling the warp-per-row IQ2 GEMV launches over
// (blockIdx.y = token). Larger prefill batches need a token-tiled kernel, not yet ported.
const cudaIQ2GEMVMaxTokens = 65535

// cudaIQ2GEMV launches the resident IQ2_XXS / IQ2_XS GEMV (cuda/iq_gemv.cuh) for P tokens.
// Caller holds cudaMu and has validated out/in against the weight shape (in % 256 == 0 is a
// NewRawIQ invariant).
func cudaIQ2GEMV(dt Dtype, wb *cudaBuf, x, y *C.float, out, in, P int) {
	if P <= 0 || P > cudaIQ2GEMVMaxTokens || in%rawIQSuper != 0 {
		panic(&CUDAOpError{
			Op:    "MatMul",
			Site:  "iq2-gemv-geometry",
			Msg:   "cuda " + dt.String() + " GEMV needs 1 <= P <= " + strconv.Itoa(cudaIQ2GEMVMaxTokens) + " and in % 256 == 0 (got P=" + strconv.Itoa(P) + ", in=" + strconv.Itoa(in) + ")",
			Class: MemoryActivation,
		})
	}
	w := (*C.uint8_t)(wb.ptr)
	switch dt {
	case IQ2_XXS:
		C.fcuda_iq2xxs_gemv(w, x, y, C.int(out), C.int(in), C.int(P))
	case IQ2_XS:
		C.fcuda_iq2xs_gemv(w, x, y, C.int(out), C.int(in), C.int(P))
	default:
		panic(&CUDAOpError{Op: "MatMul", Site: "iq2-gemv-dtype", Msg: "cudaIQ2GEMV: unsupported dtype " + dt.String(), Class: MemoryActivation})
	}
}
