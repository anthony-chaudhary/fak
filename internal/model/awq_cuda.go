//go:build cuda && awqcuda

// awq_cuda.go — CUDA-accelerated AWQ matmul. Compiled ONLY under `-tags "cuda awqcuda"`.
// This module provides GPU acceleration for AWQ 4-bit quantized weights when the
// CUDA backend is available.
//
// NOTE: this file uses cgo (import "C"). Go forbids a package that uses cgo from also
// containing Go (Plan 9) assembly files, and internal/model ships SIMD .s kernels
// (fdot/quant/saxpy). So a plain `-tags cuda` build of any binary that imports
// internal/model (e.g. cmd/fak, for `fak serve --backend cuda`) would fail to compile
// with this file in. It is therefore gated behind an EXTRA `awqcuda` tag: the GPU
// serving path (the compute.Backend HAL, Q8/F32) needs none of it and runs under a
// plain `-tags cuda` build; AWQ-on-GPU stays opt-in via `-tags "cuda awqcuda"`.

package model

import (
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../compute
#cgo LDFLAGS: -L${SRCDIR}/../compute -lfakcuda -lcudart -lcublas -lstdc++ -lm
#include <stdlib.h>
#include "../compute/cuda_backend.h"
*/
import "C"

// awqMatRowsCUDA computes y = AWQ @ x on GPU where AWQ is a 4-bit quantized tensor.
// This is a direct cgo call to the CUDA kernel for AWQ matrix-vector multiplication.
func awqMatRowsCUDA(qt *awqTensor, x []float32) []float32 {
	out, in := qt.out, qt.in
	y := make([]float32, out)

	if len(x) == 0 || out == 0 {
		return y
	}

	// Allocate device memory
	dW := C.fcuda_malloc(C.size_t(len(qt.raw)))
	dScales := C.fcuda_malloc(C.size_t(len(qt.scales) * 4))
	dX := C.fcuda_malloc(C.size_t(len(x) * 4))
	dY := C.fcuda_malloc(C.size_t(len(y) * 4))

	// Copy data to device
	C.fcuda_h2d(dW, unsafe.Pointer(&qt.raw[0]), C.size_t(len(qt.raw)))
	C.fcuda_h2d(dScales, unsafe.Pointer(&qt.scales[0]), C.size_t(len(qt.scales)*4))
	C.fcuda_h2d(dX, unsafe.Pointer(&x[0]), C.size_t(len(x)*4))

	// Run AWQ GEMV kernel
	C.fcuda_awq_gemv((*C.uint8_t)(dW), (*C.float)(dScales), (*C.float)(dX), (*C.float)(dY), C.int(out), C.int(in))

	// Copy result back
	C.fcuda_d2h(unsafe.Pointer(&y[0]), dY, C.size_t(len(y)*4))

	// Free device memory
	C.fcuda_free(dW)
	C.fcuda_free(dScales)
	C.fcuda_free(dX)
	C.fcuda_free(dY)

	return y
}

// awqGemmCUDA computes Y = X @ AWQ^T on GPU for P tokens.
func awqGemmCUDA(qt *awqTensor, X []float32, P int) []float32 {
	out, in := qt.out, qt.in
	Y := make([]float32, P*out)

	if len(X) == 0 || out == 0 || P == 0 {
		return Y
	}

	// Allocate device memory
	dW := C.fcuda_malloc(C.size_t(len(qt.raw)))
	dScales := C.fcuda_malloc(C.size_t(len(qt.scales) * 4))
	dX := C.fcuda_malloc(C.size_t(len(X) * 4))
	dY := C.fcuda_malloc(C.size_t(len(Y) * 4))

	// Copy data to device
	C.fcuda_h2d(dW, unsafe.Pointer(&qt.raw[0]), C.size_t(len(qt.raw)))
	C.fcuda_h2d(dScales, unsafe.Pointer(&qt.scales[0]), C.size_t(len(qt.scales)*4))
	C.fcuda_h2d(dX, unsafe.Pointer(&X[0]), C.size_t(len(X)*4))

	// Run AWQ GEMM kernel
	C.fcuda_awq_gemm((*C.uint8_t)(dW), (*C.float)(dScales), (*C.float)(dX), (*C.float)(dY), C.int(out), C.int(in), C.int(P))

	// Copy result back
	C.fcuda_d2h(unsafe.Pointer(&Y[0]), dY, C.size_t(len(Y)*4))

	// Free device memory
	C.fcuda_free(dW)
	C.fcuda_free(dScales)
	C.fcuda_free(dX)
	C.fcuda_free(dY)

	return Y
}
