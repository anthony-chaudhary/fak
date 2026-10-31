//go:build cuda && cgo && cudaarchspecific && fak_cuda_flashmla_decode

package compute

/*
#include "cuda_flashmla_sparse_decode_owner.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Package-local acceptance entry only: no Supports* method, model dispatch,
// automatic activation, or host retry. The linked symbol requires matching
// native registration and the bridge checks the actual current device.
func (c *cudaBackend) qualifyFlashMLADecode(x flashMLADecodeInput) (flashMLADecodeOutput, error) {
	var zero flashMLADecodeOutput
	if err := validateFlashMLADecode(x); err != nil {
		return zero, err
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c == nil || c != cudaDev || c.capturing {
		return zero, &CUDAOpError{Op: "FlashMLADecode", Site: "owner/capture"}
	}
	if err := c.faultLatch.Admit("flashmla-decode"); err != nil {
		return zero, err
	}
	cudaV41CheckedSubmit()
	y := flashMLADecodeOutput{ValuesBF16: make([]uint16, 32768), LSE: make([]float32, 64)}
	ptr := func(b []byte) *C.uint8_t {
		if len(b) == 0 {
			return nil
		}
		return (*C.uint8_t)(unsafe.Pointer(&b[0]))
	}
	ids := func(b []int32) *C.int32_t {
		if len(b) == 0 {
			return nil
		}
		return (*C.int32_t)(unsafe.Pointer(&b[0]))
	}
	var sink *C.float
	if len(x.Sink) != 0 {
		sink = (*C.float)(unsafe.Pointer(&x.Sink[0]))
	}
	var extraLength *C.int32_t
	if len(x.ExtraIndices) != 0 {
		extraLength = (*C.int32_t)(unsafe.Pointer(&x.ExtraLength))
	}
	invariant := C.int(0)
	if x.BatchInvariant {
		invariant = 1
	}
	code := int(C.fak_flashmla_decode_host(
		(*C.uint16_t)(unsafe.Pointer(&x.Query[0])), ptr(x.KV), ids(x.Indices), (*C.int32_t)(unsafe.Pointer(&x.Length)), sink,
		ptr(x.ExtraKV), ids(x.ExtraIndices), extraLength,
		(*C.uint16_t)(unsafe.Pointer(&y.ValuesBF16[0])), (*C.float)(unsafe.Pointer(&y.LSE[0])),
		C.int(x.Blocks), C.int(x.PageSize), C.int(len(x.Indices)), C.int(x.ExtraBlocks), C.int(x.ExtraPageSize), C.int(len(x.ExtraIndices)), C.float(x.Scale), invariant))
	runtime.KeepAlive(x)
	if code != 0 {
		return zero, &CUDAOpError{Op: "FlashMLADecode", Site: "native", Code: code}
	}
	if err := validateFlashMLADecodeOutput(x, y); err != nil {
		return zero, err
	}
	return y, nil
}
