//go:build cuda && cgo && fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool && !fak_cuda_v41_compressor_pending

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI. The library must be built with FAK_CUDA_V41_COMPRESSOR.
int fak_v41_compressor_state(void);
int fak_v41_compressor_ring_store_f32(const float *, const float *, float *, float *, float *, int, int, int, int);
int fak_v41_compressor_pool_f32(const float *, const float *, float *, float *, int, int);
uint64_t fak_v41_compressor_completed(void);
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

var (
	_ V41CompressorRingBackend = (*cudaBackend)(nil)
	_ V41CompressorPoolBackend = (*cudaBackend)(nil)
)

// The CUDA compressor ports ds4's store and update-pool kernels
// (cuda/v41_compressor.cuh). Its pool is the ds4 numerical class (expf,
// accumulate kv*w then one quotient), pinned to a binary64 oracle within a few
// ulps rather than bit-compared with the host or Vulkan order. Ring emit and
// pool run the same device kernel, so on this backend a ring's emitted row is
// bit-identical to V41CompressorPool over the same group. Compiling the tag is
// the opt-in; the default cuda build lacks these methods and the model keeps
// its host-staged group. The separate unqualified pool-only candidate tag
// fak_cuda_v41_compressor_pool (cuda_v41_compressor_pool.go) is mutually
// exclusive with this one.
func (c *cudaBackend) SupportsV41CompressorRing() bool {
	return c.supportsV41Compressor("SupportsV41CompressorRing")
}

func (c *cudaBackend) SupportsV41CompressorPool() bool {
	return c.supportsV41Compressor("SupportsV41CompressorPool")
}

func (c *cudaBackend) supportsV41Compressor(site string) bool {
	if c == nil {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.cudaV41CompressorState(site) == nil
}

func cudaV41CompressorError(site, message string, status int, cause error) error {
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-v41-compressor-failed",
		Message: fmt.Sprintf("%s (status %d)", message, status), Err: cause, Recovered: cause}
}

// Caller holds cudaMu. Faults take precedence over native admission.
func (c *cudaBackend) cudaV41CompressorState(site string) error {
	if c == nil || c.capturing {
		return cudaV41CompressorError(site, "nil backend or active capture", -3, nil)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return cudaV41CompressorError(site, "sticky CUDA fault", -3, err)
	}
	if status := int(C.fak_v41_compressor_state()); status != 0 {
		return cudaV41CompressorError(site, "native admission refused", status, nil)
	}
	return nil
}

// cudaV41CompressorOperand admits a device-only F32 operand owned by c.
func (c *cudaBackend) cudaV41CompressorOperand(site, name string, t Tensor, elems int) (*cudaBuf, error) {
	b, ok := t.buf.(*cudaBuf)
	if t.be != c || !ok || b == nil || b.ptr == nil || b.n < elems*4 || b.managed ||
		b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return nil, cudaV41CompressorError(site, name+" is not owned device-only F32 storage", -2, ErrV41CompressorRingGeometry)
	}
	if invalid := b.invalidStateError(site + " " + name); invalid != nil {
		return nil, cudaV41CompressorError(site, "invalid storage lifetime", -2, invalid)
	}
	return b, nil
}

func (c *cudaBackend) cudaV41CompressorNativeError(site string, status int) error {
	var cause error
	if status >= 30000 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, site, status)
	} else if status == -1 || status == -2 {
		cause = ErrV41CompressorRingGeometry
	}
	return cudaV41CompressorError(site, "checked CUDA compressor refused", status, cause)
}

func (c *cudaBackend) V41CompressorRingAlloc(ratio, width int) (kv, scores Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var bufs [2]*cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, "V41CompressorRingAlloc", "compressor-ring")
			err = cudaV41CompressorError("V41CompressorRingAlloc", "candidate panicked", -1, cause)
		}
		if err != nil {
			for _, b := range bufs {
				if b != nil && b.ptr != nil {
					C.fcuda_free(b.ptr)
					b.ptr, b.n = nil, 0
				}
			}
			kv, scores = Tensor{}, Tensor{}
		}
	}()
	if verr := ValidateV41CompressorRingAlloc(ratio, width); verr != nil {
		return Tensor{}, Tensor{}, cudaV41CompressorError("V41CompressorRingAlloc", "invalid geometry", 2, verr)
	}
	if err = c.cudaV41CompressorState("V41CompressorRingAlloc"); err != nil {
		return Tensor{}, Tensor{}, err
	}
	for i, site := range []string{"v41-compressor-ring-kv", "v41-compressor-ring-scores"} {
		if bufs[i], err = c.dallocDeviceOnlyClass(ratio*width*4, MemoryKVCache, site); err != nil {
			return Tensor{}, Tensor{}, cudaV41CompressorError("V41CompressorRingAlloc", "device-only ring allocation failed", -2, err)
		}
	}
	return makeTensor(c, F32, RowMajor, []int{ratio, width}, nil, bufs[0]),
		makeTensor(c, F32, RowMajor, []int{ratio, width}, nil, bufs[1]), nil
}

// V41CompressorRingStore enqueues the row store; with emit the same call pools
// the ring and drains, so the returned [2,width] tensor is complete.
func (c *cudaBackend) V41CompressorRingStore(kv, scores, kvRow, scoreRow Tensor, slot, ratio, width int, emit bool) (out Tensor, err error) {
	const site = "V41CompressorRingStore"
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output *cudaBuf
	leak := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "compressor-ring")
			err = cudaV41CompressorError(site, "candidate panicked", -1, cause)
		}
		if err != nil {
			if output != nil && output.ptr != nil && !leak {
				C.fcuda_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	if verr := ValidateV41CompressorRingStore(kv, scores, kvRow, scoreRow, slot, ratio, width); verr != nil {
		return Tensor{}, cudaV41CompressorError(site, "invalid geometry", 2, verr)
	}
	if err = c.cudaV41CompressorState(site); err != nil {
		return Tensor{}, err
	}
	var operands [4]*cudaBuf
	for i, operand := range []struct {
		name  string
		t     Tensor
		elems int
	}{{"kv ring", kv, ratio * width}, {"score ring", scores, ratio * width}, {"kv row", kvRow, width}, {"score row", scoreRow, width}} {
		if operands[i], err = c.cudaV41CompressorOperand(site, operand.name, operand.t, operand.elems); err != nil {
			return Tensor{}, err
		}
	}
	outPtr := unsafe.Pointer(nil)
	emitFlag := 0
	if emit {
		if output, err = c.dallocDeviceOnlyClass(2*width*4, MemoryActivation, "v41-compressor-ring-output"); err != nil {
			return Tensor{}, cudaV41CompressorError(site, "device-only output allocation failed", -2, err)
		}
		outPtr, emitFlag = output.ptr, 1
	}
	status := int(C.fak_v41_compressor_ring_store_f32((*C.float)(operands[2].ptr), (*C.float)(operands[3].ptr),
		(*C.float)(operands[0].ptr), (*C.float)(operands[1].ptr), (*C.float)(outPtr),
		C.int(ratio), C.int(width), C.int(slot), C.int(emitFlag)))
	if status != 0 {
		leak = status >= 40000
		return Tensor{}, c.cudaV41CompressorNativeError(site, status)
	}
	if !emit {
		return Tensor{}, nil
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, []int{2, width}, nil, output), nil
}

// V41CompressorPool pools one uploaded group and refuses a non-finite group
// with the ring's typed per-dimension error before the row is published.
func (c *cudaBackend) V41CompressorPool(kv, scores Tensor, ratio, width int) (out Tensor, err error) {
	const site = "V41CompressorPool"
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output *cudaBuf
	leak := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "compressor-pool")
			err = cudaV41CompressorError(site, "candidate panicked", -1, cause)
		}
		if err != nil {
			if output != nil && output.ptr != nil && !leak {
				C.fcuda_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	if verr := validateV41CompressorPool(kv, scores, ratio, width); verr != nil {
		return Tensor{}, cudaV41CompressorError(site, "invalid geometry", 2, fmt.Errorf("%w: %v", ErrV41CompressorRingGeometry, verr))
	}
	if err = c.cudaV41CompressorState(site); err != nil {
		return Tensor{}, err
	}
	var operands [2]*cudaBuf
	for i, operand := range []struct {
		name string
		t    Tensor
	}{{"kv", kv}, {"scores", scores}} {
		if operands[i], err = c.cudaV41CompressorOperand(site, operand.name, operand.t, ratio*width); err != nil {
			return Tensor{}, err
		}
	}
	if output, err = c.dallocDeviceOnlyClass(width*4, MemoryActivation, "v41-compressor-pool-output"); err != nil {
		return Tensor{}, cudaV41CompressorError(site, "device-only output allocation failed", -2, err)
	}
	words := make([]float32, width)
	status := int(C.fak_v41_compressor_pool_f32((*C.float)(operands[0].ptr), (*C.float)(operands[1].ptr),
		(*C.float)(output.ptr), (*C.float)(unsafe.Pointer(&words[0])), C.int(ratio), C.int(width)))
	if status != 0 {
		leak = status >= 40000
		return Tensor{}, c.cudaV41CompressorNativeError(site, status)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	if statusErr := v41CompressorRingStatusErr(words); statusErr != nil {
		return Tensor{}, cudaV41CompressorError(site, statusErr.Error(), 1, statusErr)
	}
	return makeTensor(c, F32, RowMajor, []int{width}, nil, output), nil
}

func cudaV41CompressorCompleted() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fak_v41_compressor_completed())
}
