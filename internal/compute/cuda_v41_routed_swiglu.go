//go:build cuda && cgo && fak_cuda_v41_routed_swiglu

// SPDX-License-Identifier: Apache-2.0
package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
int fak_v41_routed_swiglu_state(void);
int fak_v41_routed_swiglu_input(const float *, uint64_t);
int fak_v41_routed_swiglu_run(const float *, const float *, float *, unsigned *, unsigned *, int, float, float);
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var _ V41RoutedSwiGLUBackend = (*cudaBackend)(nil)

// Compilation is not numerical qualification. Ordinary model selection stays
// closed until exact module/device evidence is accepted.
func (c *cudaBackend) V41RoutedSwiGLUAvailable() bool { return false }
func (c *cudaBackend) V41RoutedSwiGLU(gate, up Tensor, limit, weight float32) (Tensor, error) {
	return c.cudaV41RoutedSwiGLU(gate, up, limit, weight, false)
}
func (c *cudaBackend) cudaV41RoutedSwiGLUError(status int, cause error) error {
	if c != nil && status > 0 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, "V41RoutedSwiGLU", status)
	}
	return &BackendError{Backend: "cuda", Site: "V41RoutedSwiGLU", Class: "cuda-routed-swiglu-failed", Message: fmt.Sprintf("checked routed SwiGLU refused (status %d)", status), Err: cause}
}

// The private witness enables qualification without selecting unproven math in
// the model. Input/output values never cross the host; only 16 status bytes do.
func (c *cudaBackend) cudaV41RoutedSwiGLU(gate, up Tensor, limit, weight float32, witness bool) (out Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output, statusBuf *cudaBuf
	defer func() {
		if v := recover(); v != nil {
			e, _ := ConvertCUDAPanic(v, "V41RoutedSwiGLU", "routed activation")
			err = c.cudaV41RoutedSwiGLUError(-1, e)
		}
		for _, b := range []*cudaBuf{statusBuf, output} {
			if b != nil && b.ptr != nil && (b == statusBuf || err != nil) {
				C.fcuda_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
		runtime.KeepAlive(gate)
		runtime.KeepAlive(up)
	}()
	count, invalid := validateV41RoutedSwiGLU(gate, up, limit, weight)
	if invalid != nil || count > int(^uint(0)>>1)/4 {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-1, invalid)
	}
	if c == nil || c.capturing {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-3, nil)
	}
	if e := c.faultLatch.Admit("V41RoutedSwiGLU"); e != nil {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-3, e)
	}
	if x := int(C.fak_v41_routed_swiglu_state()); x != 0 {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(x, nil)
	}
	var ptrs [2]*C.float
	for i, t := range []Tensor{gate, up} {
		b, ok := t.buf.(*cudaBuf)
		if t.be != c || !ok || b == nil || b.ptr == nil || b.n < count*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return Tensor{}, c.cudaV41RoutedSwiGLUError(-2, nil)
		}
		if e := b.invalidStateError("V41RoutedSwiGLU"); e != nil {
			return Tensor{}, c.cudaV41RoutedSwiGLUError(-2, e)
		}
		ptrs[i] = (*C.float)(b.ptr)
		if x := int(C.fak_v41_routed_swiglu_input(ptrs[i], C.uint64_t(count))); x != 0 {
			return Tensor{}, c.cudaV41RoutedSwiGLUError(x, nil)
		}
	}
	if !witness {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-3, fmt.Errorf("exact module/device numerical qualification absent"))
	}
	output, err = c.dallocDeviceOnlyClass(count*4, MemoryActivation, "v41-routed-swiglu-output")
	if err != nil {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-2, err)
	}
	statusBuf, err = c.dallocDeviceOnlyClass(v41RoutedSwiGLUStatusBytes, MemoryActivation, "v41-routed-swiglu-status")
	if err != nil {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-2, err)
	}
	var words [4]uint32
	x := int(C.fak_v41_routed_swiglu_run(ptrs[0], ptrs[1], (*C.float)(output.ptr), (*C.uint)(statusBuf.ptr), (*C.uint)(unsafe.Pointer(&words[0])), C.int(count), C.float(limit), C.float(weight)))
	runtime.KeepAlive(words)
	if x != 0 {
		return Tensor{}, c.cudaV41RoutedSwiGLUError(x, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	if e := decodeV41RoutedSwiGLUStatus(words, count); e != nil {
		var fault *V41RoutedSwiGLUFault
		if errors.As(e, &fault) {
			return Tensor{}, fault
		}
		return Tensor{}, c.cudaV41RoutedSwiGLUError(-4, e)
	}
	return makeTensor(c, F32, RowMajor, []int{count}, nil, output), nil
}
