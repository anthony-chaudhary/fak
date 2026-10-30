//go:build cuda && cgo && fak_cuda_v41_compressor_pool

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
int fak_v41_pool_state(void);
int fak_v41_pool_input(const float *, uint64_t);
int fak_v41_pool_run(const float *, const float *, float *, int, int);
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync/atomic"
)

var _ V41CompressorPoolBackend = (*cudaBackend)(nil)

// Compilation is not numerical qualification. The ordinary model stays on its
// existing path until exact-module/device proof enables a separate change.
func (c *cudaBackend) SupportsV41CompressorPool() bool { return false }
func (c *cudaBackend) V41CompressorPool(kv, scores Tensor, ratio, width int) (Tensor, error) {
	return c.cudaV41CompressorPool(kv, scores, ratio, width, false)
}
func (c *cudaBackend) cudaV41PoolError(status int, cause error) error {
	if c != nil && status > 0 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, "V41CompressorPool", status)
	}
	return &BackendError{Backend: "cuda", Site: "V41CompressorPool", Class: "cuda-compressor-pool-failed", Message: fmt.Sprintf("checked compressor pool refused (status %d)", status), Err: cause}
}

// witness is package-private for exact-module/device qualification only.
func (c *cudaBackend) cudaV41CompressorPool(kv, scores Tensor, ratio, width int, witness bool) (out Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output *cudaBuf
	defer func() {
		if p := recover(); p != nil {
			cause, _ := ConvertCUDAPanic(p, "V41CompressorPool", "pool")
			err = c.cudaV41PoolError(-1, cause)
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				C.fcuda_free(output.ptr)
				output.ptr = nil
				output.n = 0
			}
			out = Tensor{}
		}
		runtime.KeepAlive(kv)
		runtime.KeepAlive(scores)
	}()
	if e := validateV41CompressorPool(kv, scores, ratio, width); e != nil {
		return Tensor{}, c.cudaV41PoolError(-1, e)
	}
	if c == nil || c.capturing {
		return Tensor{}, c.cudaV41PoolError(-3, nil)
	}
	if e := c.faultLatch.Admit("V41CompressorPool"); e != nil {
		return Tensor{}, c.cudaV41PoolError(-3, e)
	}
	if s := int(C.fak_v41_pool_state()); s != 0 {
		return Tensor{}, c.cudaV41PoolError(s, nil)
	}
	inputs := [2]Tensor{kv, scores}
	var ptrs [2]*C.float
	for i, t := range inputs {
		b, ok := t.buf.(*cudaBuf)
		if t.be != c || !ok || b == nil || b.ptr == nil || b.n < ratio*width*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return Tensor{}, c.cudaV41PoolError(-2, fmt.Errorf("input %d is not owned F32 device storage", i))
		}
		if e := b.invalidStateError("V41CompressorPool"); e != nil {
			return Tensor{}, c.cudaV41PoolError(-2, e)
		}
		ptrs[i] = (*C.float)(b.ptr)
		if s := int(C.fak_v41_pool_input(ptrs[i], C.uint64_t(ratio*width))); s != 0 {
			return Tensor{}, c.cudaV41PoolError(s, nil)
		}
	}
	if !witness {
		return Tensor{}, c.cudaV41PoolError(-3, fmt.Errorf("exact module/device numerical qualification absent"))
	}
	// One extra word holds finite-input/output status, never part of the tensor.
	output, err = c.dallocDeviceOnlyClass((width+1)*4, MemoryActivation, "v41-compressor-pool-output")
	if err != nil {
		return Tensor{}, c.cudaV41PoolError(-2, err)
	}
	if s := int(C.fak_v41_pool_run(ptrs[0], ptrs[1], (*C.float)(output.ptr), C.int(ratio), C.int(width))); s != 0 {
		return Tensor{}, c.cudaV41PoolError(s, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, []int{width}, nil, output), nil
}
