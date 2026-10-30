//go:build cuda && cgo && fak_cuda_v41_mhc

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
int fak_v41_mhc_state(void);
int fak_v41_mhc_readback(float *,const float *,uint64_t);
int fak_v41_mhc_input(const float *, uint64_t);
int fak_v41_mhc_run(int,const float *,const float *,const float *,float *,int,int,int,float,float);
*/
import "C"
import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var _ V41MHCBackend = (*cudaBackend)(nil)

// The explicit build tag compiles a candidate, not a numerical qualification.
func (c *cudaBackend) SupportsV41MHC() bool { return false }
func (c *cudaBackend) V41MHCInitialStreams(x Tensor, rows, hidden int) (Tensor, error) {
	return c.cudaV41MHC(V41MHCOpInitial, rows, hidden, 0, 0, 0, false, x)
}
func (c *cudaBackend) V41MHCSplit(m, s, b Tensor, rows, iters int, eps float32) (Tensor, error) {
	return c.cudaV41MHC(V41MHCOpSplit, rows, 0, iters, eps, 0, false, m, s, b)
}
func (c *cudaBackend) V41MHCPre(s, k Tensor, rows, hidden int) (Tensor, error) {
	return c.cudaV41MHC(V41MHCOpPre, rows, hidden, 0, 0, 0, false, s, k)
}
func (c *cudaBackend) V41MHCPost(x, s, k Tensor, rows, hidden int) (Tensor, error) {
	return c.cudaV41MHC(V41MHCOpPost, rows, hidden, 0, 0, 0, false, x, s, k)
}
func (c *cudaBackend) V41MHCRMSSplit(raw, s, p Tensor, rows, hidden, iters int, eps, rmsEps float32) (Tensor, error) {
	return c.cudaV41MHC(V41MHCOpRMSSplit, rows, hidden, iters, eps, rmsEps, false, raw, s, p)
}
func (c *cudaBackend) cudaV41MHCError(site string, status int, cause error) error {
	if c != nil && status > 0 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, site, status)
	}
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-mhc-failed", Message: fmt.Sprintf("checked mHC candidate refused (status %d)", status), Err: cause}
}

// witness is package-private for exact-module/device qualification. Public calls
// never override missing qualification and never replay on CPU after dispatch.
func (c *cudaBackend) cudaV41MHC(op V41MHCOp, rows, hidden, iters int, eps, rmsEps float32, witness bool, inputs ...Tensor) (out Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	site := "V41MHC " + op.String()
	var output *cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "mhc")
			err = c.cudaV41MHCError(site, -1, cause)
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				C.fcuda_free(output.ptr)
				output.ptr = nil
				output.n = 0
			}
			out = Tensor{}
		}
		runtime.KeepAlive(inputs)
	}()
	operands, shape, validation := validateV41MHC(op, rows, hidden, iters, eps, inputs...)
	if validation != nil {
		return Tensor{}, c.cudaV41MHCError(site, -1, validation)
	}
	if op == V41MHCOpRMSSplit && (rmsEps <= 0 || math.IsInf(float64(rmsEps), 0) || math.IsNaN(float64(rmsEps))) {
		return Tensor{}, c.cudaV41MHCError(site, -1, fmt.Errorf("invalid rms epsilon"))
	}
	if c == nil || c.capturing {
		return Tensor{}, c.cudaV41MHCError(site, -3, nil)
	}
	if e := c.faultLatch.Admit(site); e != nil {
		return Tensor{}, c.cudaV41MHCError(site, -3, e)
	}
	if s := int(C.fak_v41_mhc_state()); s != 0 {
		return Tensor{}, c.cudaV41MHCError(site, s, nil)
	}
	var ptrs [3]*C.float
	for i, operand := range operands {
		n := v41MHCElements(operand.shape) * 4
		b, ok := operand.t.buf.(*cudaBuf)
		if operand.t.be != c || !ok || b == nil || b.ptr == nil || b.n < n || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return Tensor{}, c.cudaV41MHCError(site, -2, fmt.Errorf("%s is not owned F32 device storage", operand.name))
		}
		if e := b.invalidStateError(site); e != nil {
			return Tensor{}, c.cudaV41MHCError(site, -2, e)
		}
		ptrs[i] = (*C.float)(b.ptr)
		if s := int(C.fak_v41_mhc_input(ptrs[i], C.uint64_t(n/4))); s != 0 {
			return Tensor{}, c.cudaV41MHCError(site, s, nil)
		}
	}
	if !witness {
		return Tensor{}, c.cudaV41MHCError(site, -3, fmt.Errorf("mHC exact module/device numerical qualification is absent"))
	}
	output, err = c.dallocDeviceOnlyClass(v41MHCElements(shape)*4, MemoryActivation, "v41-mhc-output")
	if err != nil {
		return Tensor{}, c.cudaV41MHCError(site, -2, err)
	}
	status := int(C.fak_v41_mhc_run(C.int(op), ptrs[0], ptrs[1], ptrs[2], (*C.float)(output.ptr), C.int(rows), C.int(hidden), C.int(iters), C.float(eps), C.float(rmsEps)))
	if status != 0 {
		return Tensor{}, c.cudaV41MHCError(site, status, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, shape, nil, output), nil
}

// Qualification reads must not rely on ordinary Read's logging-only failure path.
func (c *cudaBackend) cudaV41MHCReadback(t Tensor, count int) ([]float32, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	site := "V41MHC readback"
	if c == nil || c.capturing || count <= 0 || uint64(count) > math.MaxUint32 || count > int(^uint(0)>>1)/4 || t.be != c || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil {
		return nil, c.cudaV41MHCError(site, -2, nil)
	}
	total := uint64(1)
	for _, dim := range t.Shape {
		if dim <= 0 || uint64(dim) > uint64(count)/total {
			return nil, c.cudaV41MHCError(site, -2, nil)
		}
		total *= uint64(dim)
	}
	if total != uint64(count) {
		return nil, c.cudaV41MHCError(site, -2, nil)
	}
	if e := c.faultLatch.Admit(site); e != nil {
		return nil, c.cudaV41MHCError(site, -3, e)
	}
	b, ok := t.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.n < count*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return nil, c.cudaV41MHCError(site, -2, nil)
	}
	if e := b.invalidStateError(site); e != nil {
		return nil, c.cudaV41MHCError(site, -2, e)
	}
	host := make([]float32, count)
	status := int(C.fak_v41_mhc_readback((*C.float)(unsafe.Pointer(&host[0])), (*C.float)(b.ptr), C.uint64_t(count)))
	runtime.KeepAlive(t)
	runtime.KeepAlive(host)
	if status != 0 {
		return nil, c.cudaV41MHCError(site, status, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return host, nil
}
