//go:build cuda && cgo && fak_cuda_v41_engram_gate

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
int fak_v41_engram_state(void);
int fak_v41_engram_readback(float *,const float *,uint64_t);
int fak_v41_engram_input(const float *, uint64_t);
int fak_v41_engram_run(const float *, const float *, const float *, float *, float *, int, int, float);
*/
import "C"
import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var _ V41EngramGateBackend = (*cudaBackend)(nil)

// Numerical proof is still required. Ordinary status Read must retain the
// checked CUDA copy and selected-device panic boundary pinned in PROVENANCE.md.
func (c *cudaBackend) SupportsV41EngramGate() bool { return false }
func (c *cudaBackend) V41EngramGate(s, p, n Tensor, rows, hidden int, eps float32) (Tensor, Tensor, error) {
	return c.cudaV41EngramGate(s, p, n, rows, hidden, eps, false)
}
func (c *cudaBackend) cudaV41EngramError(status int, cause error) error {
	if c != nil && status > 0 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, "V41EngramGate", status)
	}
	return &BackendError{Backend: "cuda", Site: "V41EngramGate", Class: "cuda-engram-gate-failed", Message: fmt.Sprintf("checked Engram gate refused (status %d)", status), Err: cause}
}

// witness is exclusively for in-package exact-module/device qualification.
func (c *cudaBackend) cudaV41EngramGate(s, p, n Tensor, rows, hidden int, eps float32, witness bool) (next, status Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var outputs [2]*cudaBuf
	defer func() {
		if v := recover(); v != nil {
			e, _ := ConvertCUDAPanic(v, "V41EngramGate", "engram")
			err = c.cudaV41EngramError(-1, e)
		}
		if err != nil {
			for _, b := range outputs {
				if b != nil && b.ptr != nil {
					C.fcuda_free(b.ptr)
					b.ptr = nil
					b.n = 0
				}
			}
			next = Tensor{}
			status = Tensor{}
		}
		runtime.KeepAlive(s)
		runtime.KeepAlive(p)
		runtime.KeepAlive(n)
	}()
	// Bound by division before any potentially wrapping row product.
	if rows <= 0 || hidden <= 0 || hidden > V41EngramGateMaxHidden || rows > math.MaxInt32/(5*hidden) || rows > int(^uint(0)>>1)/(20*hidden) {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(-1, nil)
	}
	if e := validateV41EngramGate(s, p, n, rows, hidden, eps); e != nil {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(-1, e)
	}
	// Existing interface checks native indexing; additionally bound host byte counts.
	if uint64(rows)*5*uint64(hidden) > uint64(int(^uint(0)>>1)/4) {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(-1, nil)
	}
	if c == nil || c.capturing {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(-3, nil)
	}
	if e := c.faultLatch.Admit("V41EngramGate"); e != nil {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(-3, e)
	}
	if x := int(C.fak_v41_engram_state()); x != 0 {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(x, nil)
	}
	inputs := [3]Tensor{s, p, n}
	counts := [3]int{rows * 4 * hidden, rows * 5 * hidden, 8 * hidden}
	var ptrs [3]*C.float
	for i, t := range inputs {
		b, ok := t.buf.(*cudaBuf)
		if t.be != c || !ok || b == nil || b.ptr == nil || b.n < counts[i]*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return Tensor{}, Tensor{}, c.cudaV41EngramError(-2, nil)
		}
		if e := b.invalidStateError("V41EngramGate"); e != nil {
			return Tensor{}, Tensor{}, c.cudaV41EngramError(-2, e)
		}
		ptrs[i] = (*C.float)(b.ptr)
		if x := int(C.fak_v41_engram_input(ptrs[i], C.uint64_t(counts[i]))); x != 0 {
			return Tensor{}, Tensor{}, c.cudaV41EngramError(x, nil)
		}
	}
	if !witness {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(-3, fmt.Errorf("exact module/device numerical qualification absent"))
	}
	for i, count := range []int{rows * 4 * hidden, rows * 4} {
		outputs[i], err = c.dallocDeviceOnlyClass(count*4, MemoryActivation, "v41-engram-gate-output")
		if err != nil {
			return Tensor{}, Tensor{}, c.cudaV41EngramError(-2, err)
		}
	}
	x := int(C.fak_v41_engram_run(ptrs[0], ptrs[1], ptrs[2], (*C.float)(outputs[0].ptr), (*C.float)(outputs[1].ptr), C.int(rows), C.int(hidden), C.float(eps)))
	if x != 0 {
		return Tensor{}, Tensor{}, c.cudaV41EngramError(x, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, []int{rows, 4, hidden}, nil, outputs[0]), makeTensor(c, F32, RowMajor, []int{rows, 4}, nil, outputs[1]), nil
}

// Qualification uses explicit returned errors; ordinary Read separately fails
// closed through its checked copy and typed panic boundary.
func (c *cudaBackend) cudaV41EngramReadback(t Tensor, count int) ([]float32, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	site := "V41EngramGate readback"
	if c == nil || c.capturing || count <= 0 || uint64(count) > math.MaxUint32 || count > int(^uint(0)>>1)/4 || t.be != c || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil {
		return nil, c.cudaV41EngramError(-2, nil)
	}
	total := uint64(1)
	for _, dim := range t.Shape {
		if dim <= 0 || uint64(dim) > uint64(count)/total {
			return nil, c.cudaV41EngramError(-2, nil)
		}
		total *= uint64(dim)
	}
	if total != uint64(count) {
		return nil, c.cudaV41EngramError(-2, nil)
	}
	if e := c.faultLatch.Admit(site); e != nil {
		return nil, c.cudaV41EngramError(-3, e)
	}
	b, ok := t.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.n < count*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return nil, c.cudaV41EngramError(-2, nil)
	}
	if e := b.invalidStateError(site); e != nil {
		return nil, c.cudaV41EngramError(-2, e)
	}
	host := make([]float32, count)
	status := int(C.fak_v41_engram_readback((*C.float)(unsafe.Pointer(&host[0])), (*C.float)(b.ptr), C.uint64_t(count)))
	runtime.KeepAlive(t)
	runtime.KeepAlive(host)
	if status != 0 {
		return nil, c.cudaV41EngramError(status, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return host, nil
}
