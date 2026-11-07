//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var _ SamplingCandidateBackend = (*cudaBackend)(nil)

func (c *cudaBackend) SamplingCandidates(row Tensor, k int, retain bool, out *SamplingCandidatePacket) (kept Tensor, available bool, err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cudaMu.Lock()
	defer cudaMu.Unlock()
	opError := func(site string, code int, cause error) error {
		if code > 0 {
			c.faultLatch.Observe(DeviceFaultExecution, site, code)
		}
		return &CUDAOpError{Op: "SamplingCandidates", Site: site, Code: code, Msg: "checked raw candidate transport failed", Err: cause}
	}
	if err = c.faultLatch.Admit("sampling-candidates"); err != nil {
		return Tensor{}, false, opError("sampling-candidates-state", 0, err)
	}
	status := int(C.fcuda_sampling_candidates_capability())
	if status < 0 {
		return Tensor{}, false, opError("sampling-candidates-state", -status, nil)
	}
	if status == 0 || c.capturing || out == nil || row.Backend() != c ||
		row.Dtype != F32 || row.Layout != RowMajor || row.Quant != nil || len(row.Shape) == 0 ||
		k < 1 || k > SamplingCandidateMaxK {
		return Tensor{}, false, nil
	}
	vocab := 1
	for _, dim := range row.Shape {
		if dim <= 0 || dim > math.MaxInt32/vocab || dim > (math.MaxInt/4)/vocab {
			return Tensor{}, false, nil
		}
		vocab *= dim
	}
	if k >= vocab {
		return Tensor{}, false, nil
	}
	b, ok := row.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.n < vocab*4 || b.device != 0 || b.managed ||
		b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return Tensor{}, false, nil
	}
	if err = b.invalidStateError("SamplingCandidates row"); err != nil {
		return Tensor{}, false, opError("sampling-candidates-row", 0, err)
	}
	var scratch, retained *cudaBuf
	published := false
	defer func() {
		// Native failure latches before returning. These release calls then
		// preserve every live native allocation rather than pooling unknown work.
		if scratch != nil {
			code := int(C.fcuda_sampling_candidates_release(scratch.ptr))
			if code != 0 {
				releaseErr := opError("sampling-candidates-workspace-release", code, nil)
				if err == nil {
					err = releaseErr
				}
			}
		}
		if retained != nil && (!published || err != nil) {
			code := int(C.fcuda_sampling_candidates_release(retained.ptr))
			if code != 0 {
				releaseErr := opError("sampling-candidates-retained-release", code, nil)
				if err == nil {
					err = releaseErr
				}
			}
		}
		if err != nil {
			kept, available = Tensor{}, false
		}
	}()
	workspaceBytes := uint64(C.fcuda_sampling_candidates_workspace_bytes())
	if workspaceBytes == 0 || workspaceBytes > uint64(math.MaxInt) {
		return Tensor{}, false, opError("sampling-candidates-workspace", 0, nil)
	}
	allocationError := func(site string, cause error) error {
		code := int(C.fcuda_sampling_candidates_allocation_status())
		if code != 0 || cause != nil {
			return opError(site, code, cause)
		}
		return nil
	}
	scratch, err = c.dallocDeviceOnlyClass(int(workspaceBytes), MemoryScratchpad, "sampling-candidates-workspace")
	if err = allocationError("sampling-candidates-allocation", err); err != nil {
		return Tensor{}, false, err
	}
	if retain {
		retained, err = c.dallocDeviceOnlyClass(vocab*4, MemoryActivation, "sampling-candidates-retained")
		if err = allocationError("sampling-candidates-retained-allocation", err); err != nil {
			return Tensor{}, false, err
		}
	}
	var retainedPtr *C.uint32_t
	if retained != nil {
		retainedPtr = (*C.uint32_t)(retained.ptr)
	}
	status = int(C.fcuda_sampling_candidates((*C.uint32_t)(b.ptr), C.int(vocab), C.int(k), scratch.ptr,
		retainedPtr, (*C.uint32_t)(unsafe.Pointer(&out.nativeWords[0]))))
	runtime.KeepAlive(out)
	runtime.KeepAlive(row)
	if status != 0 {
		return Tensor{}, false, opError("sampling-candidates-execution", status, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	out.Vocab, out.Count, out.Valid = vocab, int(out.nativeWords[1]), out.nativeWords[0] == 2
	if out.Count == k+1 {
		for i := 0; i < out.Count; i++ {
			out.IDs[i] = int(out.nativeWords[2+2*i])
			out.Raw[i] = math.Float32frombits(out.nativeWords[3+2*i])
		}
	}
	if retained != nil {
		kept = makeTensor(c, F32, RowMajor, append([]int(nil), row.Shape...), nil, retained)
	}
	published = true
	return kept, true, nil
}
