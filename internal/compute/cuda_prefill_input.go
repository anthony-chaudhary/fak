//go:build cuda && cgo

package compute

import (
	"fmt"
	"runtime"
)

// One global owner matches the native global stream/allocator. Access requires
// cudaMu. A failed completion retains both the device handle and pinned source.
var cudaPrefillInputOwner struct {
	buffer   *cudaBuf
	host     []float32
	pin      runtime.Pinner
	terminal error
}

type cudaPrefillInputOperations struct {
	admit   func() int
	acquire func(host []float32, panel bool, dst *cudaBuf) int
	retire  func(*cudaBuf) int
	failed  func(int)
}

func (c *cudaBackend) prefillHostProjections(x Tensor, host []float32, wq, wk, wv Tensor, P int) (q, k, v []float32, err error) {
	if x.Dtype != F32 || x.Layout != RowMajor || x.Numel() <= 0 || x.Numel() > len(host) || x.Numel() > int(^uint(0)>>1)/4 {
		return nil, nil, nil, fmt.Errorf("compute: invalid host prefill input extent or layout")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	owner := &cudaPrefillInputOwner
	if owner.terminal != nil {
		return nil, nil, nil, owner.terminal
	}
	if owner.buffer != nil || c.capturing || c.graphOnce != nil || cudaGraphOnceOwner != nil {
		return nil, nil, nil, fmt.Errorf("compute: prefill input owner or graph scope occupied")
	}
	if err = AdmitDevice(c, "prefill-input"); err != nil {
		return
	}
	if status := cudaPrefillInputNative.admit(); status != 0 {
		return nil, nil, nil, &CUDAOpError{Op: "PrefillBatch", Site: "input-admission", Code: status, Msg: "owned input admission refused", Class: MemoryActivation}
	}
	// Validate every needed upload layout before touching native storage.
	for _, w := range []Tensor{wq, wk, wv} {
		if _, host := c.Host(w); host {
			requireCUDAUploadLayout("cuda Upload", w, F32, MemoryWeights, "upload-weight")
		}
	}
	// Publish the Go owner before pinning, allocation or any fallible submission.
	owner.buffer = &cudaBuf{n: x.Numel() * 4, class: MemoryActivation}
	// A private snapshot survives uncertain H2D without retaining caller-owned
	// mutable bytes for an in-flight native read after this call returns.
	owner.host = append([]float32(nil), host[:x.Numel()]...)
	defer func() {
		primary := recover()
		cleanupErr := func() (cleanupErr error) {
			defer func() {
				if failure := recover(); failure != nil {
					cleanupErr = fmt.Errorf("compute: prefill input cleanup panicked: %v", failure)
					if owner.terminal == nil {
						owner.terminal = cleanupErr
					}
				}
			}()
			// Read reports a checked completion failure through this exact typed seam.
			// Preserve it before a later retirement could observe a successful fence.
			if failure, ok := primary.(*CUDAOpError); ok && failure.Op == "Read" && failure.Site == "read-d2h" && failure.Code != 0 {
				owner.terminal = failure
				cudaPrefillInputNative.failed(failure.Code)
			}
			if owner.terminal != nil {
				return owner.terminal
			}
			if owner.buffer.ptr != nil {
				if status := cudaPrefillInputNative.retire(owner.buffer); status != 0 {
					cleanupErr = &CUDAOpError{Op: "PrefillBatch", Site: "input-retirement", Code: status, Msg: "input retained after retirement failure", Class: MemoryActivation}
					owner.terminal = cleanupErr
					return cleanupErr
				}
				// Native acknowledgement is the only permission to clear this handle.
				owner.buffer.ptr = nil
			}
			owner.pin.Unpin()
			owner.buffer = nil
			owner.host = nil
			return nil
		}()
		if primary != nil {
			panic(primary)
		}
		if err == nil {
			err = cleanupErr
		}
	}()
	// Immutable weights keep their original cache policy, inside this admitted
	// critical section. No public locking upload is called recursively.
	if _, host := c.Host(wq); host {
		wq = c.uploadClassLocked(wq, F32, MemoryWeights, "upload-weight")
	}
	if _, host := c.Host(wk); host {
		wk = c.uploadClassLocked(wk, F32, MemoryWeights, "upload-weight")
	}
	if _, host := c.Host(wv); host {
		wv = c.uploadClassLocked(wv, F32, MemoryWeights, "upload-weight")
	}
	owner.pin.Pin(&owner.host[0])
	status := cudaPrefillInputNative.acquire(owner.host, c.promptPanel, owner.buffer)
	runtime.KeepAlive(owner.host)
	if status != 0 {
		err = &CUDAOpError{Op: "PrefillBatch", Site: "input-upload", Code: status, Msg: "owned input acquisition failed", Class: MemoryActivation}
		if owner.buffer.ptr != nil {
			owner.terminal = err
		}
		return
	}
	// Fresh private input: no Upload/UploadClass cache and no devTr registration.
	input := makeTensor(c, F32, RowMajor, append([]int(nil), x.Shape...), nil, owner.buffer)
	qt := c.batchedMatMulLocked(wq, input, P)
	kt := c.batchedMatMulLocked(wk, input, P)
	vt := c.batchedMatMulLocked(wv, input, P)
	q = c.readLocked(qt)
	k = c.readLocked(kt)
	v = c.readLocked(vt)
	return
}
