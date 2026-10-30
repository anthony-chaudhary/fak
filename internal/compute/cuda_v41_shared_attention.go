//go:build cuda && cgo && fak_cuda_v41_shared_attention

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI. The library must be built with FAK_CUDA_V41_SHARED_ATTENTION.
int fak_v41_shared_attention_state(void);
int fak_v41_shared_attention_f32(const float *, const float *, const float *, float *, unsigned *,
    int, int, int, int, int, float, int, int *, int *, float *);
uint64_t fak_v41_shared_attention_completed(void);
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

var _ V41SharedAttentionBackend = (*cudaBackend)(nil)

// The CUDA candidate (fak#13993) runs the FlashMLA numerical class: split-KV
// tiled online softmax in base 2, an N-way LSE combine, sink folded last. It is
// pinned against v41SparseMLADecodeReference, never bit-compared with the
// serial three-pass oracle, exactly like Vulkan's RunToRun/SplitK classes. The
// base-2 rescale narrows the finite envelope: a scaled score beyond
// MaxFloat32/log2(e) is a typed score fault, and Plain/Compressed are not
// distinguished (they differ only when every score equals -MaxFloat32).
// Compiling the candidate tag is the opt-in; the default cuda build lacks the
// method set, so the model keeps the host path.
func (c *cudaBackend) SupportsV41SharedAttention() bool {
	if c == nil {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.cudaV41SharedAttentionState("SupportsV41SharedAttention") == nil
}

func cudaV41SharedAttentionError(site, message string, status int, cause error) error {
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-v41-shared-attention-failed",
		Message: fmt.Sprintf("%s (status %d)", message, status), Err: cause, Recovered: cause}
}

// Caller holds cudaMu. Faults take precedence over native admission.
func (c *cudaBackend) cudaV41SharedAttentionState(site string) error {
	if c == nil || c.capturing {
		return cudaV41SharedAttentionError(site, "nil backend or active capture", -3, nil)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return cudaV41SharedAttentionError(site, "sticky CUDA fault", -3, err)
	}
	if status := int(C.fak_v41_shared_attention_state()); status != 0 {
		return cudaV41SharedAttentionError(site, "native admission refused", status, nil)
	}
	return nil
}

// cudaV41SharedAttentionRun is one checked dispatch. Splits 0 lets the native
// planner choose; the returned tile and split count name the reference
// partition the output must match.
type cudaV41SharedAttentionRun struct {
	Tile, Splits int
	KernelMS     float32
}

func (c *cudaBackend) V41SharedAttention(q, sharedKV, sink Tensor, selectedRows, heads, headDim int, scale float32, mode V41SharedAttentionMode, hasSink bool) (Tensor, error) {
	out, _, err := c.cudaV41SharedAttention(q, sharedKV, sink, selectedRows, heads, headDim, scale, mode, hasSink, 0)
	return out, err
}

func (c *cudaBackend) cudaV41SharedAttention(q, sharedKV, sink Tensor, selectedRows, heads, headDim int, scale float32, mode V41SharedAttentionMode, hasSink bool, forceSplits int) (out Tensor, run cudaV41SharedAttentionRun, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output *cudaBuf
	leak := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, "V41SharedAttention", "shared-attention")
			err = cudaV41SharedAttentionError("V41SharedAttention", "candidate panicked", -1, cause)
		}
		if err != nil {
			// An uncertain drain may still have the kernel writing output:
			// never return that allocation to the pool.
			if output != nil && output.ptr != nil && !leak {
				C.fcuda_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	qBytes, kvBytes, sinkBytes, statusBytes, validation := validateV41SharedAttention(q, sharedKV, sink, selectedRows, heads, headDim, scale, mode, hasSink)
	if validation != nil {
		return Tensor{}, run, cudaV41SharedAttentionError("V41SharedAttention", "invalid geometry", 2, validation)
	}
	if err = c.cudaV41SharedAttentionState("V41SharedAttention"); err != nil {
		return Tensor{}, run, err
	}
	var buffers [3]*cudaBuf
	for i, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{{"q", q, qBytes}, {"sharedKV", sharedKV, kvBytes}, {"sink", sink, sinkBytes}} {
		b, ok := operand.t.buf.(*cudaBuf)
		if operand.t.be != c || !ok || b == nil || b.ptr == nil || b.n < operand.bytes || b.managed ||
			b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return Tensor{}, run, cudaV41SharedAttentionError("V41SharedAttention", operand.name+" is not owned device-only F32 storage", -2, nil)
		}
		if invalid := b.invalidStateError("V41SharedAttention " + operand.name); invalid != nil {
			return Tensor{}, run, cudaV41SharedAttentionError("V41SharedAttention", "invalid storage lifetime", -2, invalid)
		}
		buffers[i] = b
	}
	output, err = c.dallocDeviceOnlyClass(qBytes, MemoryActivation, "v41-shared-attention-output")
	if err != nil {
		return Tensor{}, run, cudaV41SharedAttentionError("V41SharedAttention", "device-only output allocation failed", -2, err)
	}
	words := make([]uint32, statusBytes/4)
	var tile, splits C.int
	var ms C.float
	var sinkFlag C.int
	if hasSink {
		sinkFlag = 1
	}
	status := int(C.fak_v41_shared_attention_f32((*C.float)(buffers[0].ptr), (*C.float)(buffers[1].ptr),
		(*C.float)(buffers[2].ptr), (*C.float)(output.ptr), (*C.unsigned)(unsafe.Pointer(&words[0])),
		C.int(heads), C.int(headDim), C.int(selectedRows), C.int(mode), sinkFlag, C.float(scale),
		C.int(forceSplits), &tile, &splits, &ms))
	if status != 0 {
		var cause error
		if status >= 30000 {
			leak = status >= 40000
			cause = c.faultLatch.Observe(DeviceFaultExecution, "V41SharedAttention", status)
		}
		return Tensor{}, run, cudaV41SharedAttentionError("V41SharedAttention", "checked CUDA shared attention refused", status, cause)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	if statusErr := decodeV41SharedAttentionStatus(words, heads, selectedRows, headDim); statusErr != nil {
		return Tensor{}, run, cudaV41SharedAttentionError("V41SharedAttention", statusErr.Error(), 1, statusErr)
	}
	run = cudaV41SharedAttentionRun{Tile: int(tile), Splits: int(splits), KernelMS: float32(ms)}
	return makeTensor(c, F32, RowMajor, []int{heads, headDim}, nil, output), run, nil
}

func cudaV41SharedAttentionCompleted() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fak_v41_shared_attention_completed())
}
