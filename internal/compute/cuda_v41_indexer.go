//go:build cuda && cgo && fak_cuda_v41_indexer

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI. The library must be built with FAK_CUDA_V41_INDEXER.
int fak_v41_indexer_state(void);
int fak_v41_indexer_input(const float *, uint64_t);
int fak_v41_indexer_score_f32(const float *, const float *, const float *, float *, int, int, int);
int fak_v41_indexer_readback(float *, const float *, uint64_t);
uint64_t fak_v41_indexer_completed(void);
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var _ V41IndexerScoreBackend = (*cudaBackend)(nil)

// No environment variable, build tag or source review substitutes for an exact
// archive/compiler/device numerical witness. The ordinary model binding stays
// absent until a separate qualification change establishes that evidence.
func (c *cudaBackend) SupportsV41IndexerScore() bool { return false }
func (c *cudaBackend) V41IndexerScoreUnavailableReason() string {
	return "CUDA indexer SIMT candidate lacks exact module/device F32 qualification"
}
func (c *cudaBackend) V41IndexerScoreSpareKeyCapacity() bool { return true }

func cudaV41IndexerError(site, message string, status int, cause error) error {
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-indexer-failed",
		Message: fmt.Sprintf("%s (status %d)", message, status), Err: cause}
}

// Caller holds cudaMu. Faults take precedence over absent qualification.
func (c *cudaBackend) cudaV41IndexerState(site string) error {
	if c == nil || c.capturing {
		return cudaV41IndexerError(site, "nil backend or active capture", -3, nil)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return cudaV41IndexerError(site, "sticky CUDA fault", -3, err)
	}
	if status := int(C.fak_v41_indexer_state()); status != 0 {
		return c.cudaV41IndexerStatus(site, status)
	}
	return nil
}

func (c *cudaBackend) cudaV41IndexerStatus(site string, status int) error {
	var cause error
	if c != nil && status > 0 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, site, status)
	}
	return cudaV41IndexerError(site, "checked CUDA indexer operation refused", status, cause)
}

func (c *cudaBackend) V41IndexerScoreAdmission() (bool, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if err := c.cudaV41IndexerState("V41IndexerScoreAdmission"); err != nil {
		return false, err
	}
	return false, nil
}

func (c *cudaBackend) V41IndexerScore(q, keys, weights Tensor, rows, heads, headDim int) (Tensor, error) {
	return c.cudaV41IndexerScore(q, keys, weights, rows, heads, headDim, false)
}

// The internal witness arm is callable only by this package's physical test;
// Backend has no qualification override. Normal invocations always pass false.
func (c *cudaBackend) cudaV41IndexerScore(q, keys, weights Tensor, rows, heads, headDim int, witness bool) (out Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output *cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, "V41IndexerScore", "indexer")
			err = cudaV41IndexerError("V41IndexerScore", "candidate panicked", -1, cause)
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				// Both native free entries preserve terminal quarantine first.
				C.fcuda_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	qb, kb, wb, ob, validation := validateV41IndexerScore(q, keys, weights, rows, heads, headDim)
	if validation != nil {
		return Tensor{}, cudaV41IndexerError("V41IndexerScore", "invalid geometry", -1, validation)
	}
	if err = c.cudaV41IndexerState("V41IndexerScore"); err != nil {
		return Tensor{}, err
	}
	var buffers [3]*cudaBuf
	for i, operand := range []struct {
		name  string
		t     Tensor
		bytes int
		spare bool
	}{{"q", q, qb, false}, {"keys", keys, kb, true}, {"weights", weights, wb, false}} {
		if operand.bytes == 0 && operand.t.buf == nil && operand.t.be == nil {
			continue
		}
		b, ok := operand.t.buf.(*cudaBuf)
		if operand.t.be != c || !ok || b == nil || b.ptr == nil || b.n < operand.bytes ||
			(!operand.spare && b.n != operand.bytes) || b.managed || b.device != 0 ||
			b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return Tensor{}, cudaV41IndexerError("V41IndexerScore", operand.name+" is not owned device-only F32 storage", -2, nil)
		}
		if invalid := b.invalidStateError("V41IndexerScore " + operand.name); invalid != nil {
			return Tensor{}, cudaV41IndexerError("V41IndexerScore", "invalid storage lifetime", -2, invalid)
		}
		if operand.bytes > 0 {
			if status := int(C.fak_v41_indexer_input((*C.float)(b.ptr), C.uint64_t(operand.bytes/4))); status != 0 {
				return Tensor{}, c.cudaV41IndexerStatus("V41IndexerScore input", status)
			}
		}
		buffers[i] = b
	}
	if rows == 0 {
		return makeTensor(c, F32, RowMajor, []int{0}, nil, &cudaBuf{}), nil
	}
	if !witness {
		return Tensor{}, cudaV41IndexerError("V41IndexerScore", c.V41IndexerScoreUnavailableReason(), -3, nil)
	}
	output, err = c.dallocDeviceOnlyClass(ob, MemoryActivation, "v41-indexer-output")
	if err != nil {
		return Tensor{}, cudaV41IndexerError("V41IndexerScore", "device-only output allocation failed", -2, err)
	}
	status := int(C.fak_v41_indexer_score_f32((*C.float)(buffers[0].ptr), (*C.float)(buffers[1].ptr),
		(*C.float)(buffers[2].ptr), (*C.float)(output.ptr), C.int(rows), C.int(heads), C.int(headDim)))
	if status != 0 {
		return Tensor{}, c.cudaV41IndexerStatus("V41IndexerScore execution", status)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, []int{rows}, nil, output), nil
}

func (c *cudaBackend) cudaV41IndexerReadback(t Tensor, rows int) ([]float32, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if rows <= 0 || rows > int(^uint(0)>>1)/4 || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) != 1 || t.Shape[0] != rows || t.be != c {
		return nil, cudaV41IndexerError("V41IndexerReadback", "invalid output", -2, nil)
	}
	if err := c.cudaV41IndexerState("V41IndexerReadback"); err != nil {
		return nil, err
	}
	b, ok := t.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.n != rows*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 {
		return nil, cudaV41IndexerError("V41IndexerReadback", "invalid output storage", -2, nil)
	}
	if err := b.invalidStateError("V41IndexerReadback"); err != nil {
		return nil, err
	}
	values := make([]float32, rows)
	status := int(C.fak_v41_indexer_readback((*C.float)(unsafe.Pointer(&values[0])), (*C.float)(b.ptr), C.uint64_t(rows)))
	runtime.KeepAlive(values)
	if status != 0 {
		return nil, c.cudaV41IndexerStatus("V41IndexerReadback", status)
	}
	return values, nil
}

func cudaV41IndexerCompleted() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fak_v41_indexer_completed())
}
