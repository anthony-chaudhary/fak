//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var _ Qwen35MTPDraftBackend = (*cudaBackend)(nil)
var _ Qwen35MTPTransferCounter = (*cudaBackend)(nil)

// Only the existing explicit draft constructor gains this operation. This is
// not CUDA target raw-hidden/prefix replay admission or Q4_K_M/Q6 support.
func (c *cudaBackend) Qwen35MTPDraftPath() string {
	if c != nil && c == cudaDev && os.Getenv("FAK_CUDA_QWEN35_MTP") == "1" {
		return Qwen35MTPDraftPath
	}
	return ""
}
func (c *cudaBackend) Qwen35MTPTransferBytes() (h2d, d2h uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())
}
func (c *cudaBackend) Qwen35MTPFuse(req Qwen35MTPFuseRequest) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.mtpPrelayerWithOps(req, c.Qwen35MTPDraftPath() == Qwen35MTPDraftPath, cudaMTPPrelayerOps{
		admit: func(p [5]unsafe.Pointer, h int, eps float32) int {
			return int(C.fcuda_qwen35_mtp_admit((*C.float)(p[0]), (*C.float)(p[1]), (*C.float)(p[2]), (*C.float)(p[3]), (*C.float)(p[4]), C.int(h), C.float(eps)))
		},
		allocate: func(n int) (Tensor, *cudaBuf, error) { return c.devTrDeviceOnly([]int{n}, F32, "qwen35-mtp-prelayer") },
		run: func(p [5]unsafe.Pointer, joined, out unsafe.Pointer, h int, eps float32) int {
			return int(C.fcuda_qwen35_mtp_f32((*C.float)(p[0]), (*C.float)(p[1]), (*C.float)(p[2]), (*C.float)(p[3]), (*C.float)(p[4]), (*C.float)(joined), (*C.float)(out), C.int(h), C.float(eps)))
		},
		stop: func() { C.fcuda_qwen35_mtp_stop() },
	})
}

// BEGIN MTP prelayer owner core.
// Private software-witness seam. The caller holds cudaMu and, for model calls,
// the complete ForwardExecutionOwner. No callback may escape these scopes.
type cudaMTPPrelayerOps struct {
	admit    func([5]unsafe.Pointer, int, float32) int
	allocate func(int) (Tensor, *cudaBuf, error)
	run      func([5]unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int, float32) int
	stop     func()
}
type cudaMTPPrelayerFlight struct {
	request        Qwen35MTPFuseRequest
	joined, output Tensor
	failure        error
}

var cudaMTPPrelayerActive *cudaMTPPrelayerFlight

func cudaMTPPrelayerError(reason string) error {
	return &UnsupportedQwen35MTPDraftError{Backend: "cuda", Path: Qwen35MTPDraftPath, Stage: "pre-layer fusion", Reason: reason}
}
func (c *cudaBackend) mtpPrelayerInputs(req Qwen35MTPFuseRequest) (p [5]unsafe.Pointer, h int, err error) {
	fail := func(s string) ([5]unsafe.Pointer, int, error) { return p, 0, cudaMTPPrelayerError(s) }
	if c == nil || c != cudaDev || c.capturing || c.graphOnce != nil || c.faultLatch == nil || c.faultLatch.Health().Refusing() {
		return fail("backend is absent, capturing, graph-owned, or stopped")
	}
	if len(req.HiddenNorm.Shape) != 1 {
		return fail("hidden norm must be a vector")
	}
	h = req.HiddenNorm.Shape[0]
	if h <= 0 || h > 5120 || !(req.Epsilon > 0) || math.IsInf(float64(req.Epsilon), 0) {
		return fail("unsupported hidden dimension or epsilon")
	}
	tensors := [5]Tensor{req.PriorHidden, req.CurrentEmbedding, req.HiddenNorm, req.EmbeddingNorm, req.FC}
	for i, t := range tensors {
		elements := h
		shapeOK := len(t.Shape) == 1 && t.Shape[0] == h
		if i == 4 {
			elements = 2 * h * h
			shapeOK = len(t.Shape) == 2 && t.Shape[0] == h && t.Shape[1] == 2*h
		}
		if !shapeOK || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || t.Backend() != c {
			return fail(fmt.Sprintf("operand %d requires exact F32 canonical shape", i))
		}
		b, ok := t.buf.(*cudaBuf)
		if !ok || b == nil || b.ptr == nil || b.n != elements*4 || b.device != 0 || b.managed || b.scales != nil || b.scalesN != 0 ||
			b.graphOncePinned || (b.be != nil && b.be != c) || !b.Ready() || cudaAsyncPending[b.ptr] != nil {
			return fail(fmt.Sprintf("operand %d is not exact ready owned device storage", i))
		}
		if uintptr(b.ptr) > ^uintptr(0)-uintptr(b.n) {
			return fail("operand extent wraps")
		}
		p[i] = b.ptr
	}
	return p, h, nil
}
func (c *cudaBackend) mtpPrelayerWithOps(req Qwen35MTPFuseRequest, enabled bool, ops cudaMTPPrelayerOps) (out Tensor, err error) {
	if !enabled {
		return Tensor{}, cudaMTPPrelayerError("explicit FAK_CUDA_QWEN35_MTP=1 is required")
	}
	if cudaMTPPrelayerActive != nil {
		return Tensor{}, cudaMTPPrelayerError("an earlier operation is active or quarantined")
	}
	p, h, err := c.mtpPrelayerInputs(req)
	if err != nil {
		return Tensor{}, err
	}
	if ops.admit == nil || ops.allocate == nil || ops.run == nil || ops.stop == nil {
		return Tensor{}, cudaMTPPrelayerError("incomplete private operations")
	}
	flight := &cudaMTPPrelayerFlight{request: req}
	cudaMTPPrelayerActive = flight
	normal := false
	defer func() {
		if !normal {
			if flight.failure == nil {
				flight.failure = cudaMTPPrelayerError("uncertain operation or abnormal unwind")
			}
			for _, t := range []Tensor{flight.joined, flight.output} {
				if b, ok := t.buf.(*cudaBuf); ok && b != nil {
					atomic.StoreUint32(&b.invalid, 1)
				}
			}
			c.faultLatch.Observe(DeviceFaultExecution, "qwen35-mtp-prelayer", 95001)
			ops.stop()
			out = Tensor{}
			if err == nil {
				err = flight.failure
			}
		}
		runtime.KeepAlive(req)
	}()
	status := ops.admit(p, h, req.Epsilon)
	if status != 0 {
		err = cudaMTPPrelayerError(fmt.Sprintf("native admission status %d", status))
		flight.failure = err
		if status < 0 {
			normal = true
			cudaMTPPrelayerActive = nil
		}
		return Tensor{}, err
	}
	var joined, output *cudaBuf
	flight.joined, joined, err = ops.allocate(2 * h)
	if err != nil {
		flight.failure = err
		return Tensor{}, err
	}
	flight.output, output, err = ops.allocate(h)
	if err != nil {
		flight.failure = err
		return Tensor{}, err
	}
	valid := func(t Tensor, b *cudaBuf, n int) bool {
		return b != nil && t.buf == b && t.Backend() == c && t.Dtype == F32 && t.Layout == RowMajor && t.Quant == nil &&
			len(t.Shape) == 1 && t.Shape[0] == n && b.ptr != nil && b.n == n*4 && b.device == 0 && !b.managed &&
			b.scales == nil && b.scalesN == 0 && b.be == c && !b.invalidForSubmit() && !b.graphOncePinned
	}
	if !valid(flight.joined, joined, 2*h) || !valid(flight.output, output, h) {
		err = cudaMTPPrelayerError("allocator returned incompatible private storage")
		flight.failure = err
		return Tensor{}, err
	}
	status = ops.run(p, joined.ptr, output.ptr, h, req.Epsilon)
	if status != 0 {
		err = cudaMTPPrelayerError(fmt.Sprintf("checked native operation status %d", status))
		flight.failure = err
		return Tensor{}, err
	}
	// Native success includes checked stream completion, not mere submission.
	atomic.AddUint64(&c.fenceGen, 1)
	// Both allocations remain ordinary token transients until the caller recycles.
	// No new retirement boundary is introduced before publishing the checked output.
	out = flight.output
	cudaMTPPrelayerActive = nil
	normal = true
	return out, nil
}

// END MTP prelayer owner core.
