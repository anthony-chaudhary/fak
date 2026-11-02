//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "sync/atomic"

// Selection is immutable native-archive identity, not an environment override.
var cudaQ56KMMQQualification = C.fcuda_q56k_mmq_enabled() != 0

const cudaQ56KMMQOp = "Q56KMMQ"

func CUDAQ56KMMQQualificationEnabled() bool { return cudaQ56KMMQQualification }

func cudaQ56KMMQSelected(w Tensor, p int) bool {
	return cudaQ56KMMQQualification && p >= 128 && (w.Dtype == Q5_K || w.Dtype == Q6_K)
}

func cudaQ56KMMQRefusal(reason string, cause error) error {
	return &CUDAOpError{Op: cudaQ56KMMQOp, Site: "preflight", Msg: reason, Code: -75703, Class: MemoryActivation, Err: cause}
}

// Ordinary Go projections explicitly select this family. Packed GDN and the
// legacy native checked/void ABIs remain unchanged. Whole-Qwen admission checks
// selected weights before recurrent/KV mutation; actual projections repeat
// operand admission when their future F32 activation exists. No host widening.
func (c *cudaBackend) validateQ56KProjectionWeight(w Tensor, p int) error {
	if cudaQ56KMMQSelected(w, p) {
		return c.validateQ56KMMQWeight(w, p)
	}
	return c.validateQ56KMMVQWeight(w, p)
}

func (c *cudaBackend) validateQ56KProjectionOperands(w, x Tensor, p int) error {
	if cudaQ56KMMQSelected(w, p) {
		return c.validateQ56KMMQOperands(w, x, p)
	}
	return c.validateQ56KMMVQOperands(w, x, p)
}

// Caller holds cudaMu and has validated the weight/activation before allocating
// output. Keep the checked native status in the same C call as every enqueue.
func (c *cudaBackend) q56KProjectionStatus(w Tensor, wb *cudaBuf, x, y Tensor, p int) int {
	if cudaQ56KMMQSelected(w, p) {
		kind := 5
		if w.Dtype == Q6_K {
			kind = 6
		}
		return int(C.fcuda_q56k_mmq_matmul_checked_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y),
			C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(p), C.int(kind)))
	}
	if w.Dtype == Q5_K {
		return int(C.fcuda_q5k_matmul_checked_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y),
			C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(p)))
	}
	return int(C.fcuda_q6k_matmul_checked_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y),
		C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(p)))
}
func (c *cudaBackend) validateQ56KMMQWeight(w Tensor, p int) error {
	if !cudaQ56KMMQSelected(w, p) {
		return nil
	}
	if err := c.faultLatch.Admit("q56k-mmq"); err != nil {
		return err
	}
	if c.capturing || cudaQ4KAsyncReleaseBlockedLocked() {
		return cudaQ56KMMQRefusal("capture or pending asynchronous peer owner is unsupported", nil)
	}
	if len(w.Shape) != 2 || w.Shape[0] <= 0 || w.Shape[1] <= 0 || w.Shape[1]%256 != 0 ||
		w.Layout != RowMajor || w.Quant == nil || w.Quant.Block != 256 {
		return cudaQ56KMMQRefusal("require canonical positive block-256 row-major Q5_K/Q6_K", nil)
	}
	// Strictly below INT_MAX matches all native offset/ceil arithmetic.
	for _, pair := range [][2]int{{w.Shape[0], w.Shape[1]}, {p, w.Shape[0]}, {p, w.Shape[1]}} {
		if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt-1, int64(pair[0]), int64(pair[1])); !ok {
			return cudaQ56KMMQRefusal("matrix or panel indexing exceeds the CUDA int ABI", nil)
		}
	}
	blockBytes, kind := 176, 5
	if w.Dtype == Q6_K {
		blockBytes, kind = 210, 6
	}
	b, err := c.cudaQ56KMMVQBuffer(w, w.Shape[0]*(w.Shape[1]/256)*blockBytes)
	if err != nil {
		return cudaQ56KMMQRefusal("packed weight is not a live exclusive device owner", err)
	}
	rc := int(C.fcuda_q56k_mmq_preflight((*C.uint8_t)(b.ptr), nil,
		C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(p), C.int(kind)))
	if rc != 0 {
		return &CUDAOpError{Op: cudaQ56KMMQOp, Site: "weight-admission", Code: rc, Msg: "native packed panel admission refused", Class: MemoryActivation}
	}
	return nil
}

func (c *cudaBackend) validateQ56KMMQOperands(w, x Tensor, p int) error {
	if err := c.validateQ56KMMQWeight(w, p); err != nil {
		return err
	}
	in := w.Shape[1]
	if len(x.Shape) != 2 || x.Shape[0] != p || x.Shape[1] != in ||
		x.Dtype != F32 || x.Layout != RowMajor || x.Quant != nil {
		return cudaQ56KMMQRefusal("activation must be F32 row-major [P,in]", nil)
	}
	b, err := c.cudaQ56KMMVQBuffer(x, p*in*4)
	if err != nil {
		return cudaQ56KMMQRefusal("activation is not a live exclusive device owner", err)
	}
	kind := 5
	if w.Dtype == Q6_K {
		kind = 6
	}
	wb := w.buf.(*cudaBuf)
	rc := int(C.fcuda_q56k_mmq_preflight((*C.uint8_t)(wb.ptr), (*C.float)(b.ptr),
		C.int(w.Shape[0]), C.int(in), C.int(p), C.int(kind)))
	if rc != 0 {
		return &CUDAOpError{Op: cudaQ56KMMQOp, Site: "operand-admission", Code: rc, Msg: "native packed panel operands refused", Class: MemoryActivation}
	}
	return nil
}

// Native status is captured in the same C call as every submission. On error,
// the native family drains once; failed drains retain all live native owners.
// Never return a partial output or authorize a retry after state mutation.
func (c *cudaBackend) q56KMMQFailure(y Tensor, rc int) error {
	if rc == 0 {
		return nil
	}
	if b, ok := y.buf.(*cudaBuf); ok {
		atomic.StoreUint32(&b.invalid, 1)
		c.releaseTransientBuffers([]*cudaBuf{b}) // existing unconditional shared terminal guard
	}
	var cause error
	if fault := c.faultLatch.Observe(DeviceFaultExecution, "q56k-mmq", rc); fault != nil {
		cause = fault
	}
	return &CUDAOpError{Op: cudaQ56KMMQOp, Site: "launch", Code: rc,
		Msg: "retained Q5_K/Q6_K MMQ output withheld", Err: cause}
}

// Reuse the existing generic shared-stream fence bridge (its historical ABI
// name mentions IQ2). It records private split-loan failures and the sticky
// shared terminal together, with one ordinary final fence and no retry.
func (c *cudaBackend) qwen35Q56KMMQFenceLocked() int {
	rc := int(C.fcuda_qwen35_iq2_sequence_fence())
	if rc != 0 {
		c.faultLatch.Observe(DeviceFaultExecution, "q56k-mmq-final-fence", rc)
	}
	return rc
}

func (c *cudaBackend) Q56KMMQLaunchCounts() (uint64, uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_q56k_mmq_launch_count(5)), uint64(C.fcuda_q56k_mmq_launch_count(6))
}

func (c *cudaBackend) cudaQ56KMMQForTest() (bool, uint64, uint64, uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return cudaQ56KMMQQualification, uint64(C.fcuda_q56k_mmq_launch_count(5)),
		uint64(C.fcuda_q56k_mmq_launch_count(6)), atomic.LoadUint64(&c.fenceGen)
}

func (c *cudaBackend) cudaQ56KMMQNullInputForTest(w, y Tensor, p int) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if !cudaQ56KMMQSelected(w, p) || c.validateQ56KMMQWeight(w, p) != nil {
		return -75701
	}
	b, err := c.cudaQ56KMMVQBuffer(y, p*w.Shape[0]*4)
	if err != nil {
		return -75703
	}
	wb := w.buf.(*cudaBuf)
	kind := 5
	if w.Dtype == Q6_K {
		kind = 6
	}
	return int(C.fcuda_q56k_mmq_matmul_checked_f32((*C.uint8_t)(wb.ptr), nil,
		(*C.float)(b.ptr), C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(p), C.int(kind)))
}

// Caller holds cudaMu. The Read completion has already failed; record its exact
// nonzero status before deferred cleanup, without repeating the copy or fence.
func cudaQ56KMMQRetirementFailedLocked(first int) {
	C.fcuda_q56k_mmq_retirement_failed(C.int(first))
}
