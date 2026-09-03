//go:build cuda

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakcuda -lcudart -lcublas -lstdc++ -lm
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"sync/atomic"
	"unsafe"
)

// DSASparseAttend runs GLM-MoE-DSA's sparse attention (model.glmDsaAttendCached's inner loop) on the
// device via k_dsa_sparse_attend: per query head, softmax(scale·q·selK)·selV over the nSel host-SELECTED
// keys. q/selK/selV arrive device-resident (the model uploads the query + the host-gathered selected K/V
// rows); selK is [nSel,nH*qkHead], selV [nSel,nH*vHead] (qkHead and vHead differ under MLA). It is the
// optional compute.DSASparseBackend capability — the cuda backend advertises it, so a GLM-5.2 forward on
// this backend runs its attention math on the pure GPU kernel, not host-resident. Approx vs the cpuref
// reference (cudaDsaSparseAttnCosineMin); the selection (index scores + top-k) is host-side, so the
// device attends the same keys and the divergence is only f32 reduction order.
func (c *cudaBackend) DSASparseAttend(q, selK, selV Tensor, nSel, nH, qkHead, vHead int, scale float32) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, _ := c.devTr([]int{nH * vHead}, F32)
	C.fcuda_dsa_sparse_attend_f32(c.cf(q), c.cf(selK), c.cf(selV), c.cf(out),
		C.int(nSel), C.int(nH), C.int(qkHead), C.int(vHead), C.float(scale))
	return out
}

// DSAIndexSelect runs GLM-MoE-DSA's learned-indexer score + top-k SELECTION on the device via
// k_dsa_index_score + k_dsa_index_topk: it scores every cached key against the query
// (Σ_h weights[h]·relu(scale·dot)), masks keys past queryPos, and returns the top-k selected key
// POSITIONS — the last GLM-5.2 compute that was host-resident even after the projections and the
// sparse-attention math moved to the kernel. The per-key score dot is accumulated in f64 on-device,
// so the selected set is bit-identical to the host f64 selection (selection-stable — the indexer
// drives a discrete top-k, so it is held reduction-faithful, NOT to a cosine floor). It is the
// optional compute.DSAIndexBackend capability; a backend without it leaves the selection host-side.
// q/k/weights arrive device-resident; only the small index list crosses back to the host.
func (c *cudaBackend) DSAIndexSelect(indexQ, indexK, weights Tensor, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if nKeys <= 0 || topK <= 0 {
		return nil
	}
	out := make([]C.int, topK)
	n := int(C.fcuda_dsa_index_select_f32(c.cf(indexQ), c.cf(indexK), c.cf(weights),
		C.int(nKeys), C.int(nH), C.int(indexDim), C.int(queryPos), C.int(topK), C.float(scale),
		&out[0]))
	atomic.AddUint64(&c.fenceGen, 1) // the index list crossed host-ward — same fence as Argmax
	if n < 0 {
		// The device DECLINED (nKeys past the shared-mem top-k cap): return an empty selection so the
		// model's glmDsaValidSelection rejects it and keeps the host f64 score+top-k loop — the device
		// path can only ever match the host selection, never silently degrade it on a long window.
		return nil
	}
	if n > topK {
		n = topK
	}
	sel := make([]int, n)
	for i := 0; i < n; i++ {
		sel[i] = int(out[i])
	}
	return sel
}

// attentionNaive runs the same op through the RETAINED naive kernel (full global scores[nH*nPos]
// scratch, four passes). It exists only so the #486 microbench can time fused-vs-naive on identical
// inputs; the live Attention path never calls it. Same arguments, same result up to reduction order.
func (c *cudaBackend) attentionNaive(q Tensor, kv KVStore, layer int, grp int, scale float32) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	ck := kv.(*cudaKV)
	hd, nKV := ck.cfg.HeadDim, ck.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := ck.K[layer].len / w
	out, _ := c.devTr([]int{nH * hd}, F32)
	C.fcuda_attention_f32(c.cf(q),
		(*C.float)(ck.K[layer].ptr), (*C.float)(ck.V[layer].ptr),
		c.cf(out), C.int(nPos), C.int(ck.maxPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	return out
}

// Argmax is the OTHER host fence (#482) and the one greedy decode uses: it runs the reduction
// ON-DEVICE (k_argmax over the resident logits) and copies back only the single winning token
// id — the full logits vector never crosses the bus. Like Read, the int copy drains g_stream,
// so it bumps the fence generation (the logits it reduced are now Ready).
func (c *cudaBackend) Argmax(logits Tensor) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	id := int(C.fcuda_argmax_f32(c.cf(logits), C.int(logits.Numel())))
	atomic.AddUint64(&c.fenceGen, 1)
	return id
}

// ---- async host-transfer witness (#482) -----------------------------------------
//
// HostXferBytes reports the cumulative bytes copied DEVICE->HOST since the last reset. The two
// host fences are the only d2h transfers and both feed this counter: fcuda_d2h (a full Read)
// adds the vector's bytes, while fcuda_argmax_f32 adds only sizeof(int). So over an Argmax-only
// decode step it reads the size of one token id, whereas a full-logits Read reads vocab*4 —
// the seam the witness test reads to prove only the argmax id crosses the bus per token.
// ResetHostXfer zeroes it. H2DXferBytes/ResetH2DXfer are the independent
// host->device half of the same witness. Access and reset are serialized with
// device work so a measured interval cannot race an upload or proof read.
func (c *cudaBackend) HostXferBytes() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_hostxfer_bytes())
}

func (c *cudaBackend) ResetHostXfer() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_hostxfer_reset()
}

func (c *cudaBackend) H2DXferBytes() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_h2dxfer_bytes())
}

func (c *cudaBackend) ResetH2DXfer() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_h2dxfer_reset()
}

// ---- Qwen3.5/3.6 Gated-DeltaNet whole-operation decode (#4725) -----------------

// Qwen35GDNPath is the stable structural identity consumed by
// model.Qwen35GDNBackend. The compute package cannot import model, so the method
// returns the cycle-free constant declared beside the typed refusals.
func (c *cudaBackend) Qwen35GDNPath() string { return Qwen35GDNCUDAPath }

// Qwen35GDNOperationCount reports successful whole GDN operations completed since
// the last reset. The C ABI increments only after cudaStreamSynchronize confirms
// every stage executed successfully. It is a device-path witness, not a
// performance estimate: the deterministic CUDA fixture pairs it with both H2D
// and D2H counters to prove the measured call ran wholly on device.
func (c *cudaBackend) Qwen35GDNOperationCount() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_qwen35_gdn_operations())
}

func (c *cudaBackend) ResetQwen35GDNOperationCount() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_qwen35_gdn_operations_reset()
}

// qwen35GDNInjectFaultForTest arms a one-shot deterministic C-side failure at a
// launch stage (2..6) or at final synchronization (7). It is intentionally
// unexported: only package-local -tags cuda tests can exercise the fail-closed
// path, while the production Go API exposes no fault knob.
func qwen35GDNInjectFaultForTest(stage int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_qwen35_gdn_test_fault(C.int(stage))
}

func (c *cudaBackend) allocateQwen35GDN(allocations []qwen35GDNAllocation, sitePrefix string) ([]Tensor, []*cudaBuf, error) {
	tensors := make([]Tensor, 0, len(allocations))
	buffers := make([]*cudaBuf, 0, len(allocations))
	for _, allocation := range allocations {
		site := sitePrefix + allocation.name
		tensor, buffer, err := c.devTrDeviceOnly(allocation.shape, F32, site)
		if err != nil {
			c.releaseTransientBuffers(buffers)
			c.faultLatch.ObserveError(err, site)
			return nil, nil, err
		}
		tensors = append(tensors, tensor)
		buffers = append(buffers, buffer)
	}
	return tensors, buffers, nil
}

func (c *cudaBackend) qwen35GDNQ8Args(t Tensor) (unsafe.Pointer, *C.float, C.int) {
	buf := c.cudaBufForSubmit(t)
	if t.Dtype == Q8_0 {
		return buf.ptr, (*C.float)(buf.scales), 1
	}
	return buf.ptr, nil, 0
}

// failQwen35GDN invalidates every possibly partial result and both mutable
// states before releasing operation-owned buffers and poisoning the session.
func (c *cudaBackend) failQwen35GDN(buffers []*cudaBuf, convState, recurrentState Tensor, status int, site string) *Qwen35GDNKernelError {
	for _, buffer := range buffers {
		atomic.StoreUint32(&buffer.invalid, 1)
	}
	atomic.StoreUint32(&convState.buf.(*cudaBuf).invalid, 1)
	atomic.StoreUint32(&recurrentState.buf.(*cudaBuf).invalid, 1)
	c.releaseTransientBuffers(buffers)
	kernelErr := &Qwen35GDNKernelError{Stage: qwen35GDNKernelStage(status), Code: status}
	c.faultLatch.ObserveError(kernelErr, site)
	return kernelErr
}

// Qwen35GDNDecode executes one complete recurrent linear-attention token mixer:
// one fused qkv/z/b/a projection launch, causal depthwise conv + conv-state
// update, q/k L2 normalization, decay/beta delta-rule state update fused with
// the per-head gated RMSNorm, and the output projection. Every pointer handed to
// the C ABI is a validated CUDA allocation. The method never calls Host/Read,
// never constructs a []float32, and never enters generic QKV attention or a CPU
// fallback. Conv/recurrent states are updated in place and returned as the next
// device-resident states.
func (c *cudaBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState Tensor, err error) {
	// Session fault gate (#6412), FIRST — before geometry, locks, or any allocation. Once a
	// prior operation observed a device fault, this context is suspect and no result computed
	// on it may be returned; the typed refusal outranks every per-operand diagnosis.
	if err := c.faultLatch.Admit("qwen35-gdn-decode"); err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	_, hidden, keyDim, valueDim, convDim, operands, err := qwen35GDNEntry(normalizedInput, normalizedInput, inProjQKV, inProjZ, inProjB, inProjA, conv1D, aLog, dtBias, norm, outProj, convState, recurrentState, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon)
	if err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if err := c.validateQwen35GDNOperands(operands, convState, recurrentState); err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}

	// All scratch/output must be cudaMalloc-backed device memory. The general
	// allocator's managed-memory fallback is deliberately not available here.
	allocations := qwen35GDNAllocations("", "_", 0, 0, hidden, keyDim, valueDim, numValueHeads, convDim)
	strictTensors, strictBuffers, allocErr := c.allocateQwen35GDN(allocations, "qwen35-gdn-")
	if allocErr != nil {
		return Tensor{}, Tensor{}, Tensor{}, allocErr
	}
	mixed, z, b, a := strictTensors[0], strictTensors[1], strictTensors[2], strictTensors[3]
	convOut, qNorm, kNorm, core := strictTensors[4], strictTensors[5], strictTensors[6], strictTensors[7]
	output = strictTensors[8]

	qkvPtr, qkvScale, qkvQ8 := c.qwen35GDNQ8Args(inProjQKV)
	zPtr, zScale, zQ8 := c.qwen35GDNQ8Args(inProjZ)
	bPtr, bScale, bQ8 := c.qwen35GDNQ8Args(inProjB)
	aPtr, aScale, aQ8 := c.qwen35GDNQ8Args(inProjA)
	outPtr, outScale, outQ8 := c.qwen35GDNQ8Args(outProj)
	status := int(C.fcuda_qwen35_gdn_decode_f32(
		c.cf(normalizedInput),
		qkvPtr, qkvScale, qkvQ8,
		zPtr, zScale, zQ8,
		bPtr, bScale, bQ8,
		aPtr, aScale, aQ8,
		c.cf(conv1D), c.cf(aLog), c.cf(dtBias), c.cf(norm),
		outPtr, outScale, outQ8,
		c.cf(convState), c.cf(recurrentState), c.cf(output),
		c.cf(mixed), c.cf(z), c.cf(b), c.cf(a), c.cf(convOut),
		c.cf(qNorm), c.cf(kNorm), c.cf(core),
		C.int(hidden), C.int(numKeyHeads), C.int(numValueHeads),
		C.int(keyHeadDim), C.int(valueHeadDim), C.int(convKernel),
		C.float(rmsNormEpsilon),
	))
	if status != 0 {
		// The ABI drains the stream before returning any launch error and fences
		// once at the end to surface asynchronous execution errors. A failed
		// in-place update may therefore be partial: invalidate both mutable states
		// plus all outputs/scratch so no caller can observe or reuse them as valid.
		// The ABI drained the stream, so operation-owned buffers are safe to free;
		// durable state remains caller-owned but invalid for explicit cleanup.
		kernelErr := c.failQwen35GDN(strictBuffers, convState, recurrentState, status, "qwen35-gdn-decode")
		return Tensor{}, Tensor{}, Tensor{}, kernelErr
	}
	// fcuda_qwen35_gdn_decode_f32 synchronized g_stream successfully. Record the
	// host fence so this output and any earlier stream-ordered outputs become Ready.
	atomic.AddUint64(&c.fenceGen, 1)
	return output, convState, recurrentState, nil
}

// ---- AWQ (Activation-aware Weight Quantization) 4-bit matmul -------------------

// AWQMatMul computes y = W @ x where W is an AWQ 4-bit quantized tensor.
// W is [out, in] stored as 4-bit packed bytes [out, in/2], with per-channel scales [out].
func (c *cudaBackend) AWQMatMul(w, scales, x Tensor) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	wbuf := c.cudaBufForSubmit(w)
	sbuf := c.cudaBufForSubmit(scales)
	xbuf := c.cudaBufForSubmit(x)
	y, _ := c.devTr([]int{out}, F32)

	// Get device pointers
	wp := wbuf.ptr
	sp := sbuf.ptr
	xp := xbuf.ptr
	yp := c.cf(y)

	C.fcuda_awq_gemv((*C.uint8_t)(wp), (*C.float)(sp), (*C.float)(xp), yp, C.int(out), C.int(in))
	return y
}

// AWQBatchedMatMul computes Y = X @ W^T where W is an AWQ 4-bit quantized tensor.
// X is [P, in], W is [out, in] stored as 4-bit packed [out, in/2], scales is [out].
// Output Y is [P, out].
func (c *cudaBackend) AWQBatchedMatMul(w, scales, X Tensor, P int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	wbuf := c.cudaBufForSubmit(w)
	sbuf := c.cudaBufForSubmit(scales)
	xbuf := c.cudaBufForSubmit(X)
	y, _ := c.devTr([]int{P, out}, F32)

	// Get device pointers
	wp := wbuf.ptr
	sp := sbuf.ptr
	xp := xbuf.ptr
	yp := c.cf(y)

	C.fcuda_awq_gemm((*C.uint8_t)(wp), (*C.float)(sp), (*C.float)(xp), yp, C.int(out), C.int(in), C.int(P))
	return y
}

// ---- GPTQ (AutoGPTQ/GPTQModel) int32-packed weight-only matmul ------------------

// GPTQMatMul computes y = W @ x where W is a native packed GPTQ 4/8-bit quantized weight,
// dequant-fused into the device GEMV tile (the GPU remainder of the CPU-resident spine, #3030).
// The operands mirror internal/model/gptq.go's resident layout exactly:
//   - qweight: int32-packed codes [in/pack, out]   (pack = 32/bits)
//   - qzeros:  int32-packed zero-points [nGroups, out/pack]
//   - scales:  f32 [nGroups, out]
//   - gidx:    optional int32 [in] activation-order group index; pass the zero Tensor
//     (Tensor{}) for the group = i/groupSize path (no g_idx / desc_act off).
//
// The quant tensors are staged resident as raw device bytes (their host dtype label is
// cosmetic — the kernel reads only the device pointers + these explicit dims). Output is F32 [out].
func (c *cudaBackend) GPTQMatMul(qweight, qzeros, scales, gidx, x Tensor, out, in, bits, groupSize, nGroups int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	wbuf := c.cudaBufForSubmit(qweight)
	zbuf := c.cudaBufForSubmit(qzeros)
	sbuf := c.cudaBufForSubmit(scales)
	xbuf := c.cudaBufForSubmit(x)
	var gbuf *cudaBuf
	if gidx.buf != nil {
		gbuf = c.cudaBufForSubmit(gidx)
	}
	y, _ := c.devTr([]int{out}, F32)

	wp := (*C.uint32_t)(wbuf.ptr)
	zp := (*C.uint32_t)(zbuf.ptr)
	sp := (*C.float)(sbuf.ptr)
	var gp *C.int32_t
	if gbuf != nil {
		gp = (*C.int32_t)(gbuf.ptr)
	}
	xp := (*C.float)(xbuf.ptr)
	yp := c.cf(y)

	C.fcuda_gptq_gemv(wp, zp, sp, gp, xp, yp,
		C.int(out), C.int(in), C.int(bits), C.int(groupSize), C.int(nGroups))
	return y
}

// ---- device-resident KV store ---------------------------------------------------
