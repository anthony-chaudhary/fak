//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"
)

var _ Qwen35IndependentDecodeBatchBackend = (*cudaBackend)(nil)

// qwen35DecodeBatchKVChunk is the growable-KV reservation quantum, in positions.
const qwen35DecodeBatchKVChunk = 256

// Qwen35IndependentDecodeBatchPath advertises the shared capability identity the model
// layer dispatches on; the CUDA implementation honours the same contract as Vulkan.
func (*cudaBackend) Qwen35IndependentDecodeBatchPath() string {
	return Qwen35IndependentDecodeBatchPath
}

// cudaQwen35LaneRowView is a non-owning row view of a transient panel. It is never
// registered as transient and never freed; the parent panel owns the allocation.
func (c *cudaBackend) cudaQwen35LaneRowView(parent *cudaBuf, row, width int) Tensor {
	view := &cudaBuf{ptr: offsetF(parent.ptr, row*width), n: width * F32.Bytes(), class: parent.class, be: parent.be, bornGen: parent.bornGen}
	return makeTensor(c, F32, RowMajor, []int{1, width}, nil, view)
}

// Qwen35IndependentDecodeBatch advances 2..8 independent one-token Qwen3.5/3.8 lanes in one
// device step. Every projection (GDN in/out, attention Q/K/V/O, SwiGLU, head) runs once over
// the [lanes, width] activation panel, so each weight byte streams once per step instead of
// once per lane. Conv/recurrent state, RoPE position, KV append, and attention stay per lane.
// Failures before the first state or KV write are declines (Mutated=false) the model replays
// serially; later failures poison every lane's state and report Mutated=true.
func (c *cudaBackend) Qwen35IndependentDecodeBatch(req Qwen35IndependentDecodeBatchRequest) (result Qwen35IndependentDecodeBatchResult, err error) {
	if err := validateQwen35IndependentDecodeBatchContract(req); err != nil {
		return result, err
	}
	if admitErr := c.faultLatch.Admit("qwen35-independent-decode-batch"); admitErr != nil {
		return result, &Qwen35IndependentDecodeBatchError{Stage: "device-admission", Cause: admitErr}
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	lanes := len(req.Lanes)
	first := req.Lanes[0]
	stage, mutated, completed := "preflight", false, false
	transientStart := len(c.transient)
	fail := func(cause error) error {
		if mutated {
			for _, lane := range req.Lanes {
				c.qwen35SequencePoisonStates(lane)
			}
		}
		return &Qwen35IndependentDecodeBatchError{Stage: stage, Mutated: mutated, Cause: cause}
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, ok := recovered.(error)
			if !ok {
				cause = fmt.Errorf("%v", recovered)
			}
			err = fail(cause)
		}
		if !completed {
			c.qwen35SequenceReleaseSinceLocked(transientStart)
		}
	}()

	kvs := make([]*cudaKV, lanes)
	statePointers := make(map[unsafe.Pointer]int)
	for i, lane := range req.Lanes {
		if !sameQwen35DecodeLaneWeights(first, lane) {
			return result, fail(fmt.Errorf("lane %d does not share the resident weights of lane 0", i))
		}
		if lane.TokenEmbeddingRows != first.TokenEmbeddingRows || (!lane.TokenEmbeddingRows && lane.TokenEmbedding.Buf() != first.TokenEmbedding.Buf()) {
			return result, fail(fmt.Errorf("lane %d embedding source differs from lane 0", i))
		}
		kv, verr := c.validateQwen35SequenceRequestLocked(lane)
		if verr != nil {
			return result, fail(verr)
		}
		for layer := range lane.Layers {
			if !lane.Layers[layer].Linear {
				continue
			}
			for _, state := range []Tensor{lane.States[layer].Conv, lane.States[layer].Recurrent} {
				ptr := state.buf.(*cudaBuf).ptr
				if owner, aliased := statePointers[ptr]; aliased && owner != i {
					return result, fail(fmt.Errorf("lane %d layer %d GDN state aliases lane %d", i, layer, owner))
				}
				statePointers[ptr] = i
			}
		}
		kvs[i] = kv
	}
	compactLayers := len(first.Layers) / 4
	stage = "kv-reserve"
	for i, lane := range req.Lanes {
		// A growable store reserves in 256-position chunks: an exact +1 reservation would
		// reallocate and copy every attention layer's KV on every step, and the allocator churn
		// of those ever-growing sizes filled a 24 GB card within ~150 steps (Raider, Q4_K_M).
		// A fixed graph store (maxPos > 0) already holds its whole capacity.
		positions := lane.StartPos + 1
		if kvs[i].maxPos == 0 {
			positions = (positions + qwen35DecodeBatchKVChunk - 1) / qwen35DecodeBatchKVChunk * qwen35DecodeBatchKVChunk
		}
		if rerr := c.qwen35SequenceReserveKVLocked(kvs[i], compactLayers, positions*kvs[i].stride()); rerr != nil {
			return result, fail(rerr)
		}
	}

	hidden, eps := first.Hidden, first.RMSNormEpsilon
	h2dStart, d2hStart := uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())
	stage = "embedding"
	var x Tensor
	var xBuf *cudaBuf
	if first.TokenEmbeddingRows {
		x, xBuf, err = c.qwen35SequenceAllocLocked([]int{lanes, hidden}, "batch-embedding-rows")
		if err != nil {
			return result, fail(err)
		}
		for i, lane := range req.Lanes {
			C.fcuda_d2d(offsetF(xBuf.ptr, i*hidden), c.cudaBufForSubmit(lane.TokenEmbedding).ptr, C.size_t(hidden*F32.Bytes()))
		}
	} else {
		ids := make([]int, lanes)
		for i, lane := range req.Lanes {
			ids[i] = lane.TokenIDs[0]
		}
		x, xBuf, err = c.qwen35SequenceEmbeddingLocked(first.TokenEmbedding, ids, hidden)
		if err != nil {
			return result, fail(err)
		}
	}
	h2dAfterGather, d2hAfterGather := uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())

	panels, macs := 0, int64(0)
	project := func(w, in Tensor, site string) (Tensor, *cudaBuf, error) {
		panels++
		macs += int64(w.Numel()) * int64(lanes)
		return c.qwen35SequenceMatMulLocked(w, in, lanes, site)
	}
	qWidth, kvWidth := first.NumHeads*first.HeadDim, first.NumKVHeads*first.HeadDim
	scale := float32(1 / math.Sqrt(float64(first.HeadDim)))
	compact := 0
	for index, layer := range first.Layers {
		stage = fmt.Sprintf("layer-%d", index)
		normed, normBuf, nerr := c.qwen35SequenceRMSNormLocked(x, layer.InputNorm, lanes, eps, "batch-input-norm")
		if nerr != nil {
			return result, fail(nerr)
		}
		var postNorm Tensor
		var postNormBuf *cudaBuf
		fused := false
		var rerr error
		if layer.Linear {
			mutated = true
			branch, branchBuf, gerr := c.qwen35BatchGDNLocked(normed, layer, req, index)
			if gerr != nil {
				return result, fail(gerr)
			}
			for _, w := range []Tensor{layer.GDNInQKV, layer.GDNInZ, layer.GDNInB, layer.GDNInA, layer.GDNOut} {
				panels++
				macs += int64(w.Numel()) * int64(lanes)
			}
			postNorm, postNormBuf, fused, rerr = c.qwen35SequenceResidualNormLocked(x, branch, layer.PostNorm, lanes, eps)
			if rerr != nil {
				return result, fail(rerr)
			}
			if !fused {
				C.fcuda_add_f32(c.cf(x), c.cf(branch), C.int(lanes*hidden))
			}
			c.releaseTransientBuffers([]*cudaBuf{normBuf, branchBuf})
		} else {
			qg, qgBuf, perr := project(layer.Q, normed, "batch-attention-qg")
			if perr != nil {
				return result, fail(perr)
			}
			kRaw, kRawBuf, perr := project(layer.K, normed, "batch-attention-k")
			if perr != nil {
				return result, fail(perr)
			}
			_, valueBuf, perr := project(layer.V, normed, "batch-attention-v")
			if perr != nil {
				return result, fail(perr)
			}
			q, qBuf, gate, gateBuf, serr := c.qwen35SequenceSplitQGLocked(qg, lanes, first.NumHeads, first.HeadDim)
			if serr != nil {
				return result, fail(serr)
			}
			c.releaseTransientBuffers([]*cudaBuf{qgBuf})
			if layer.QNorm.Buf() != nil {
				qNormed, qNormBuf, e := c.qwen35SequenceRMSNormLocked(q, layer.QNorm, lanes*first.NumHeads, eps, "batch-attention-q-norm")
				if e != nil {
					return result, fail(e)
				}
				kNormed, kNormBuf, e := c.qwen35SequenceRMSNormLocked(kRaw, layer.KNorm, lanes*first.NumKVHeads, eps, "batch-attention-k-norm")
				if e != nil {
					return result, fail(e)
				}
				c.releaseTransientBuffers([]*cudaBuf{qBuf, kRawBuf})
				q, qBuf, kRaw, kRawBuf = qNormed, qNormBuf, kNormed, kNormBuf
			}
			attn, attnBuf, aerr := c.qwen35SequenceAllocLocked([]int{lanes, qWidth}, "batch-attention")
			if aerr != nil {
				return result, fail(aerr)
			}
			mutated = true
			for i, lane := range req.Lanes {
				qRow := c.cudaQwen35LaneRowView(qBuf, i, qWidth)
				kRow := c.cudaQwen35LaneRowView(kRawBuf, i, kvWidth)
				vRow := c.cudaQwen35LaneRowView(valueBuf, i, kvWidth)
				qRoPE, qRoPEBuf, kRoPE, kRoPEBuf, ropeErr := c.qwen35SequenceRoPELocked(qRow, kRow, 1, lane.StartPos, first.NumHeads, first.NumKVHeads, first.HeadDim, first.RotaryDim, first.RoPEThetaForLayer[index])
				if ropeErr != nil {
					return result, fail(ropeErr)
				}
				if appendErr := c.qwen35SequenceAppendKVLocked(kvs[i], compact, kRow, kRoPE, vRow, lane.StartPos, 1); appendErr != nil {
					return result, fail(appendErr)
				}
				if attnErr := c.qwen35BatchAttentionIntoLocked(qRoPE, kvs[i], compact, lane.StartPos, first.NumHeads, scale, c.cudaQwen35LaneRowView(attnBuf, i, qWidth)); attnErr != nil {
					return result, fail(attnErr)
				}
				c.releaseTransientBuffers([]*cudaBuf{qRoPEBuf, kRoPEBuf})
			}
			C.fcuda_sigmoid_mul_f32(c.cf(attn), c.cf(gate), C.int(lanes*qWidth))
			projected, projectedBuf, perr := project(layer.O, attn, "batch-attention-output")
			if perr != nil {
				return result, fail(perr)
			}
			postNorm, postNormBuf, fused, rerr = c.qwen35SequenceResidualNormLocked(x, projected, layer.PostNorm, lanes, eps)
			if rerr != nil {
				return result, fail(rerr)
			}
			if !fused {
				C.fcuda_add_f32(c.cf(x), c.cf(projected), C.int(lanes*hidden))
			}
			c.releaseTransientBuffers([]*cudaBuf{normBuf, qBuf, kRawBuf, valueBuf, gateBuf, attnBuf, projectedBuf})
			compact++
		}
		if !fused {
			var perr error
			postNorm, postNormBuf, perr = c.qwen35SequenceRMSNormLocked(x, layer.PostNorm, lanes, eps, "batch-post-attention-norm")
			if perr != nil {
				return result, fail(perr)
			}
		}
		gate, gateBuf, gerr := project(layer.Gate, postNorm, "batch-ffn-gate")
		if gerr != nil {
			return result, fail(gerr)
		}
		up, upBuf, uerr := project(layer.Up, postNorm, "batch-ffn-up")
		if uerr != nil {
			return result, fail(uerr)
		}
		activated, activatedBuf, allocErr := c.qwen35SequenceAllocLocked([]int{lanes, first.Intermediate}, "batch-ffn-swiglu")
		if allocErr != nil {
			return result, fail(allocErr)
		}
		C.fcuda_swiglu_f32(c.cf(gate), c.cf(up), c.cf(activated), C.int(lanes*first.Intermediate))
		delta, deltaBuf, derr := project(layer.Down, activated, "batch-ffn-down")
		if derr != nil {
			return result, fail(derr)
		}
		C.fcuda_add_f32(c.cf(x), c.cf(delta), C.int(lanes*hidden))
		c.releaseTransientBuffers([]*cudaBuf{postNormBuf, gateBuf, upBuf, activatedBuf, deltaBuf})
	}
	stage = "output-head"
	finalPanel, finalPanelBuf, ferr := c.qwen35SequenceRMSNormLocked(x, first.OutputNorm, lanes, eps, "batch-output-norm")
	if ferr != nil {
		return result, fail(ferr)
	}
	logits, logitsBuf, herr := project(first.Output, finalPanel, "batch-output-head")
	if herr != nil {
		return result, fail(herr)
	}
	logits.Shape = []int{lanes, first.Output.Shape[0]}
	c.releaseTransientBuffers([]*cudaBuf{xBuf, finalPanelBuf})
	stage = "final-fence"
	if serr := c.qwen35SequenceSyncLocked(); serr != nil {
		atomic.StoreUint32(&logitsBuf.invalid, 1)
		c.faultLatch.ObserveError(serr, "qwen35-independent-decode-batch-final-fence")
		return result, fail(serr)
	}
	h2dEnd, d2hEnd := uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())
	transfers := Qwen35SequenceTransferCounters{
		H2DBytes: h2dEnd - h2dStart, D2HBytes: d2hEnd - d2hStart,
		ActivationH2DBytes: h2dEnd - h2dAfterGather, ActivationD2HBytes: d2hEnd - d2hAfterGather,
	}
	if transfers.ActivationH2DBytes != 0 || transfers.ActivationD2HBytes != 0 {
		stage = "transfer-witness"
		return result, fail(fmt.Errorf("activation traffic H2D=%d D2H=%d", transfers.ActivationH2DBytes, transfers.ActivationD2HBytes))
	}
	result = Qwen35IndependentDecodeBatchResult{Logits: logits, Lanes: lanes, SharedProjectionPanels: panels, ProjectionMACs: macs, Transfers: transfers}
	c.qwen35SequenceReleaseSinceLocked(transientStart, logitsBuf)
	completed = true
	return result, nil
}

// qwen35BatchGDNLocked runs one GDN mixer for every lane: shared-weight in/out projections
// over the [lanes, hidden] panel and per-lane conv/recurrent updates in lane-owned state.
func (c *cudaBackend) qwen35BatchGDNLocked(x Tensor, layer Qwen35SequenceLayer, req Qwen35IndependentDecodeBatchRequest, layerIndex int) (Tensor, *cudaBuf, error) {
	first := req.Lanes[0]
	lanes := len(req.Lanes)
	keyDim := first.NumKeyHeads * first.KeyHeadDim
	valueDim := first.NumValueHeads * first.ValueHeadDim
	convDim := 2*keyDim + valueDim
	allocations := qwen35GDNAllocations("gdn-batch-", "-", lanes, lanes, first.Hidden, keyDim, valueDim, first.NumValueHeads, convDim)
	tensors := make([]Tensor, 0, len(allocations))
	buffers := make([]*cudaBuf, 0, len(allocations))
	for _, allocation := range allocations {
		t, b, err := c.qwen35SequenceAllocLocked(allocation.shape, allocation.name)
		if err != nil {
			c.releaseTransientBuffers(buffers)
			return Tensor{}, nil, &Qwen35SequenceError{Stage: allocation.name, Layer: layerIndex, Cause: err}
		}
		tensors = append(tensors, t)
		buffers = append(buffers, b)
	}
	convStates := make([]*C.float, lanes)
	recurrentStates := make([]*C.float, lanes)
	for i, lane := range req.Lanes {
		convStates[i] = c.cf(lane.States[layerIndex].Conv)
		recurrentStates[i] = c.cf(lane.States[layerIndex].Recurrent)
	}
	qkvPtr, qkvScale, qkvQ8 := c.qwen35GDNQ8Args(layer.GDNInQKV)
	zPtr, zScale, zQ8 := c.qwen35GDNQ8Args(layer.GDNInZ)
	bPtr, bScale, bQ8 := c.qwen35GDNQ8Args(layer.GDNInB)
	aPtr, aScale, aQ8 := c.qwen35GDNQ8Args(layer.GDNInA)
	outPtr, outScale, outQ8 := c.qwen35GDNQ8Args(layer.GDNOut)
	status := int(C.fcuda_qwen35_gdn_decode_lanes_f32(c.cf(x), C.int(lanes), qkvPtr, qkvScale, qkvQ8, zPtr, zScale, zQ8, bPtr, bScale, bQ8, aPtr, aScale, aQ8, c.cf(layer.GDNConv), c.cf(layer.GDNALog), c.cf(layer.GDNDTBias), c.cf(layer.GDNNorm), outPtr, outScale, outQ8, &convStates[0], &recurrentStates[0], c.cf(tensors[8]), c.cf(tensors[0]), c.cf(tensors[1]), c.cf(tensors[2]), c.cf(tensors[3]), c.cf(tensors[4]), c.cf(tensors[5]), c.cf(tensors[6]), c.cf(tensors[7]), C.int(first.Hidden), C.int(first.NumKeyHeads), C.int(first.NumValueHeads), C.int(first.KeyHeadDim), C.int(first.ValueHeadDim), C.int(first.ConvKernel), C.float(first.RMSNormEpsilon)))
	if status != 0 {
		for _, b := range buffers {
			atomic.StoreUint32(&b.invalid, 1)
		}
		c.releaseTransientBuffers(buffers)
		kernelErr := &Qwen35GDNKernelError{Stage: qwen35GDNKernelStage(status), Code: status}
		c.faultLatch.ObserveError(kernelErr, "qwen35-batch-gdn")
		return Tensor{}, nil, &Qwen35SequenceError{Stage: "batch-gdn", Layer: layerIndex, Cause: kernelErr}
	}
	atomic.AddUint64(&c.fenceGen, 1)
	output, outputBuffer := tensors[8], buffers[8]
	c.releaseTransientBuffers(buffers[:8])
	return output, outputBuffer, nil
}

// Qwen35GDNLanePanelLaunches is the execution witness that the lane-panel GDN kernel, not a
// per-lane decode loop, carried the batch: one count per linear layer per batched step.
func (*cudaBackend) Qwen35GDNLanePanelLaunches() uint64 {
	return uint64(C.fcuda_qwen35_gdn_lane_panel_launches())
}
