//go:build cuda

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"

int fak_qwen35_embedding_gather_f32(
    const float *dEmbedding, const int *hIDs, int *dIDs, float *dOut,
    int tokens, int hidden);
int fak_qwen35_pointer_is_device(const void *pointer);
int fak_qwen35_split_qg_panel_f32(
    const float *dQG, float *dQ, float *dGate,
    int tokens, int nHeads, int headDim);
int fak_qwen35_partial_rope_panel_f32(
    const float *dQ, const float *dK, float *dQOut, float *dKOut,
    int tokens, int startPos, int nQHeads, int nKHeads, int headDim,
    int rotaryDim, double theta);
int fak_qwen35_kv_write_f32(
    float *dDst, const float *dSrc, int offset, int n);
int fak_qwen35_causal_attention_panel_f32(
    const float *dQ, const float *dK, const float *dV, float *dOut,
    int tokens, int prefix, int nH, int nKV, int hd, float scale);
int fak_qwen35_sequence_sync(void);
*/
import "C"

import (
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"
)

const (
	qwen38CausalAttentionPanelTokens     = 2
	qwen38CausalAttentionPanelPrefix     = 1
	qwen38CausalAttentionPanelHeads      = 24
	qwen38CausalAttentionPanelKVHeads    = 4
	qwen38CausalAttentionPanelHeadDim    = 256
	qwen35CausalAttentionPanelMaxHeadDim = qwen38CausalAttentionPanelHeadDim
)

func validateQwen35CausalAttentionPanelKVGeometry(tokens, prefix, nKV, headDim int) error {
	if tokens != qwen38CausalAttentionPanelTokens ||
		prefix != qwen38CausalAttentionPanelPrefix ||
		nKV != qwen38CausalAttentionPanelKVHeads ||
		headDim != qwen38CausalAttentionPanelHeadDim {
		return &Qwen35SequenceError{
			Stage: "causal-attention-geometry", Layer: -1,
			Reason: fmt.Sprintf(
				"prompt attention supports only Qwen3.8 source-spine geometry tokens=%d prefix=%d KV_heads=%d head_dim=%d",
				qwen38CausalAttentionPanelTokens, qwen38CausalAttentionPanelPrefix,
				qwen38CausalAttentionPanelKVHeads, qwen38CausalAttentionPanelHeadDim,
			),
		}
	}
	if int64(tokens) > qwen35GDNMaxCInt || int64(prefix) > qwen35GDNMaxCInt-int64(tokens) {
		return &Qwen35SequenceError{Stage: "causal-attention-geometry", Layer: -1, Reason: "causal position range overflows the CUDA int ABI"}
	}
	positions := int64(prefix) + int64(tokens)
	if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, positions, int64(nKV), int64(headDim)); !ok {
		return &Qwen35SequenceError{Stage: "causal-attention-geometry", Layer: -1, Reason: "causal KV panel element count overflows the CUDA int ABI"}
	}
	return nil
}

func validateQwen35CausalAttentionPanelGeometry(tokens, prefix, nH, nKV, headDim int) error {
	if err := validateQwen35CausalAttentionPanelKVGeometry(tokens, prefix, nKV, headDim); err != nil {
		return err
	}
	if nH != qwen38CausalAttentionPanelHeads {
		return &Qwen35SequenceError{
			Stage: "causal-attention-geometry", Layer: -1,
			Reason: fmt.Sprintf("prompt attention supports only %d Qwen3.8 query heads", qwen38CausalAttentionPanelHeads),
		}
	}
	if int64(nH) > qwen35GDNMaxCInt {
		return &Qwen35SequenceError{Stage: "causal-attention-geometry", Layer: -1, Reason: "query head count overflows the CUDA int ABI"}
	}
	if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(tokens), int64(nH)); !ok {
		return &Qwen35SequenceError{Stage: "causal-attention-geometry", Layer: -1, Reason: "causal attention launch grid overflows the CUDA int ABI"}
	}
	if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(tokens), int64(nH), int64(headDim)); !ok {
		return &Qwen35SequenceError{Stage: "causal-attention-geometry", Layer: -1, Reason: "causal query/output panel element count overflows the CUDA int ABI"}
	}
	return nil
}

func validateQwen35CausalAttentionPanelScale(scale float32) error {
	if !(scale > 0) || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return &Qwen35SequenceError{Stage: "causal-attention-geometry", Layer: -1, Reason: "attention scale must be finite and positive"}
	}
	return nil
}

func (c *cudaBackend) qwen35SequenceAllocLocked(shape []int, site string) (Tensor, *cudaBuf, error) {
	t, b, err := c.devTrDeviceOnly(shape, F32, "qwen35-sequence-"+site)
	if err != nil {
		return Tensor{}, nil, &Qwen35SequenceError{Stage: site, Layer: -1, Cause: err}
	}
	return t, b, nil
}

func (c *cudaBackend) qwen35SequenceMatMulLocked(w, x Tensor, tokens int, site string) (Tensor, *cudaBuf, error) {
	y, b, err := c.qwen35SequenceAllocLocked([]int{tokens, w.Shape[0]}, site)
	if err != nil {
		return Tensor{}, nil, err
	}
	out, in := w.Shape[0], w.Shape[1]
	wb := w.buf.(*cudaBuf)
	switch w.Dtype {
	case F32:
		C.fcuda_matmul_f32((*C.float)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens))
	case F16:
		C.fcuda_matmul_f16(wb.ptr, c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens), colMajorFlag(w))
	case Q8_0:
		C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens), C.int(w.Quant.Block))
	case Q4_K:
		C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens))
	case Q5_K:
		C.fcuda_q5k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens))
	case Q6_K:
		C.fcuda_q6k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens))
	case Q2_0:
		C.fcuda_q2_0_matmul_f32((*C.uint8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens), C.int(w.Quant.Block))
	default:
		c.releaseTransientBuffers([]*cudaBuf{b})
		return Tensor{}, nil, &Qwen35SequenceError{Stage: site, Layer: -1, Reason: "unsupported resident matrix dtype " + w.Dtype.String()}
	}
	return y, b, nil
}

func (c *cudaBackend) qwen35SequenceRMSNormLocked(x, weight Tensor, tokens int, eps float32, site string) (Tensor, *cudaBuf, error) {
	y, b, err := c.qwen35SequenceAllocLocked(append([]int(nil), x.Shape...), site)
	if err != nil {
		return Tensor{}, nil, err
	}
	C.fcuda_rmsnorm_f32(c.cf(x), c.cf(weight), c.cf(y), C.int(tokens), C.int(weight.Numel()), C.float(eps))
	return y, b, nil
}

func (c *cudaBackend) qwen35SequenceEmbeddingLocked(embedding Tensor, ids []int, hidden int) (Tensor, *cudaBuf, error) {
	tokens := len(ids)
	out, outBuf, err := c.qwen35SequenceAllocLocked([]int{tokens, hidden}, "embedding-gather-output")
	if err != nil {
		return Tensor{}, nil, err
	}
	idBytes, ok := qwen35GDNShapeBytes([]int{tokens}, 4)
	if !ok {
		c.releaseTransientBuffers([]*cudaBuf{outBuf})
		return Tensor{}, nil, &Qwen35SequenceError{Stage: "embedding-gather", Layer: -1, Reason: "token-id byte size overflow"}
	}
	idBuf, allocErr := c.dallocDeviceOnlyClass(idBytes, MemoryScratchpad, "qwen35-sequence-token-ids")
	if allocErr != nil {
		c.releaseTransientBuffers([]*cudaBuf{outBuf})
		return Tensor{}, nil, &Qwen35SequenceError{Stage: "embedding-gather", Layer: -1, Cause: allocErr}
	}
	deviceIDs := make([]C.int, tokens)
	for i, id := range ids {
		deviceIDs[i] = C.int(id)
	}
	status := int(C.fak_qwen35_embedding_gather_f32(c.cf(embedding), &deviceIDs[0], (*C.int)(idBuf.ptr), c.cf(out), C.int(tokens), C.int(hidden)))
	C.fcuda_free(idBuf.ptr)
	if status != 0 {
		C.fak_qwen35_sequence_sync()
		atomic.StoreUint32(&outBuf.invalid, 1)
		c.releaseTransientBuffers([]*cudaBuf{outBuf})
		return Tensor{}, nil, &Qwen35SequenceError{Stage: "embedding-gather", Layer: -1, Reason: fmt.Sprintf("CUDA status %d", status)}
	}
	return out, outBuf, nil
}

func (c *cudaBackend) qwen35SequenceSplitQGLocked(qg Tensor, tokens, heads, headDim int) (Tensor, *cudaBuf, Tensor, *cudaBuf, error) {
	q, qb, err := c.qwen35SequenceAllocLocked([]int{tokens, heads * headDim}, "attention-query-split")
	if err != nil {
		return Tensor{}, nil, Tensor{}, nil, err
	}
	gate, gb, err := c.qwen35SequenceAllocLocked([]int{tokens, heads * headDim}, "attention-gate-split")
	if err != nil {
		c.releaseTransientBuffers([]*cudaBuf{qb})
		return Tensor{}, nil, Tensor{}, nil, err
	}
	status := int(C.fak_qwen35_split_qg_panel_f32(c.cf(qg), c.cf(q), c.cf(gate), C.int(tokens), C.int(heads), C.int(headDim)))
	if status != 0 {
		C.fak_qwen35_sequence_sync()
		c.releaseTransientBuffers([]*cudaBuf{qb, gb})
		return Tensor{}, nil, Tensor{}, nil, &Qwen35SequenceError{Stage: "attention-query-split", Layer: -1, Reason: fmt.Sprintf("CUDA status %d", status)}
	}
	return q, qb, gate, gb, nil
}

func (c *cudaBackend) qwen35SequenceRoPELocked(q, k Tensor, tokens, start, qHeads, kvHeads, headDim, rotary int, theta float64) (Tensor, *cudaBuf, Tensor, *cudaBuf, error) {
	if err := validateQwen35CausalAttentionPanelGeometry(tokens, start, qHeads, kvHeads, headDim); err != nil {
		return Tensor{}, nil, Tensor{}, nil, err
	}
	qOut, qb, err := c.qwen35SequenceAllocLocked([]int{tokens, qHeads * headDim}, "attention-query-rope")
	if err != nil {
		return Tensor{}, nil, Tensor{}, nil, err
	}
	kOut, kb, err := c.qwen35SequenceAllocLocked([]int{tokens, kvHeads * headDim}, "attention-key-rope")
	if err != nil {
		c.releaseTransientBuffers([]*cudaBuf{qb})
		return Tensor{}, nil, Tensor{}, nil, err
	}
	status := int(C.fak_qwen35_partial_rope_panel_f32(c.cf(q), c.cf(k), c.cf(qOut), c.cf(kOut), C.int(tokens), C.int(start), C.int(qHeads), C.int(kvHeads), C.int(headDim), C.int(rotary), C.double(theta)))
	if status != 0 {
		C.fak_qwen35_sequence_sync()
		c.releaseTransientBuffers([]*cudaBuf{qb, kb})
		return Tensor{}, nil, Tensor{}, nil, &Qwen35SequenceError{Stage: "attention-rope", Layer: -1, Reason: fmt.Sprintf("CUDA status %d", status)}
	}
	return qOut, qb, kOut, kb, nil
}

type qwen35KVReservation struct {
	dst  *dslice
	old  unsafe.Pointer
	next unsafe.Pointer
	cap  int
}

func (c *cudaBackend) qwen35SequenceReserveKVLocked(kv *cudaKV, compactLayers, needed int) error {
	exactNeeded := (qwen38CausalAttentionPanelPrefix + qwen38CausalAttentionPanelTokens) *
		qwen38CausalAttentionPanelKVHeads * qwen38CausalAttentionPanelHeadDim
	if kv == nil || kv.cfg.NumKVHeads != qwen38CausalAttentionPanelKVHeads ||
		kv.cfg.HeadDim != qwen38CausalAttentionPanelHeadDim || needed != exactNeeded {
		return &Qwen35SequenceError{
			Stage:  "causal-attention-geometry",
			Layer:  -1,
			Reason: "KV reservation exceeds the bounded CUDA prompt-attention geometry",
		}
	}
	if graphEnabled && kv.maxPos > 0 && needed/kv.stride() > kv.maxPos {
		return &Qwen35SequenceError{Stage: "kv-reserve", Layer: -1, Reason: fmt.Sprintf("fixed CUDA graph KV capacity %d is smaller than requested position %d", kv.maxPos, needed/kv.stride())}
	}
	reservations := make([]qwen35KVReservation, 0, compactLayers*3)
	cleanup := func() {
		for _, reservation := range reservations {
			if reservation.next != nil {
				C.fcuda_free(reservation.next)
			}
		}
	}
	for layer := 0; layer < compactLayers; layer++ {
		for _, dst := range []*dslice{&kv.Kraw[layer], &kv.K[layer], &kv.V[layer]} {
			if dst.ptr != nil && C.fak_qwen35_pointer_is_device(dst.ptr) == 0 {
				cleanup()
				return &Qwen35SequenceError{Stage: "kv-reserve", Layer: layer, Reason: "existing KV storage is not strict device memory"}
			}
			if dst.cap >= needed {
				continue
			}
			bytes, ok := qwen35GDNShapeBytes([]int{needed}, F32.Bytes())
			if !ok {
				cleanup()
				return &Qwen35SequenceError{Stage: "kv-reserve", Layer: layer, Reason: "KV capacity byte size overflow"}
			}
			buf, err := c.dallocDeviceOnlyClass(bytes, MemoryKVCache, "qwen35-sequence-kv-reserve")
			if err != nil {
				cleanup()
				return &Qwen35SequenceError{Stage: "kv-reserve", Layer: layer, Cause: err}
			}
			reservations = append(reservations, qwen35KVReservation{dst: dst, old: dst.ptr, next: buf.ptr, cap: needed})
		}
	}
	for _, reservation := range reservations {
		if reservation.dst.len > 0 {
			C.fcuda_d2d(reservation.next, reservation.old, C.size_t(reservation.dst.len*F32.Bytes()))
		}
		reservation.dst.ptr = reservation.next
		reservation.dst.cap = reservation.cap
		if reservation.old != nil {
			C.fcuda_free(reservation.old)
		}
	}
	return nil
}

func (c *cudaBackend) qwen35SequenceAppendKVLocked(kv *cudaKV, layer int, kRaw, kRoPE, value Tensor, start, tokens int) error {
	if kv == nil || layer < 0 || layer >= len(kv.Kraw) || layer >= len(kv.K) || layer >= len(kv.V) {
		return &Qwen35SequenceError{Stage: "kv-append", Layer: layer, Reason: "KV cache does not contain the requested compact attention layer"}
	}
	if err := validateQwen35CausalAttentionPanelKVGeometry(tokens, start, kv.cfg.NumKVHeads, kv.cfg.HeadDim); err != nil {
		return err
	}
	width := kv.stride()
	want := start * width
	items := []struct {
		name string
		dst  *dslice
		src  Tensor
	}{{"raw key", &kv.Kraw[layer], kRaw}, {"key", &kv.K[layer], kRoPE}, {"value", &kv.V[layer], value}}
	for _, item := range items {
		if item.dst.len != want || item.dst.cap < want+tokens*width {
			return &Qwen35SequenceError{Stage: "kv-append", Layer: layer, Reason: fmt.Sprintf("%s cache len/cap=%d/%d, want len=%d cap>=%d", item.name, item.dst.len, item.dst.cap, want, want+tokens*width)}
		}
		if err := c.validateQwen35SequenceTensor(item.name, item.src, false, tokens, width); err != nil {
			return &Qwen35SequenceError{Stage: "kv-append", Layer: layer, Cause: err}
		}
	}
	n := tokens * width
	for _, item := range items {
		status := int(C.fak_qwen35_kv_write_f32((*C.float)(item.dst.ptr), c.cf(item.src), C.int(item.dst.len), C.int(n)))
		if status != 0 {
			C.fak_qwen35_sequence_sync()
			return &Qwen35SequenceError{Stage: "kv-append", Layer: layer, Reason: fmt.Sprintf("%s CUDA status %d", item.name, status)}
		}
	}
	for _, item := range items {
		item.dst.len += n
	}
	if layer == 0 {
		for token := 0; token < tokens; token++ {
			kv.pos = append(kv.pos, start+token)
		}
	}
	return nil
}

func (c *cudaBackend) qwen35SequenceAttentionLocked(q Tensor, kv *cudaKV, layer, tokens, prefix, heads int, scale float32) (Tensor, *cudaBuf, error) {
	if err := validateQwen35CausalAttentionPanelGeometry(tokens, prefix, heads, kv.cfg.NumKVHeads, kv.cfg.HeadDim); err != nil {
		return Tensor{}, nil, err
	}
	if err := validateQwen35CausalAttentionPanelScale(scale); err != nil {
		return Tensor{}, nil, err
	}
	out, b, err := c.qwen35SequenceAllocLocked([]int{tokens, heads * kv.cfg.HeadDim}, "causal-attention")
	if err != nil {
		return Tensor{}, nil, err
	}
	status := int(C.fak_qwen35_causal_attention_panel_f32(c.cf(q), (*C.float)(kv.K[layer].ptr), (*C.float)(kv.V[layer].ptr), c.cf(out), C.int(tokens), C.int(prefix), C.int(heads), C.int(kv.cfg.NumKVHeads), C.int(kv.cfg.HeadDim), C.float(scale)))
	if status != 0 {
		C.fak_qwen35_sequence_sync()
		c.releaseTransientBuffers([]*cudaBuf{b})
		return Tensor{}, nil, &Qwen35SequenceError{Stage: "causal-attention", Layer: layer, Reason: fmt.Sprintf("CUDA status %d", status)}
	}
	return out, b, nil
}

// qwen35CausalAttentionPanelIntoForTest exposes the supplied-output ABI used by the
// deterministic sentinel witness. Production allocates its output internally through
// qwen35SequenceAttentionLocked. This test seam validates every resident tensor before
// launch and fences the stream so asynchronous execution faults are returned to the caller.
func (c *cudaBackend) qwen35CausalAttentionPanelIntoForTest(q, k, v, out Tensor, tokens, prefix, heads, kvHeads, headDim int, scale float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if err := validateQwen35CausalAttentionPanelGeometry(tokens, prefix, heads, kvHeads, headDim); err != nil {
		return err
	}
	if err := validateQwen35CausalAttentionPanelScale(scale); err != nil {
		return err
	}
	for _, tensor := range []struct {
		name  string
		value Tensor
		shape []int
	}{
		{"query", q, []int{tokens, heads * headDim}},
		{"key", k, []int{prefix + tokens, kvHeads * headDim}},
		{"value", v, []int{prefix + tokens, kvHeads * headDim}},
		{"output", out, []int{tokens, heads * headDim}},
	} {
		if err := c.validateQwen35SequenceTensor(tensor.name, tensor.value, false, tensor.shape...); err != nil {
			return err
		}
	}
	status := int(C.fak_qwen35_causal_attention_panel_f32(c.cf(q), c.cf(k), c.cf(v), c.cf(out), C.int(tokens), C.int(prefix), C.int(heads), C.int(kvHeads), C.int(headDim), C.float(scale)))
	if status != 0 {
		C.fak_qwen35_sequence_sync()
		return &Qwen35SequenceError{Stage: "causal-attention", Layer: -1, Reason: fmt.Sprintf("CUDA status %d", status)}
	}
	if err := c.qwen35SequenceSyncLocked(); err != nil {
		atomic.StoreUint32(&out.buf.(*cudaBuf).invalid, 1)
		return err
	}
	return nil
}

func (c *cudaBackend) qwen35SequenceGDNLocked(x Tensor, layer Qwen35SequenceLayer, state Qwen35SequenceState, req Qwen35SequencePrefillRequest, tokens, layerIndex int) (Tensor, *cudaBuf, error) {
	keyDim := req.NumKeyHeads * req.KeyHeadDim
	valueDim := req.NumValueHeads * req.ValueHeadDim
	convDim := 2*keyDim + valueDim
	allocations := qwen35GDNAllocations("gdn-", "-", tokens, tokens, req.Hidden, keyDim, valueDim, req.NumValueHeads, convDim)
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
	qkvPtr, qkvScale, qkvQ8 := c.qwen35GDNQ8Args(layer.GDNInQKV)
	zPtr, zScale, zQ8 := c.qwen35GDNQ8Args(layer.GDNInZ)
	bPtr, bScale, bQ8 := c.qwen35GDNQ8Args(layer.GDNInB)
	aPtr, aScale, aQ8 := c.qwen35GDNQ8Args(layer.GDNInA)
	outPtr, outScale, outQ8 := c.qwen35GDNQ8Args(layer.GDNOut)
	status := int(C.fcuda_qwen35_gdn_sequence_f32(c.cf(x), C.int(tokens), qkvPtr, qkvScale, qkvQ8, zPtr, zScale, zQ8, bPtr, bScale, bQ8, aPtr, aScale, aQ8, c.cf(layer.GDNConv), c.cf(layer.GDNALog), c.cf(layer.GDNDTBias), c.cf(layer.GDNNorm), outPtr, outScale, outQ8, c.cf(state.Conv), c.cf(state.Recurrent), c.cf(tensors[8]), c.cf(tensors[0]), c.cf(tensors[1]), c.cf(tensors[2]), c.cf(tensors[3]), c.cf(tensors[4]), c.cf(tensors[5]), c.cf(tensors[6]), c.cf(tensors[7]), C.int(req.Hidden), C.int(req.NumKeyHeads), C.int(req.NumValueHeads), C.int(req.KeyHeadDim), C.int(req.ValueHeadDim), C.int(req.ConvKernel), C.float(req.RMSNormEpsilon)))
	if status != 0 {
		kernelErr := c.failQwen35GDN(buffers, state.Conv, state.Recurrent, status, "qwen35-sequence-gdn")
		return Tensor{}, nil, &Qwen35SequenceError{Stage: "gdn", Layer: layerIndex, Cause: kernelErr}
	}
	atomic.AddUint64(&c.fenceGen, 1)
	output, outputBuffer := tensors[8], buffers[8]
	c.releaseTransientBuffers(buffers[:8])
	return output, outputBuffer, nil
}

func (c *cudaBackend) qwen35SequenceSyncLocked() error {
	status := int(C.fak_qwen35_sequence_sync())
	if status != 0 {
		return &Qwen35SequenceError{Stage: "final-fence", Layer: -1, Reason: fmt.Sprintf("CUDA status %d", status)}
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return nil
}
