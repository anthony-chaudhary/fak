//go:build cuda

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

func (*cudaBackend) Qwen35SequencePrefillPath() string { return Qwen35SequencePrefillPath }

func qwen35SequenceFailure(stage string, layer int, reason string) error {
	return &Qwen35SequenceError{Stage: stage, Layer: layer, Reason: reason}
}

func validateQwen35SequenceGeometry(req Qwen35SequencePrefillRequest) error {
	if req.Path != Qwen35SequencePrefillPath {
		return qwen35SequenceFailure("request", -1, "capability path mismatch")
	}
	if len(req.TokenIDs) == 0 || req.StartPos < 0 {
		return qwen35SequenceFailure("request", -1, "token panel must be non-empty and start position non-negative")
	}
	if req.Hidden <= 0 || req.Intermediate <= 0 || req.NumHeads <= 0 || req.NumKVHeads <= 0 || req.HeadDim <= 0 || req.RotaryDim <= 0 {
		return qwen35SequenceFailure("geometry", -1, "hidden, FFN, attention head counts, and dimensions must be positive")
	}
	if req.NumHeads%req.NumKVHeads != 0 || req.RotaryDim > req.HeadDim || req.RotaryDim%2 != 0 {
		return qwen35SequenceFailure("geometry", -1, "attention requires grouped heads and an even partial rotary dimension")
	}
	// Use the launcher's bounded panel contract during request preflight. This
	// runs before tensor validation, KV reservation, allocation, or transfer, so
	// a dimension the CUDA source cannot execute never reaches an effect.
	if err := validateQwen35CausalAttentionPanelGeometry(len(req.TokenIDs), req.StartPos, req.NumHeads, req.NumKVHeads, req.HeadDim); err != nil {
		return err
	}
	if req.NumKeyHeads <= 0 || req.NumValueHeads <= 0 || req.KeyHeadDim <= 0 || req.ValueHeadDim <= 0 || req.ConvKernel < 1 || req.NumValueHeads%req.NumKeyHeads != 0 {
		return qwen35SequenceFailure("geometry", -1, "GDN heads/dimensions must be positive, value heads divisible by key heads, and convolution non-empty")
	}
	if req.KeyHeadDim > 1024 || req.ValueHeadDim > 1024 {
		return qwen35SequenceFailure("geometry", -1, "GDN state dimensions must fit one CUDA block")
	}
	if !(req.RMSNormEpsilon > 0) || math.IsNaN(float64(req.RMSNormEpsilon)) || math.IsInf(float64(req.RMSNormEpsilon), 0) {
		return qwen35SequenceFailure("geometry", -1, "RMS epsilon must be finite and positive")
	}
	tokens := int64(len(req.TokenIDs))
	for _, scalar := range []int{req.Hidden, req.Intermediate, req.NumHeads, req.NumKVHeads, req.HeadDim, req.RotaryDim, req.NumKeyHeads, req.NumValueHeads, req.KeyHeadDim, req.ValueHeadDim, req.ConvKernel, len(req.TokenIDs)} {
		if int64(scalar) > qwen35GDNMaxCInt {
			return qwen35SequenceFailure("geometry", -1, "a scalar dimension overflows the CUDA int ABI")
		}
	}
	for _, product := range [][]int64{
		{tokens, int64(req.Hidden)},
		{tokens, int64(req.Intermediate)},
		{tokens, int64(req.NumHeads), int64(req.HeadDim)},
		{tokens, int64(req.NumKVHeads), int64(req.HeadDim)},
	} {
		if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, product...); !ok {
			return qwen35SequenceFailure("geometry", -1, "a panel element count overflows the CUDA int ABI")
		}
	}
	if len(req.Layers) == 0 || len(req.Layers)%4 != 0 || len(req.States) != len(req.Layers) || len(req.RoPEThetaForLayer) != len(req.Layers) {
		return qwen35SequenceFailure("geometry", -1, "layers must be complete four-layer hybrid groups with one state and RoPE theta per main layer")
	}
	for layer := range req.Layers {
		wantLinear := (layer+1)%4 != 0
		if req.Layers[layer].Linear != wantLinear {
			return qwen35SequenceFailure("geometry", layer, "dense Qwen3.5 requires three GDN layers followed by one full-attention layer")
		}
		theta := req.RoPEThetaForLayer[layer]
		if !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
			return qwen35SequenceFailure("geometry", layer, "RoPE theta must be finite and positive")
		}
	}
	if len(req.Layers) == Qwen35DenseMainLayers {
		if req.Hidden != Qwen35DenseHidden || req.Intermediate != Qwen35DenseIntermediate ||
			req.NumHeads != Qwen35DenseQueryHeads || req.NumKVHeads != Qwen35DenseKVHeads ||
			req.HeadDim != Qwen35DenseHeadDim || req.RotaryDim != Qwen35DenseHeadDim/4 ||
			req.NumKeyHeads != Qwen35DenseGDNGroups || req.NumValueHeads != Qwen35DenseGDNRank ||
			req.KeyHeadDim != Qwen35DenseGDNState || req.ValueHeadDim != Qwen35DenseGDNState ||
			req.NumValueHeads*req.ValueHeadDim != Qwen35DenseGDNInner || req.ConvKernel != Qwen35DenseGDNConv {
			return qwen35SequenceFailure("production-geometry", -1, "64-layer dense checkpoint does not match Qwen3.8 27B text geometry; the trailing MTP metadata layer must not be executed")
		}
	}
	return nil
}

func qwen35SequenceSameShape(t Tensor, want ...int) bool {
	return qwen35GDNSameShape(t.Shape, want)
}

func qwen35SequenceMatrixBytes(t Tensor) (int, bool) {
	n := t.Numel()
	if n < 0 {
		return 0, false
	}
	switch t.Dtype {
	case F32, F16, Q8_0:
		return qwen35GDNShapeBytes(t.Shape, t.Dtype.Bytes())
	case Q4_K:
		if n%256 != 0 {
			return 0, false
		}
		return n / 256 * 144, true
	case Q5_K:
		if n%256 != 0 {
			return 0, false
		}
		return n / 256 * 176, true
	case Q6_K:
		if n%256 != 0 {
			return 0, false
		}
		return n / 256 * 210, true
	case Q2_0:
		if n%4 != 0 {
			return 0, false
		}
		return n / 4, true
	default:
		return 0, false
	}
}

func (c *cudaBackend) validateQwen35SequenceTensor(name string, t Tensor, matrix bool, shape ...int) error {
	if !qwen35SequenceSameShape(t, shape...) {
		return qwen35SequenceFailure("tensor-preflight", -1, fmt.Sprintf("%s shape %v, want %v", name, t.Shape, shape))
	}
	if t.Backend() != c || t.Layout != RowMajor {
		return qwen35SequenceFailure("tensor-preflight", -1, name+" must be row-major and owned by the executing CUDA backend")
	}
	buf, ok := t.buf.(*cudaBuf)
	if !ok || buf == nil || buf.ptr == nil || buf.device != 0 || buf.managed {
		return qwen35SequenceFailure("tensor-preflight", -1, name+" must have a live default-device allocation; managed memory is refused")
	}
	if err := buf.invalidStateError(name); err != nil {
		return &Qwen35SequenceError{Stage: "tensor-preflight", Layer: -1, Cause: err}
	}
	if !matrix {
		if t.Dtype != F32 || buf.n < t.Numel()*F32.Bytes() {
			return qwen35SequenceFailure("tensor-preflight", -1, name+" must be capacity-valid resident f32")
		}
		return nil
	}
	bytes, valid := qwen35SequenceMatrixBytes(t)
	if !valid || buf.n < bytes {
		return qwen35SequenceFailure("tensor-preflight", -1, name+" has unsupported dtype/packing or insufficient resident capacity")
	}
	if t.Dtype == Q8_0 && (t.Quant == nil || t.Quant.Block != q8DeviceBlock || buf.scales == nil) {
		return qwen35SequenceFailure("tensor-preflight", -1, name+" Q8_0 matrix is missing block-32 resident scales")
	}
	if t.Dtype == Q2_0 && (t.Quant == nil || t.Quant.Block <= 0 || buf.scales == nil) {
		return qwen35SequenceFailure("tensor-preflight", -1, name+" Q2_0 matrix is missing resident scales")
	}
	return nil
}

func (c *cudaBackend) validateQwen35SequenceRequestLocked(req Qwen35SequencePrefillRequest) (*cudaKV, error) {
	if err := validateQwen35SequenceGeometry(req); err != nil {
		return nil, err
	}
	if int64(req.StartPos) > qwen35GDNMaxCInt-int64(len(req.TokenIDs)) {
		return nil, qwen35SequenceFailure("geometry", -1, "position range overflows the CUDA int ABI")
	}
	if err := c.validateQwen35SequenceTensor("token_embedding", req.TokenEmbedding, false, req.TokenEmbedding.Shape...); err != nil {
		return nil, err
	}
	if len(req.TokenEmbedding.Shape) != 2 || req.TokenEmbedding.Shape[1] != req.Hidden {
		return nil, qwen35SequenceFailure("tensor-preflight", -1, fmt.Sprintf("token_embedding shape %v, want [vocab,%d]", req.TokenEmbedding.Shape, req.Hidden))
	}
	vocab := req.TokenEmbedding.Shape[0]
	for _, id := range req.TokenIDs {
		if id < 0 || id >= vocab {
			return nil, qwen35SequenceFailure("embedding-gather", -1, fmt.Sprintf("token id %d is outside vocabulary [0,%d)", id, vocab))
		}
	}
	if err := c.validateQwen35SequenceTensor("output_norm", req.OutputNorm, false, req.Hidden); err != nil {
		return nil, err
	}
	if err := c.validateQwen35SequenceTensor("output", req.Output, true, vocab, req.Hidden); err != nil {
		return nil, err
	}
	kv, ok := req.KV.(*cudaKV)
	if !ok || kv == nil || kv.be != c || kv.Len() != req.StartPos {
		return nil, qwen35SequenceFailure("kv-preflight", -1, "KV must belong to this CUDA backend and match start_pos")
	}
	if kv.cfg.NumKVHeads != req.NumKVHeads || kv.cfg.HeadDim != req.HeadDim || len(kv.K) < len(req.Layers)/4 || len(kv.Kraw) < len(req.Layers)/4 || len(kv.V) < len(req.Layers)/4 {
		return nil, qwen35SequenceFailure("kv-preflight", -1, "KV geometry or compact layer capacity does not match the request")
	}
	for pos, absolute := range kv.pos {
		if absolute != pos {
			return nil, qwen35SequenceFailure("kv-preflight", -1, "Qwen recurrent sequence path requires an unevicted contiguous position prefix")
		}
	}
	keyDim := req.NumKeyHeads * req.KeyHeadDim
	valueDim := req.NumValueHeads * req.ValueHeadDim
	convDim := 2*keyDim + valueDim
	compactAttention := 0
	statePointers := make(map[unsafe.Pointer]int)
	for index, layer := range req.Layers {
		check := func(name string, tensor Tensor, matrix bool, shape ...int) error {
			if err := c.validateQwen35SequenceTensor(name, tensor, matrix, shape...); err != nil {
				return &Qwen35SequenceError{Stage: "tensor-preflight", Layer: index, Cause: err}
			}
			return nil
		}
		if err := check("input_norm", layer.InputNorm, false, req.Hidden); err != nil {
			return nil, err
		}
		if err := check("post_norm", layer.PostNorm, false, req.Hidden); err != nil {
			return nil, err
		}
		if err := check("mlp_gate", layer.Gate, true, req.Intermediate, req.Hidden); err != nil {
			return nil, err
		}
		if err := check("mlp_up", layer.Up, true, req.Intermediate, req.Hidden); err != nil {
			return nil, err
		}
		if err := check("mlp_down", layer.Down, true, req.Hidden, req.Intermediate); err != nil {
			return nil, err
		}
		if layer.Linear {
			if err := check("gdn_in_qkv", layer.GDNInQKV, true, convDim, req.Hidden); err != nil {
				return nil, err
			}
			if err := check("gdn_in_z", layer.GDNInZ, true, valueDim, req.Hidden); err != nil {
				return nil, err
			}
			if err := check("gdn_in_b", layer.GDNInB, true, req.NumValueHeads, req.Hidden); err != nil {
				return nil, err
			}
			if err := check("gdn_in_a", layer.GDNInA, true, req.NumValueHeads, req.Hidden); err != nil {
				return nil, err
			}
			for name, tensor := range map[string]Tensor{"gdn_in_qkv": layer.GDNInQKV, "gdn_in_z": layer.GDNInZ, "gdn_in_b": layer.GDNInB, "gdn_in_a": layer.GDNInA, "gdn_out": layer.GDNOut} {
				if tensor.Dtype != F32 && tensor.Dtype != Q8_0 {
					return nil, qwen35SequenceFailure("tensor-preflight", index, name+" must be resident f32 or q8_0 for the existing GDN sequence primitive")
				}
			}
			convShapeOK := qwen35SequenceSameShape(layer.GDNConv, convDim*req.ConvKernel) || qwen35SequenceSameShape(layer.GDNConv, convDim, req.ConvKernel) || qwen35SequenceSameShape(layer.GDNConv, convDim, 1, req.ConvKernel)
			if !convShapeOK {
				return nil, qwen35SequenceFailure("tensor-preflight", index, fmt.Sprintf("gdn_conv shape %v is not a depthwise [%d x %d] kernel", layer.GDNConv.Shape, convDim, req.ConvKernel))
			}
			if err := check("gdn_conv", layer.GDNConv, false, layer.GDNConv.Shape...); err != nil {
				return nil, err
			}
			if err := check("gdn_A_log", layer.GDNALog, false, req.NumValueHeads); err != nil {
				return nil, err
			}
			if err := check("gdn_dt_bias", layer.GDNDTBias, false, req.NumValueHeads); err != nil {
				return nil, err
			}
			if err := check("gdn_norm", layer.GDNNorm, false, req.ValueHeadDim); err != nil {
				return nil, err
			}
			if err := check("gdn_out", layer.GDNOut, true, req.Hidden, valueDim); err != nil {
				return nil, err
			}
			state := req.States[index]
			if err := check("gdn_conv_state", state.Conv, false, req.ConvKernel-1, convDim); err != nil {
				return nil, err
			}
			if err := check("gdn_recurrent_state", state.Recurrent, false, req.NumValueHeads, req.KeyHeadDim, req.ValueHeadDim); err != nil {
				return nil, err
			}
			if state.Conv.buf.(*cudaBuf).class != MemoryKVCache || state.Recurrent.buf.(*cudaBuf).class != MemoryKVCache || state.Conv.buf.(*cudaBuf).ptr == state.Recurrent.buf.(*cudaBuf).ptr {
				return nil, qwen35SequenceFailure("state-preflight", index, "GDN state must use distinct durable KV-cache allocations")
			}
			for _, stateBuf := range []*cudaBuf{state.Conv.buf.(*cudaBuf), state.Recurrent.buf.(*cudaBuf)} {
				if prior, aliased := statePointers[stateBuf.ptr]; aliased {
					return nil, qwen35SequenceFailure("state-preflight", index, fmt.Sprintf("GDN state aliases layer %d persistent state", prior))
				}
				statePointers[stateBuf.ptr] = index
			}
			for name, operand := range map[string]Tensor{
				"input_norm": layer.InputNorm, "post_norm": layer.PostNorm,
				"gdn_in_qkv": layer.GDNInQKV, "gdn_in_z": layer.GDNInZ,
				"gdn_in_b": layer.GDNInB, "gdn_in_a": layer.GDNInA,
				"gdn_conv": layer.GDNConv, "gdn_A_log": layer.GDNALog,
				"gdn_dt_bias": layer.GDNDTBias, "gdn_norm": layer.GDNNorm,
				"gdn_out": layer.GDNOut, "mlp_gate": layer.Gate,
				"mlp_up": layer.Up, "mlp_down": layer.Down,
			} {
				operandPointer := operand.buf.(*cudaBuf).ptr
				if operandPointer == state.Conv.buf.(*cudaBuf).ptr || operandPointer == state.Recurrent.buf.(*cudaBuf).ptr {
					return nil, qwen35SequenceFailure("state-preflight", index, name+" aliases mutable GDN state")
				}
			}
		} else {
			qWidth := req.NumHeads * req.HeadDim
			kvWidth := req.NumKVHeads * req.HeadDim
			if err := check("attention_qg", layer.Q, true, 2*qWidth, req.Hidden); err != nil {
				return nil, err
			}
			if err := check("attention_k", layer.K, true, kvWidth, req.Hidden); err != nil {
				return nil, err
			}
			if err := check("attention_v", layer.V, true, kvWidth, req.Hidden); err != nil {
				return nil, err
			}
			if err := check("attention_o", layer.O, true, req.Hidden, qWidth); err != nil {
				return nil, err
			}
			if layer.QNorm.Buf() != nil {
				if err := check("attention_q_norm", layer.QNorm, false, req.HeadDim); err != nil {
					return nil, err
				}
			}
			if layer.KNorm.Buf() != nil {
				if err := check("attention_k_norm", layer.KNorm, false, req.HeadDim); err != nil {
					return nil, err
				}
			}
			if (layer.QNorm.Buf() == nil) != (layer.KNorm.Buf() == nil) {
				return nil, qwen35SequenceFailure("tensor-preflight", index, "Q/K normalization weights must be both present or both absent")
			}
			width := kv.stride()
			for _, cache := range []dslice{kv.Kraw[compactAttention], kv.K[compactAttention], kv.V[compactAttention]} {
				if cache.len != req.StartPos*width {
					return nil, qwen35SequenceFailure("kv-preflight", index, "compact attention cache length does not match start_pos")
				}
			}
			compactAttention++
		}
	}
	return kv, nil
}

func (c *cudaBackend) qwen35SequenceReleaseSinceLocked(start int, keep ...*cudaBuf) {
	if start < 0 || start > len(c.transient) {
		return
	}
	protected := func(candidate *cudaBuf) bool {
		for _, b := range keep {
			if b == candidate {
				return true
			}
		}
		return false
	}
	var release []*cudaBuf
	for _, b := range c.transient[start:] {
		if !protected(b) {
			release = append(release, b)
		}
	}
	c.releaseTransientBuffers(release)
}

func (c *cudaBackend) qwen35SequencePoisonStates(req Qwen35SequencePrefillRequest) {
	for index, layer := range req.Layers {
		if !layer.Linear {
			continue
		}
		if b, ok := req.States[index].Conv.buf.(*cudaBuf); ok {
			atomic.StoreUint32(&b.invalid, 1)
		}
		if b, ok := req.States[index].Recurrent.buf.(*cudaBuf); ok {
			atomic.StoreUint32(&b.invalid, 1)
		}
	}
}

func (c *cudaBackend) Qwen35SequencePrefill(req Qwen35SequencePrefillRequest) (result Qwen35SequencePrefillResult, err error) {
	if admitErr := c.faultLatch.Admit("qwen35-sequence-prefill"); admitErr != nil {
		return result, &Qwen35SequenceError{Stage: "device-admission", Layer: -1, Cause: admitErr}
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	transientStart := len(c.transient)
	failed := true
	defer func() {
		if failed {
			c.qwen35SequenceReleaseSinceLocked(transientStart)
		}
	}()
	kv, err := c.validateQwen35SequenceRequestLocked(req)
	if err != nil {
		return result, err
	}
	tokens := len(req.TokenIDs)
	if err := c.qwen35SequenceReserveKVLocked(kv, len(req.Layers)/4, (req.StartPos+tokens)*kv.stride()); err != nil {
		return result, err
	}
	h2dStart, d2hStart := uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())
	x, xBuf, err := c.qwen35SequenceEmbeddingLocked(req.TokenEmbedding, req.TokenIDs, req.Hidden)
	if err != nil {
		return result, err
	}
	h2dAfterGather, d2hAfterGather := uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())
	compactAttention := 0
	for index, layer := range req.Layers {
		normed, normBuf, normErr := c.qwen35SequenceRMSNormLocked(x, layer.InputNorm, tokens, req.RMSNormEpsilon, "input-norm")
		if normErr != nil {
			return result, &Qwen35SequenceError{Stage: "input-norm", Layer: index, Cause: normErr}
		}
		if layer.Linear {
			branch, branchBuf, branchErr := c.qwen35SequenceGDNLocked(normed, layer, req.States[index], req, tokens, index)
			if branchErr != nil {
				return result, branchErr
			}
			C.fcuda_add_f32(c.cf(x), c.cf(branch), C.int(tokens*req.Hidden))
			c.releaseTransientBuffers([]*cudaBuf{normBuf, branchBuf})
		} else {
			qg, qgBuf, qErr := c.qwen35SequenceMatMulLocked(layer.Q, normed, tokens, "attention-qg")
			if qErr != nil {
				return result, &Qwen35SequenceError{Stage: "attention-qg", Layer: index, Cause: qErr}
			}
			kRaw, kRawBuf, kErr := c.qwen35SequenceMatMulLocked(layer.K, normed, tokens, "attention-k")
			if kErr != nil {
				return result, &Qwen35SequenceError{Stage: "attention-k", Layer: index, Cause: kErr}
			}
			value, valueBuf, vErr := c.qwen35SequenceMatMulLocked(layer.V, normed, tokens, "attention-v")
			if vErr != nil {
				return result, &Qwen35SequenceError{Stage: "attention-v", Layer: index, Cause: vErr}
			}
			q, qBuf, gate, gateBuf, splitErr := c.qwen35SequenceSplitQGLocked(qg, tokens, req.NumHeads, req.HeadDim)
			if splitErr != nil {
				return result, &Qwen35SequenceError{Stage: "attention-split", Layer: index, Cause: splitErr}
			}
			c.releaseTransientBuffers([]*cudaBuf{qgBuf})
			if layer.QNorm.Buf() != nil {
				qNormed, qNormBuf, qNormErr := c.qwen35SequenceRMSNormLocked(q, layer.QNorm, tokens*req.NumHeads, req.RMSNormEpsilon, "attention-q-norm")
				if qNormErr != nil {
					return result, &Qwen35SequenceError{Stage: "attention-q-norm", Layer: index, Cause: qNormErr}
				}
				kNormed, kNormBuf, kNormErr := c.qwen35SequenceRMSNormLocked(kRaw, layer.KNorm, tokens*req.NumKVHeads, req.RMSNormEpsilon, "attention-k-norm")
				if kNormErr != nil {
					return result, &Qwen35SequenceError{Stage: "attention-k-norm", Layer: index, Cause: kNormErr}
				}
				c.releaseTransientBuffers([]*cudaBuf{qBuf, kRawBuf})
				q, qBuf, kRaw, kRawBuf = qNormed, qNormBuf, kNormed, kNormBuf
			}
			qRoPE, qRoPEBuf, kRoPE, kRoPEBuf, ropeErr := c.qwen35SequenceRoPELocked(q, kRaw, tokens, req.StartPos, req.NumHeads, req.NumKVHeads, req.HeadDim, req.RotaryDim, req.RoPEThetaForLayer[index])
			if ropeErr != nil {
				return result, &Qwen35SequenceError{Stage: "attention-rope", Layer: index, Cause: ropeErr}
			}
			c.releaseTransientBuffers([]*cudaBuf{qBuf})
			if appendErr := c.qwen35SequenceAppendKVLocked(kv, compactAttention, kRaw, kRoPE, value, req.StartPos, tokens); appendErr != nil {
				return result, appendErr
			}
			attn, attnBuf, attnErr := c.qwen35SequenceAttentionLocked(qRoPE, kv, compactAttention, tokens, req.StartPos, req.NumHeads, float32(1/math.Sqrt(float64(req.HeadDim))))
			if attnErr != nil {
				return result, &Qwen35SequenceError{Stage: "causal-attention", Layer: index, Cause: attnErr}
			}
			C.fcuda_sigmoid_mul_f32(c.cf(attn), c.cf(gate), C.int(tokens*req.NumHeads*req.HeadDim))
			projected, projectedBuf, projectErr := c.qwen35SequenceMatMulLocked(layer.O, attn, tokens, "attention-output")
			if projectErr != nil {
				return result, &Qwen35SequenceError{Stage: "attention-output", Layer: index, Cause: projectErr}
			}
			C.fcuda_add_f32(c.cf(x), c.cf(projected), C.int(tokens*req.Hidden))
			c.releaseTransientBuffers([]*cudaBuf{normBuf, kRawBuf, valueBuf, qRoPEBuf, kRoPEBuf, gateBuf, attnBuf, projectedBuf})
			compactAttention++
		}
		postNorm, postNormBuf, postErr := c.qwen35SequenceRMSNormLocked(x, layer.PostNorm, tokens, req.RMSNormEpsilon, "post-attention-norm")
		if postErr != nil {
			return result, &Qwen35SequenceError{Stage: "post-attention-norm", Layer: index, Cause: postErr}
		}
		gate, gateBuf, gateErr := c.qwen35SequenceMatMulLocked(layer.Gate, postNorm, tokens, "ffn-gate")
		if gateErr != nil {
			return result, &Qwen35SequenceError{Stage: "ffn-gate", Layer: index, Cause: gateErr}
		}
		up, upBuf, upErr := c.qwen35SequenceMatMulLocked(layer.Up, postNorm, tokens, "ffn-up")
		if upErr != nil {
			return result, &Qwen35SequenceError{Stage: "ffn-up", Layer: index, Cause: upErr}
		}
		activated, activatedBuf, allocErr := c.qwen35SequenceAllocLocked([]int{tokens, req.Intermediate}, "ffn-swiglu")
		if allocErr != nil {
			return result, &Qwen35SequenceError{Stage: "ffn-swiglu", Layer: index, Cause: allocErr}
		}
		C.fcuda_swiglu_f32(c.cf(gate), c.cf(up), c.cf(activated), C.int(tokens*req.Intermediate))
		delta, deltaBuf, downErr := c.qwen35SequenceMatMulLocked(layer.Down, activated, tokens, "ffn-down")
		if downErr != nil {
			return result, &Qwen35SequenceError{Stage: "ffn-down", Layer: index, Cause: downErr}
		}
		C.fcuda_add_f32(c.cf(x), c.cf(delta), C.int(tokens*req.Hidden))
		c.releaseTransientBuffers([]*cudaBuf{postNormBuf, gateBuf, upBuf, activatedBuf, deltaBuf})
	}
	_, finalPanelBuf, finalErr := c.qwen35SequenceRMSNormLocked(x, req.OutputNorm, tokens, req.RMSNormEpsilon, "output-norm")
	if finalErr != nil {
		return result, finalErr
	}
	last, lastBuf, allocErr := c.qwen35SequenceAllocLocked([]int{req.Hidden}, "last-hidden")
	if allocErr != nil {
		return result, allocErr
	}
	C.fcuda_d2d(lastBuf.ptr, offsetF(finalPanelBuf.ptr, (tokens-1)*req.Hidden), C.size_t(req.Hidden*F32.Bytes()))
	c.releaseTransientBuffers([]*cudaBuf{xBuf, finalPanelBuf})
	var logits Tensor
	var logitsBuf *cudaBuf
	if req.NeedLogits {
		logits, logitsBuf, err = c.qwen35SequenceMatMulLocked(req.Output, last, 1, "output-head")
		if err != nil {
			return result, err
		}
		logits.Shape = []int{req.Output.Shape[0]}
	}
	if err := c.qwen35SequenceSyncLocked(); err != nil {
		atomic.StoreUint32(&lastBuf.invalid, 1)
		if logitsBuf != nil {
			atomic.StoreUint32(&logitsBuf.invalid, 1)
		}
		c.qwen35SequencePoisonStates(req)
		c.faultLatch.ObserveError(err, "qwen35-sequence-final-fence")
		return result, err
	}
	h2dEnd, d2hEnd := uint64(C.fcuda_h2dxfer_bytes()), uint64(C.fcuda_hostxfer_bytes())
	transfers := Qwen35SequenceTransferCounters{
		H2DBytes: h2dEnd - h2dStart, D2HBytes: d2hEnd - d2hStart,
		ActivationH2DBytes: h2dEnd - h2dAfterGather, ActivationD2HBytes: d2hEnd - d2hAfterGather,
	}
	if transfers.ActivationH2DBytes != 0 || transfers.ActivationD2HBytes != 0 {
		c.qwen35SequencePoisonStates(req)
		return result, qwen35SequenceFailure("transfer-witness", -1, fmt.Sprintf("activation traffic H2D=%d D2H=%d", transfers.ActivationH2DBytes, transfers.ActivationD2HBytes))
	}
	result = Qwen35SequencePrefillResult{LastHidden: last, Logits: logits, Tokens: tokens, Transfers: transfers}
	c.qwen35SequenceReleaseSinceLocked(transientStart, lastBuf, logitsBuf)
	failed = false
	return result, nil
}
