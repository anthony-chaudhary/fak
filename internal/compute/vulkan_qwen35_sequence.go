//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"unsafe"
)

var _ Qwen35SequenceRawHiddenBackend = (*vulkanBackend)(nil)

func (*vulkanBackend) Qwen35SequencePrefillPath() string { return Qwen35SequencePrefillPath }

func (*vulkanBackend) Qwen35SequenceEmbeddingRowsPath() string {
	return Qwen35SequenceEmbeddingRowsPath
}

func (*vulkanBackend) Qwen35SequenceAllLogitsPath() string { return Qwen35SequenceAllLogitsPath }

func (*vulkanBackend) Qwen35SequenceRawHiddenPath() string { return Qwen35SequenceRawHiddenPath }

func qwen35VulkanSequenceError(stage string, layer int, reason string) error {
	return &Qwen35SequenceError{Stage: stage, Layer: layer, Reason: reason}
}

func qwen35VulkanSequenceQKEpsilon(req Qwen35SequencePrefillRequest) float32 {
	return req.RMSNormEpsilon
}

// Shader dimensions and flattened indices use signed 32-bit integers.
func qwen35VulkanSequenceSize(dims ...int) (int, bool) {
	n := int64(1)
	for _, d := range dims {
		if d < 0 || int64(d) > math.MaxInt32 || (d != 0 && n > math.MaxInt32/int64(d)) {
			return 0, false
		}
		n *= int64(d)
	}
	return int(n), true
}

func (v *vulkanBackend) validateQwen35VulkanSequence(req Qwen35SequencePrefillRequest) (*vulkanKV, error) {
	fail := func(stage, reason string) (*vulkanKV, error) {
		return nil, qwen35VulkanSequenceError(stage, -1, reason)
	}
	if req.Path != Qwen35SequencePrefillPath || len(req.TokenIDs) == 0 || len(req.TokenIDs) > math.MaxInt32 || req.StartPos < 0 || req.StartPos > math.MaxInt32-len(req.TokenIDs) {
		return fail("request", "invalid capability, empty token panel, or invalid position range")
	}
	if req.CapturePrefixReplay && (!req.NeedAllLogits || len(req.TokenIDs) > 4) {
		return fail("request", "prefix replay capture requires all-row logits and 1..4 tokens")
	}
	for _, d := range []int{req.Hidden, req.Intermediate, req.NumHeads, req.NumKVHeads, req.HeadDim, req.RotaryDim, req.NumKeyHeads, req.NumValueHeads, req.KeyHeadDim, req.ValueHeadDim, req.ConvKernel} {
		if d <= 0 || int64(d) > math.MaxInt32 {
			return fail("geometry", "dimensions must be positive signed shader integers")
		}
	}
	if req.NumHeads%req.NumKVHeads != 0 || req.NumValueHeads%req.NumKeyHeads != 0 || req.HeadDim > 1024 || req.ValueHeadDim > 1024 || req.KeyHeadDim > 1024 || req.RotaryDim > req.HeadDim || req.RotaryDim%2 != 0 {
		return fail("geometry", "invalid grouped heads, rotary dimension, or bounded head width")
	}
	if !(req.RMSNormEpsilon > 0) || math.IsNaN(float64(req.RMSNormEpsilon)) || math.IsInf(float64(req.RMSNormEpsilon), 0) {
		return fail("geometry", "RMS epsilon must be finite and positive")
	}
	qkEpsilon := qwen35VulkanSequenceQKEpsilon(req)
	if !(qkEpsilon > 0) || math.IsNaN(float64(qkEpsilon)) || math.IsInf(float64(qkEpsilon), 0) {
		return fail("geometry", "Q/K RMS epsilon must be finite and positive")
	}
	if len(req.Layers) == 0 || len(req.Layers)%4 != 0 || len(req.States) != len(req.Layers) || len(req.RoPEThetaForLayer) != len(req.Layers) {
		return fail("geometry", "complete four-layer hybrid groups require matching states and RoPE parameters")
	}
	keyDim, keyOK := qwen35VulkanSequenceSize(req.NumKeyHeads, req.KeyHeadDim)
	valueDim, valueOK := qwen35VulkanSequenceSize(req.NumValueHeads, req.ValueHeadDim)
	if !keyOK || !valueOK || int64(keyDim)*2+int64(valueDim) > math.MaxInt32 {
		return fail("geometry", "GDN projection width overflows shader indexing")
	}
	convDim := 2*keyDim + valueDim
	qWidth, qOK := qwen35VulkanSequenceSize(req.NumHeads, req.HeadDim)
	kvWidth, kvOK := qwen35VulkanSequenceSize(req.NumKVHeads, req.HeadDim)
	if !qOK || !kvOK || qWidth > math.MaxInt32/2 {
		return fail("geometry", "attention projection width overflows shader indexing")
	}
	// These kernels dispatch in X only. The Vulkan guaranteed limit keeps
	// preflight portable even on devices whose implementation allows more.
	const maxGroups = 65535
	for _, groups := range []int64{
		int64(len(req.TokenIDs)) * int64(req.NumHeads),
		int64(req.NumValueHeads), (int64(convDim) + 63) / 64,
		(int64(len(req.TokenIDs))*(int64(qWidth)+int64(kvWidth)) + 255) / 256,
		(int64(len(req.TokenIDs))*int64(req.Hidden) + 255) / 256,
		(int64(len(req.TokenIDs))*int64(req.Intermediate) + 255) / 256,
	} {
		if groups > maxGroups {
			return fail("geometry", "sequence exceeds portable Vulkan X workgroup limit")
		}
	}
	for _, dims := range [][]int{
		{len(req.TokenIDs), req.Hidden}, {len(req.TokenIDs), req.Intermediate},
		{len(req.TokenIDs), 2 * qWidth}, {len(req.TokenIDs), convDim},
		{len(req.TokenIDs), valueDim}, {len(req.TokenIDs), req.NumValueHeads},
		{req.StartPos + len(req.TokenIDs), kvWidth},
		{req.NumValueHeads, req.KeyHeadDim, req.ValueHeadDim},
		{req.ConvKernel, convDim},
	} {
		n, ok := qwen35VulkanSequenceSize(dims...)
		if !ok || int64(n)*4 > int64(^uint(0)>>1) || singleResourceCapExceeded(n*4, v.maxBufferBytes) {
			return fail("geometry", "panel or persistent buffer exceeds shader/device allocation limits")
		}
	}
	if len(req.Layers) == Qwen35DenseMainLayers && (req.Hidden != Qwen35DenseHidden || req.Intermediate != Qwen35DenseIntermediate || req.NumHeads != Qwen35DenseQueryHeads || req.NumKVHeads != Qwen35DenseKVHeads || req.HeadDim != Qwen35DenseHeadDim || req.RotaryDim != Qwen35DenseHeadDim/4 || req.NumKeyHeads != Qwen35DenseGDNGroups || req.NumValueHeads != Qwen35DenseGDNRank || req.KeyHeadDim != Qwen35DenseGDNState || req.ValueHeadDim != Qwen35DenseGDNState || req.ConvKernel != Qwen35DenseGDNConv) {
		return fail("production-geometry", "64-layer dense text stack does not match Qwen3.8 geometry")
	}
	type operand struct {
		name   string
		t      Tensor
		matrix bool
		shape  []int
		layer  int
	}
	var operands []operand
	add := func(layer int, name string, t Tensor, matrix bool, shape ...int) {
		operands = append(operands, operand{name, t, matrix, shape, layer})
	}
	vocab, embeddingShape, err := qwen35SequenceEmbeddingContract(req)
	if err != nil {
		return nil, err
	}
	if int64(vocab) > math.MaxInt32 || int64(vocab)*4 > int64(^uint(0)>>1) || singleResourceCapExceeded(vocab*4, v.maxBufferBytes) {
		return fail("geometry", "output vector exceeds shader/device allocation limits")
	}
	if req.NeedAllLogits {
		panel, ok := qwen35VulkanSequenceSize(len(req.TokenIDs), vocab)
		if !ok || int64(panel)*4 > int64(^uint(0)>>1) || singleResourceCapExceeded(panel*4, v.maxBufferBytes) {
			return fail("geometry", "output panel exceeds shader/device allocation limits")
		}
	}
	for _, id := range req.TokenIDs {
		if id < 0 || id >= vocab {
			return fail("embedding-gather", "token ID is outside embedding vocabulary")
		}
	}
	add(-1, "embedding", req.TokenEmbedding, false, embeddingShape...)
	add(-1, "output_norm", req.OutputNorm, false, req.Hidden)
	add(-1, "output", req.Output, true, vocab, req.Hidden)
	kv, ok := req.KV.(*vulkanKV)
	if !ok || kv == nil || kv.be != v || kv.Len() != req.StartPos || kv.cfg.NumKVHeads != req.NumKVHeads || kv.cfg.HeadDim != req.HeadDim || len(kv.K) < len(req.Layers)/4 || len(kv.Kraw) < len(req.Layers)/4 || len(kv.V) < len(req.Layers)/4 {
		return fail("kv-preflight", "KV ownership, prefix length, or compact attention geometry mismatch")
	}
	for i, pos := range kv.pos {
		if pos != i {
			return fail("kv-preflight", "sequence requires an unevicted contiguous prefix")
		}
	}
	states := make(map[unsafe.Pointer]int)
	cachePointers := make(map[unsafe.Pointer]bool)
	attention := 0
	for i, layer := range req.Layers {
		if layer.Linear != ((i+1)%4 != 0) {
			return fail("geometry", "hybrid cadence must be three GDN blocks followed by attention")
		}
		theta := req.RoPEThetaForLayer[i]
		if theta < math.SmallestNonzeroFloat32 || math.IsNaN(theta) || theta > math.MaxFloat32 {
			return fail("geometry", "RoPE theta must be finite and positive")
		}
		add(i, "input_norm", layer.InputNorm, false, req.Hidden)
		add(i, "post_norm", layer.PostNorm, false, req.Hidden)
		add(i, "gate", layer.Gate, true, req.Intermediate, req.Hidden)
		add(i, "up", layer.Up, true, req.Intermediate, req.Hidden)
		add(i, "down", layer.Down, true, req.Hidden, req.Intermediate)
		if layer.Linear {
			add(i, "gdn_qkv", layer.GDNInQKV, true, convDim, req.Hidden)
			add(i, "gdn_z", layer.GDNInZ, true, valueDim, req.Hidden)
			add(i, "gdn_b", layer.GDNInB, true, req.NumValueHeads, req.Hidden)
			add(i, "gdn_a", layer.GDNInA, true, req.NumValueHeads, req.Hidden)
			convShape := layer.GDNConv.Shape
			validConv := len(convShape) == 1 && convShape[0] == convDim*req.ConvKernel || len(convShape) == 2 && convShape[0] == convDim && convShape[1] == req.ConvKernel || len(convShape) == 3 && convShape[0] == convDim && convShape[1] == 1 && convShape[2] == req.ConvKernel
			if !validConv {
				return fail("tensor-preflight", "invalid depthwise convolution weight shape")
			}
			add(i, "gdn_conv", layer.GDNConv, false, convShape...)
			add(i, "gdn_a_log", layer.GDNALog, false, req.NumValueHeads)
			add(i, "gdn_dt_bias", layer.GDNDTBias, false, req.NumValueHeads)
			add(i, "gdn_norm", layer.GDNNorm, false, req.ValueHeadDim)
			add(i, "gdn_out", layer.GDNOut, true, req.Hidden, valueDim)
			add(i, "conv_state", req.States[i].Conv, false, req.ConvKernel-1, convDim)
			add(i, "recurrent_state", req.States[i].Recurrent, false, req.NumValueHeads, req.KeyHeadDim, req.ValueHeadDim)
			for _, t := range []Tensor{req.States[i].Conv, req.States[i].Recurrent} {
				b, ok := t.buf.(*vulkanBuf)
				if !ok || b == nil || b.ptr == nil || b.class != MemoryKVCache {
					return fail("state-preflight", "GDN state must use live durable KV allocations")
				}
				if _, alias := states[b.ptr]; alias {
					return fail("state-preflight", "GDN states must not alias")
				}
				states[b.ptr] = i
			}
		} else {
			add(i, "attention_qg", layer.Q, true, 2*qWidth, req.Hidden)
			add(i, "attention_k", layer.K, true, kvWidth, req.Hidden)
			add(i, "attention_v", layer.V, true, kvWidth, req.Hidden)
			add(i, "attention_o", layer.O, true, req.Hidden, qWidth)
			if (layer.QNorm.Buf() == nil) != (layer.KNorm.Buf() == nil) {
				return fail("tensor-preflight", "Q/K normalization must be both present or both absent")
			}
			if layer.QNorm.Buf() != nil {
				add(i, "q_norm", layer.QNorm, false, req.HeadDim)
				add(i, "k_norm", layer.KNorm, false, req.HeadDim)
			}
			for _, cache := range []vslice{kv.Kraw[attention], kv.K[attention], kv.V[attention]} {
				if cache.len != req.StartPos*kvWidth || cache.cap < cache.len || (cache.cap > 0 && cache.ptr == nil) {
					return fail("kv-preflight", "compact KV allocation or prefix length is invalid")
				}
				if cache.ptr != nil {
					if cachePointers[cache.ptr] {
						return fail("kv-preflight", "mutable KV buffers must not alias")
					}
					cachePointers[cache.ptr] = true
				}
			}
			attention++
		}
	}
	for ptr := range states {
		if cachePointers[ptr] {
			return fail("state-preflight", "GDN state aliases an attention cache")
		}
	}
	for _, op := range operands {
		bad := func(reason string) (*vulkanKV, error) {
			return nil, qwen35VulkanSequenceError("tensor-preflight", op.layer, op.name+": "+reason)
		}
		if len(op.t.Shape) != len(op.shape) {
			return bad("shape mismatch")
		}
		for i, d := range op.shape {
			if op.t.Shape[i] != d {
				return bad("shape mismatch")
			}
		}
		n, ok := qwen35VulkanSequenceSize(op.shape...)
		b, resident := op.t.buf.(*vulkanBuf)
		if !ok || op.t.Backend() != v || op.t.Layout != RowMajor || !resident || b == nil {
			return bad("requires capacity-valid resident row-major tensor owned by this backend")
		}
		bytes := int64(n) * 4
		if op.matrix {
			rows := int64(len(req.TokenIDs))
			if op.name == "output" {
				rows = 1
				if req.NeedAllLogits {
					rows = int64(len(req.TokenIDs))
				}
			}
			groups := rows
			switch op.t.Dtype {
			case Q8_0:
				groups *= (int64(op.shape[0]) + 255) / 256
			case Q4_K, Q2_K:
				groups = (rows*int64(op.shape[0]) + 63) / 64
			}
			if groups > maxGroups {
				return bad("projection exceeds portable Vulkan X workgroup limit")
			}
			switch op.t.Dtype {
			case F32:
			case Q4_K, Q2_K:
				if op.shape[1]%256 != 0 {
					return bad("K-quant input width must be divisible by 256")
				}
				blockBytes := int64(144)
				if op.t.Dtype == Q2_K {
					blockBytes = 84
				}
				bytes = int64(n/256) * blockBytes
			case Q8_0:
				if !v.haveQ8 || op.t.Quant == nil || op.t.Quant.Block != 32 || op.shape[1]%32 != 0 {
					return bad("requires supported block-32 Q8 storage")
				}
				bytes = int64(n)
				if len(b.q8Chunks) != 0 {
					rows := 0
					for _, chunk := range b.q8Chunks {
						need := int64(chunk.rows) * int64(op.shape[1])
						if chunk.rows <= 0 || chunk.rowStart != rows || chunk.ptr == nil || chunk.scalePtr == nil || int64(chunk.n) < need || int64(chunk.scaleN) < need/32*4 {
							return bad("invalid Q8 chunk capacity or row coverage")
						}
						if _, alias := states[chunk.ptr]; alias {
							return bad("weight aliases persistent state")
						}
						if _, alias := states[chunk.scalePtr]; alias {
							return bad("scales alias persistent state")
						}
						if cachePointers[chunk.ptr] || cachePointers[chunk.scalePtr] {
							return bad("Q8 weight or scales alias an attention cache")
						}
						rows += chunk.rows
					}
					if rows != op.shape[0] {
						return bad("Q8 chunks do not cover matrix")
					}
					continue
				}
				if b.scalePtr == nil || int64(b.scaleN) < int64(n/32)*4 {
					return bad("Q8 scales have insufficient resident capacity")
				}
			default:
				return bad("unsupported matrix dtype")
			}
		} else if op.t.Dtype != F32 {
			return bad("non-matrix operand must be F32")
		}
		if b.ptr == nil || int64(b.n) < bytes {
			return bad("missing or undersized resident buffer")
		}
		if cachePointers[b.ptr] || cachePointers[b.scalePtr] {
			return bad("operand aliases mutable attention cache")
		}
		if op.name != "conv_state" && op.name != "recurrent_state" {
			if _, alias := states[b.ptr]; alias {
				return bad("operand aliases mutable GDN state")
			}
			if _, alias := states[b.scalePtr]; b.scalePtr != nil && alias {
				return bad("scales alias mutable GDN state")
			}
		}
	}
	return kv, nil
}

func (v *vulkanBackend) qwen35VulkanSequenceMatMulLocked(w, x Tensor, tokens int) Tensor {
	out, in := w.Shape[0], w.Shape[1]
	y, _ := v.devTr([]int{tokens, out}, F32)
	switch w.Dtype {
	case F32:
		C.fvk_matmul_f32(v.vp(w), v.vp(x), v.vp(y), C.int(out), C.int(in), C.int(tokens))
	case Q8_0:
		v.q8MatMulLocked(w, x, y, out, in, tokens)
	case Q4_K:
		v.q4kMatMulLocked(w, x, y, out, in, tokens)
	case Q2_K:
		v.q2kMatMulLocked(w, x, y, out, in, tokens)
	}
	return y
}

func (v *vulkanBackend) qwen35VulkanSequenceReleaseLocked(start int, keep ...Tensor) {
	retained := v.transient[:start]
	for _, b := range v.transient[start:] {
		protect := false
		for _, t := range keep {
			if t.buf == b {
				protect = true
				break
			}
		}
		if protect {
			retained = append(retained, b)
			continue
		}
		if b != nil && b.ptr != nil {
			C.fvk_free(b.ptr)
			b.ptr = nil
		}
	}
	v.transient = retained
}

// qwen35SequenceKVGeometricCapacity computes the capacity for a sequence KV cache vslice under a guarded
// geometric growth policy. When currentCap is insufficient for need, it geometrically doubles currentCap
// (borrowed from growAppend: cap*2) with integer overflow and single-resource physical device ceiling guards.
func (v *vulkanBackend) qwen35SequenceKVGeometricCapacity(currentCap, need int) int {
	if currentCap >= need {
		return currentCap
	}
	if currentCap <= 0 {
		return need
	}
	// Check for explicit exact-reserve ablation override.
	if os.Getenv("FAK_QWEN35_SEQUENCE_KV_EXACT") == "1" {
		return need
	}
	// Geometric doubling (borrowed from growAppend: cap*2).
	ncap := need
	if currentCap <= (math.MaxInt-1)/2 {
		ncap = currentCap * 2
	} else {
		ncap = math.MaxInt
	}
	if ncap < need {
		ncap = need
	}
	// Guard against byte-count integer overflow (ncap * 4).
	const maxFloats = math.MaxInt / 4
	if ncap > maxFloats {
		if need <= maxFloats {
			ncap = maxFloats
		} else {
			ncap = need
		}
	}
	// Single-resource buffer ceiling guard from physical device.
	if v != nil && v.maxBufferBytes > 0 {
		maxBufFloats := int(v.maxBufferBytes / 4)
		if maxBufFloats > 0 && ncap > maxBufFloats {
			if need <= maxBufFloats {
				ncap = maxBufFloats
			} else {
				ncap = need
			}
		}
	}
	return ncap
}

// qwen35SequenceReserveGeometric reserves capacity for a single KV cache vslice using guarded geometric
// growth. It preserves cache.len (the number of valid populated floats), allocates the geometric capacity
// via dallocKVFor, copies existing data via C.fvk_d2d, frees the old buffer, and updates cache.ptr and cache.cap.
func (v *vulkanBackend) qwen35SequenceReserveGeometric(cache *vslice, need int, what string) {
	if cache.cap >= need {
		return
	}
	ncap := v.qwen35SequenceKVGeometricCapacity(cache.cap, need)
	v.makeVSliceWritableCapacity(cache, need, ncap, true, what)
}

// qwen35SequenceUploadKVFloatsForTest copies host float32s into a cache vslice at float offset offsetFloats.
func (v *vulkanBackend) qwen35SequenceUploadKVFloatsForTest(cache *vslice, offsetFloats int, data []float32) {
	if cache == nil || cache.ptr == nil || len(data) == 0 {
		return
	}
	src := v.Upload(NewF32(Default(), []int{len(data)}, data), F32)
	defer v.Free(src)
	v.makeVSliceWritable(cache, cache.cap, true, "Qwen sequence KV test upload")
	C.fvk_d2d_off(cache.ptr, C.size_t(offsetFloats*4), v.vp(src), C.size_t(len(data)*4))
	if end := offsetFloats + len(data); cache.backing != nil && end > cache.backing.highWater {
		cache.backing.highWater = end
	}
}

// qwen35SequenceReadKVFloatsForTest reads back countFloats float32s from a cache vslice at float offset 0.
func (v *vulkanBackend) qwen35SequenceReadKVFloatsForTest(cache *vslice, countFloats int) []float32 {
	if cache == nil || cache.ptr == nil || countFloats <= 0 {
		return nil
	}
	dst, _ := v.devTr([]int{countFloats}, F32)
	defer v.Free(dst)
	C.fvk_d2d(v.vp(dst), cache.ptr, C.size_t(countFloats*4))
	return v.Read(dst)
}

// qwen35SequenceCausalAttentionForTest dispatches fvk_qwen35_causal_attention_panel_f32 on device pointers.
func (v *vulkanBackend) qwen35SequenceCausalAttentionForTest(qrPtr, kPtr, vPtr, outPtr unsafe.Pointer, tokens, prefix, nH, nKV, hd int, scale float32) int {
	return int(C.fvk_qwen35_causal_attention_panel_f32(qrPtr, kPtr, vPtr, outPtr, C.int(tokens), C.int(prefix), C.int(nH), C.int(nKV), C.int(hd), C.float(scale)))
}

// qwen35SequenceFreeVsliceForTest releases device memory associated with a vslice.
func (v *vulkanBackend) qwen35SequenceFreeVsliceForTest(cache *vslice) {
	if cache != nil && cache.ptr != nil {
		cache.releaseBacking()
		cache.len = 0
	}
}

// Qwen35SequencePrefill runs the prompt layer-major. Dense projections and full
// attention process token panels; recurrent GDN dependencies remain inside the
// device kernel. Each layer fence bounds scratch lifetime independently of depth.
func (v *vulkanBackend) Qwen35SequencePrefill(req Qwen35SequencePrefillRequest) (result Qwen35SequencePrefillResult, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	start := len(v.transient)
	stage, layerIndex := "request", -1
	executing := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, ok := recovered.(error)
			if !ok {
				cause = fmt.Errorf("%v", recovered)
			}
			err = &Qwen35SequenceError{Stage: stage, Layer: layerIndex, Cause: cause}
		}
		if err != nil {
			if executing {
				C.fvk_batch_flush_status()
				// A partial layer stack cannot be replayed against advanced recurrent
				// state. Null handles also make the scalar GDN primitive fail closed.
				for i, layer := range req.Layers {
					if layer.Linear {
						for _, t := range []Tensor{req.States[i].Conv, req.States[i].Recurrent} {
							if b, ok := t.buf.(*vulkanBuf); ok && b != nil && b.ptr != nil {
								C.fvk_free(b.ptr)
								b.ptr = nil
								b.n = 0
							}
						}
					}
				}
			}
			v.qwen35VulkanSequenceReleaseLocked(start)
			result = Qwen35SequencePrefillResult{}
		}
	}()
	kv, err := v.validateQwen35VulkanSequence(req)
	if err != nil {
		return result, err
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return result, qwen35VulkanSequenceError("device-admission", -1, fmt.Sprintf("Vulkan submission fault %d", status))
	}
	check := func(code C.int) error {
		if code != 0 {
			return qwen35VulkanSequenceError(stage, layerIndex, fmt.Sprintf("Vulkan operation failed with status %d", int(code)))
		}
		return nil
	}
	tokens := len(req.TokenIDs)
	qWidth, kvWidth := req.NumHeads*req.HeadDim, req.NumKVHeads*req.HeadDim
	valueDim := req.NumValueHeads * req.ValueHeadDim
	convDim := 2*req.NumKeyHeads*req.KeyHeadDim + valueDim
	normEpsilon := func(x, w Tensor, rows int, eps float32) Tensor {
		y, _ := v.devTr(append([]int(nil), x.Shape...), F32)
		C.fvk_rmsnorm_f32(v.vp(x), v.vp(w), v.vp(y), C.int(rows), C.int(w.Numel()), C.float(eps))
		return y
	}
	norm := func(x, w Tensor, rows int) Tensor { return normEpsilon(x, w, rows, req.RMSNormEpsilon) }
	mul := func(w, x Tensor) Tensor { return v.qwen35VulkanSequenceMatMulLocked(w, x, tokens) }
	executing = true
	stage = "kv-reserve"
	C.fvk_batch_begin()
	need := (req.StartPos + tokens) * kvWidth
	for i := 0; i < len(req.Layers)/4; i++ {
		for _, cache := range []*vslice{&kv.Kraw[i], &kv.K[i], &kv.V[i]} {
			v.qwen35SequenceReserveGeometric(cache, need, "Qwen sequence KV reservation")
		}
	}
	h2dStart, d2hStart := uint64(C.fvk_h2d_bytes()), uint64(C.fvk_d2h_bytes())
	stage = "embedding-gather"
	x, _ := v.devTr([]int{tokens, req.Hidden}, F32)
	if req.TokenEmbeddingRows {
		C.fvk_d2d(v.vp(x), v.vp(req.TokenEmbedding), C.size_t(tokens*req.Hidden*4))
	} else {
		for row, id := range req.TokenIDs {
			C.fvk_d2d_range(v.vp(x), C.size_t(row*req.Hidden*4), v.vp(req.TokenEmbedding), C.size_t(id*req.Hidden*4), C.size_t(req.Hidden*4))
		}
	}
	attention := 0
	var replayProjections []vulkanQwen35ReplayProjection
	for i, layer := range req.Layers {
		layerIndex, stage = i, "input-norm"
		C.fvk_batch_begin()
		n := norm(x, layer.InputNorm, tokens)
		var branch Tensor
		if layer.Linear {
			stage = "gdn-projections"
			mixed, z, beta, alpha := mul(layer.GDNInQKV, n), mul(layer.GDNInZ, n), mul(layer.GDNInB, n), mul(layer.GDNInA, n)
			if req.CapturePrefixReplay {
				replayProjections = append(replayProjections, vulkanQwen35ReplayProjection{layer: i, mixed: mixed, z: z, beta: beta, alpha: alpha})
			}
			core, _ := v.devTr([]int{tokens, valueDim}, F32)
			stage = "gdn-sequence"
			s := req.States[i]
			if err = check(C.fvk_qwen35_gdn_preprojected_f32(v.vp(mixed), v.vp(z), v.vp(beta), v.vp(alpha), v.vp(layer.GDNConv), v.vp(layer.GDNALog), v.vp(layer.GDNDTBias), v.vp(layer.GDNNorm), v.vp(s.Conv), v.vp(s.Recurrent), v.vp(core), C.int(tokens), C.int(convDim), C.int(req.NumKeyHeads), C.int(req.NumValueHeads), C.int(req.KeyHeadDim), C.int(req.ValueHeadDim), C.int(req.ConvKernel), C.float(req.RMSNormEpsilon))); err != nil {
				return result, err
			}
			C.fvk_batch_begin()
			branch = mul(layer.GDNOut, core)
		} else {
			stage = "attention-projections"
			qg, k, values := mul(layer.Q, n), mul(layer.K, n), mul(layer.V, n)
			q, _ := v.devTr([]int{tokens, qWidth}, F32)
			gate, _ := v.devTr([]int{tokens, qWidth}, F32)
			stage = "attention-split"
			if err = check(C.fvk_qwen35_split_qg_panel_f32(v.vp(qg), v.vp(q), v.vp(gate), C.int(tokens), C.int(req.NumHeads), C.int(req.HeadDim))); err != nil {
				return result, err
			}
			stage = "attention-qk-norm"
			if layer.QNorm.Buf() != nil {
				q = normEpsilon(q, layer.QNorm, tokens*req.NumHeads, qwen35VulkanSequenceQKEpsilon(req))
				k = normEpsilon(k, layer.KNorm, tokens*req.NumKVHeads, qwen35VulkanSequenceQKEpsilon(req))
			}
			qr, _ := v.devTr([]int{tokens, qWidth}, F32)
			kr, _ := v.devTr([]int{tokens, kvWidth}, F32)
			stage = "attention-rope"
			if err = check(C.fvk_qwen35_partial_rope_panel_f32(v.vp(q), v.vp(k), v.vp(qr), v.vp(kr), C.int(tokens), C.int(req.StartPos), C.int(req.NumHeads), C.int(req.NumKVHeads), C.int(req.HeadDim), C.int(req.RotaryDim), C.double(req.RoPEThetaForLayer[i]))); err != nil {
				return result, err
			}
			stage = "kv-append"
			v.growAppend(&kv.Kraw[attention], v.vp(k), tokens*kvWidth, "Qwen sequence raw keys")
			v.growAppend(&kv.K[attention], v.vp(kr), tokens*kvWidth, "Qwen sequence rotated keys")
			v.growAppend(&kv.V[attention], v.vp(values), tokens*kvWidth, "Qwen sequence values")
			context, _ := v.devTr([]int{tokens, qWidth}, F32)
			stage = "causal-attention"
			if err = check(C.fvk_qwen35_causal_attention_panel_f32(v.vp(qr), kv.K[attention].ptr, kv.V[attention].ptr, v.vp(context), C.int(tokens), C.int(req.StartPos), C.int(req.NumHeads), C.int(req.NumKVHeads), C.int(req.HeadDim), C.float(1/math.Sqrt(float64(req.HeadDim))))); err != nil {
				return result, err
			}
			stage = "attention-gate"
			if err = check(C.fvk_sigmoid_mul_f32(v.vp(context), v.vp(gate), C.int(tokens*qWidth))); err != nil {
				return result, err
			}
			branch = mul(layer.O, context)
			attention++
		}
		C.fvk_add_f32(v.vp(x), v.vp(branch), C.int(tokens*req.Hidden))
		stage = "ffn"
		post := norm(x, layer.PostNorm, tokens)
		gate, up := mul(layer.Gate, post), mul(layer.Up, post)
		activated, _ := v.devTr([]int{tokens, req.Intermediate}, F32)
		C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(activated), C.int(tokens*req.Intermediate))
		delta := mul(layer.Down, activated)
		C.fvk_add_f32(v.vp(x), v.vp(delta), C.int(tokens*req.Hidden))
		stage = "layer-fence"
		if err = check(C.fvk_batch_flush_status()); err != nil {
			return result, err
		}
		keep := append([]Tensor{x}, qwen35ReplayProjectionTensors(replayProjections)...)
		v.qwen35VulkanSequenceReleaseLocked(start, keep...)
	}
	layerIndex, stage = -1, "output-norm"
	C.fvk_batch_begin()
	lastRaw, _ := v.devTr([]int{req.Hidden}, F32)
	C.fvk_d2d_range(v.vp(lastRaw), 0, v.vp(x), C.size_t((tokens-1)*req.Hidden*4), C.size_t(req.Hidden*4))
	last := norm(lastRaw, req.OutputNorm, 1)
	var logits, logitsRows Tensor
	if req.NeedLogits {
		stage = "output-head"
		logits = v.qwen35VulkanSequenceMatMulLocked(req.Output, last, 1)
		logits.Shape = []int{req.Output.Shape[0]}
	}
	if req.NeedAllLogits {
		stage = "output-all-norm"
		all := norm(x, req.OutputNorm, tokens)
		stage = "output-all-head"
		logitsRows = v.qwen35VulkanSequenceMatMulLocked(req.Output, all, tokens)
		logitsRows.Shape = []int{tokens, req.Output.Shape[0]}
	}
	stage = "final-fence"
	if err = check(C.fvk_batch_flush_status()); err != nil {
		return result, err
	}
	h2d, d2h := uint64(C.fvk_h2d_bytes())-h2dStart, uint64(C.fvk_d2h_bytes())-d2hStart
	if h2d != 0 || d2h != 0 {
		return result, qwen35VulkanSequenceError("transfer-witness", -1, fmt.Sprintf("resident sequence copied H2D=%d D2H=%d bytes", h2d, d2h))
	}
	for pos := req.StartPos; pos < req.StartPos+tokens; pos++ {
		kv.pos = append(kv.pos, pos)
	}
	keep := []Tensor{last, logits, logitsRows}
	var rawHiddenRows Tensor
	if req.CaptureRawHidden {
		rawHiddenRows = x
		keep = append(keep, rawHiddenRows)
	}
	keep = append(keep, qwen35ReplayProjectionTensors(replayProjections)...)
	v.qwen35VulkanSequenceReleaseLocked(start, keep...)
	var prefixReplay Qwen35SequencePrefixReplay
	if req.CapturePrefixReplay {
		checkpoint := &vulkanQwen35PrefixReplay{
			backend: v, kv: kv, startPos: req.StartPos, tokens: tokens, kvWidth: kvWidth,
			convDim: convDim, valueDim: valueDim, numKeyHeads: req.NumKeyHeads, numValueHeads: req.NumValueHeads,
			keyHeadDim: req.KeyHeadDim, valueHeadDim: req.ValueHeadDim, convKernel: req.ConvKernel, rmsEpsilon: req.RMSNormEpsilon,
			layers: append([]Qwen35SequenceLayer(nil), req.Layers...), states: append([]Qwen35SequenceState(nil), req.States...),
			projections: replayProjections,
		}
		owned, detachErr := v.qwen35DetachReplayBuffersLocked(replayProjections)
		if detachErr != nil {
			return result, detachErr
		}
		checkpoint.owned = owned
		prefixReplay = checkpoint
	}
	result = Qwen35SequencePrefillResult{LastHidden: last, RawHiddenRows: rawHiddenRows, Logits: logits, LogitsRows: logitsRows, PrefixReplay: prefixReplay, Tokens: tokens, Transfers: Qwen35SequenceTransferCounters{H2DBytes: h2d, D2HBytes: d2h, ActivationH2DBytes: h2d, ActivationD2HBytes: d2h}}
	return result, nil
}
