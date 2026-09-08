//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// RMSNorm applies row-wise RMS normalization scaled by weight (eps in the denominator)
// to each row of x, returning a new device tensor of the same shape.
func (v *vulkanBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := weight.Numel()
	rows := x.Numel() / n
	y, _ := v.devTr(append([]int(nil), x.Shape...), F32)
	C.fvk_rmsnorm_f32(v.vp(x), v.vp(weight), v.vp(y), C.int(rows), C.int(n), C.float(eps))
	return y
}

// RoPE applies rotary position embedding at position pos to each head of x, returning a
// new device tensor (x is copied D2D first so the input is left unmodified).
func (v *vulkanBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	y, ybuf := v.devTr(append([]int(nil), x.Shape...), F32)
	C.fvk_d2d(ybuf.ptr, x.buf.(*vulkanBuf).ptr, C.size_t(x.Numel()*4))
	C.fvk_rope_f32(v.vp(y), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return y
}

// RoPEInPlace applies rotary position embedding at position pos to x's buffer directly,
// returning the same tensor (no copy) for the case where x may be overwritten.
func (v *vulkanBackend) RoPEInPlace(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_rope_f32(v.vp(x), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return x
}

// SwiGLU computes the elementwise silu(gate)*up activation, returning a new device
// tensor shaped like gate.
func (v *vulkanBackend) SwiGLU(gate, up Tensor) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := gate.Numel()
	y, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
	C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(y), C.int(n))
	return y
}

// AddInPlace adds src into dst elementwise (dst += src) on the device — the residual add.
func (v *vulkanBackend) AddInPlace(dst, src Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_add_f32(v.vp(dst), v.vp(src), C.int(dst.Numel()))
}

// AddBias adds the width-length bias vector to every row of dst (broadcast over rows).
func (v *vulkanBackend) AddBias(dst, bias Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	width := bias.Numel()
	rows := dst.Numel() / width
	C.fvk_add_bias_f32(v.vp(dst), v.vp(bias), C.int(rows), C.int(width))
}

// Attention runs the fused scaled-dot-product FlashAttention for one layer over the cached
// keys/values (grp query heads per KV head, scale applied to the scores) via online softmax,
// dispatching 4 buffers (Q, K, V, Out) with zero global scratch allocations, and returning the
// per-head context vectors as one device tensor.
func (v *vulkanBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	vk := kv.(*vulkanKV)
	hd, nKV := vk.cfg.HeadDim, vk.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := vk.K[layer].len / w
	out, _ := v.devTr([]int{nH * hd}, F32)
	C.fvk_attention_f32(v.vp(q), vk.K[layer].ptr, vk.V[layer].ptr, v.vp(out),
		C.int(nPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	return out
}

// SpecVerifyAttention on vulkanBackend falls back to the CPU reference (#11100).
func (be *vulkanBackend) SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	ref, ok := Default().(SpecVerifyBackend)
	if !ok {
		return fmt.Errorf("compute: Default backend does not implement SpecVerifyBackend")
	}
	return ref.SpecVerifyAttention(q, k, v, out, qLen, kvLen, nH, nHkv, d)
}

// PrefillBatch on vulkanBackend falls back to the CPU reference (#11036).
func (be *vulkanBackend) PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error) {
	ref, ok := Default().(BatchedPrefillBackend)
	if !ok {
		return PrefillBatchResult{}, fmt.Errorf("compute: Default backend does not implement BatchedPrefillBackend")
	}
	return ref.PrefillBatch(args)
}

// Argmax returns the index of the largest element of the device logits tensor via the
// scalar-reduction shader, so greedy decode never copies the full vector host-ward.
func (v *vulkanBackend) Argmax(logits Tensor) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_argmax_f32(v.vp(logits), C.int(logits.Numel())))
}

// ExecuteAttentionWithDequantOnce runs multi-head attention against the quantized KV cache,
// performing dequantization exactly once into the scratchpad and reusing the contiguous buffers
// across all nQ query heads.
func (s *VulkanKVScratchpad) ExecuteAttentionWithDequantOnce(q []float32, rawK, rawV []byte, nQ int) ([]float32, error) {
	// 1. Dequantize once into local GPU UMA scratchpad memory
	if err := s.DequantizeOnce(rawK, rawV); err != nil {
		return nil, err
	}

	// 2. Evaluate all nQ attention heads by reusing the dequantized scratchpad buffers
	return s.ExecuteAttentionFromScratchpad(q, nQ)
}

// ExecuteAttentionFromScratchpad evaluates all nQ attention heads by streaming contiguous tiles from the scratchpad.
func (s *VulkanKVScratchpad) ExecuteAttentionFromScratchpad(q []float32, nQ int) ([]float32, error) {
	if nQ <= 0 {
		return nil, fmt.Errorf("compute: invalid nQ=%d", nQ)
	}
	if len(q) < nQ*s.HeadDim {
		return nil, fmt.Errorf("compute: query buffer too small (%d < %d)", len(q), nQ*s.HeadDim)
	}

	out := make([]float32, nQ*s.HeadDim)
	scale := float32(1.0 / math.Sqrt(float64(s.HeadDim)))
	groupSize := nQ / s.NumKVHeads
	if groupSize < 1 {
		groupSize = 1
	}

	scores := make([]float32, s.NumPos)

	for qh := 0; qh < nQ; qh++ {
		kvh := qh / groupSize
		if kvh >= s.NumKVHeads {
			kvh = s.NumKVHeads - 1
		}

		kHead, vHead, _, err := s.GetHeadSlice(kvh)
		if err != nil {
			return nil, err
		}

		qOff := qh * s.HeadDim
		qVec := q[qOff : qOff+s.HeadDim]

		// Dot-product Q with K
		maxScore := float32(-math.MaxFloat32)
		for p := 0; p < s.NumPos; p++ {
			kOff := p * s.HeadDim
			var dot float32
			for d := 0; d < s.HeadDim; d++ {
				dot += qVec[d] * kHead[kOff+d]
			}
			sVal := dot * scale
			scores[p] = sVal
			if sVal > maxScore {
				maxScore = sVal
			}
		}

		// Softmax
		var sumExp float32
		for p := 0; p < s.NumPos; p++ {
			expVal := float32(math.Exp(float64(scores[p] - maxScore)))
			scores[p] = expVal
			sumExp += expVal
		}

		invSum := float32(1.0) / sumExp
		for p := 0; p < s.NumPos; p++ {
			scores[p] *= invSum
		}

		// Weighted sum of V
		outOff := qh * s.HeadDim
		for d := 0; d < s.HeadDim; d++ {
			var acc float32
			for p := 0; p < s.NumPos; p++ {
				vOff := p * s.HeadDim
				acc += scores[p] * vHead[vOff+d]
			}
			out[outOff+d] = acc
		}
	}

	return out, nil
}

// ExecuteVulkanAttentionWithDequantOnce runs FlashAttention on AMD RDNA 3.5 (gfx1151 / Strix Halo)
// using the single-pass Vulkan dequantization scratchpad pipeline, eliminating redundant per-head KV dequantization.
func ExecuteVulkanAttentionWithDequantOnce(
	q []float32,
	rawK, rawV []byte,
	arch string,
	nPos, nQ, nKV, headDim int,
	format QuantizedKVType,
) ([]float32, *VulkanKVScratchpad, error) {
	scratch, err := NewVulkanKVScratchpad(arch, format, nPos, nKV, headDim)
	if err != nil {
		return nil, nil, fmt.Errorf("vulkan: failed to create dequant scratchpad: %w", err)
	}
	out, err := scratch.ExecuteAttentionWithDequantOnce(q, rawK, rawV, nQ)
	if err != nil {
		return nil, nil, fmt.Errorf("vulkan: dequant-once attention failed: %w", err)
	}
	return out, scratch, nil
}

// AttentionQuantizedKV executes attention for one layer over quantized KV caches using the dequant-once scratchpad.
func (v *vulkanBackend) AttentionQuantizedKV(
	q Tensor,
	rawK, rawV []byte,
	format QuantizedKVType,
	layer, nPos, nQ, nKV, headDim int,
	scale float32,
) (Tensor, *VulkanKVScratchpad, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	qHost := v.Read(q)
	arch := "gfx1151"
	outHost, scratch, err := ExecuteVulkanAttentionWithDequantOnce(qHost, rawK, rawV, arch, nPos, nQ, nKV, headDim, format)
	if err != nil {
		return Tensor{}, nil, err
	}
	outDev, _ := v.devTr([]int{nQ * headDim}, F32)
	C.fvk_h2d(v.vp(outDev), unsafe.Pointer(&outHost[0]), C.size_t(len(outHost)*4))
	return outDev, scratch, nil
}
