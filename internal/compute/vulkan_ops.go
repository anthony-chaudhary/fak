//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

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

// Attention runs the fused scaled-dot-product attention for one layer over the cached
// keys/values (grp query heads per KV head, scale applied to the scores), returning the
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
