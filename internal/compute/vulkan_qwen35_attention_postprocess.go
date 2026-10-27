//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// tryQwen35AttentionPostprocessLocked replaces only the resident F32 panel's
// Q/K norm, partial RoPE and three KV copies. The capability is default-off at
// initialization. Refusal is mutation-free; after selecting, errors are fatal to
// the sequence and must never replay against potentially advanced cache state.
// vulkanMu is held. Existing cache append admission owns COW/frontier semantics.
func (v *vulkanBackend) tryQwen35AttentionPostprocessLocked(
	q, k, values, qNorm, kNorm Tensor, kv *vulkanKV, layer, tokens, startPos, qHeads, kHeads, headDim, rotaryDim int,
	theta float64, eps float32,
) (rotatedQ Tensor, used bool, err error) {
	const maxIndex = int64(1<<31 - 1)
	if !v.haveQwenAttentionPostprocess || tokens <= 1 || int64(tokens) > maxIndex || startPos < 0 || int64(startPos) > maxIndex ||
		qHeads <= 0 || int64(qHeads) > maxIndex || kHeads <= 0 || int64(kHeads) > maxIndex ||
		headDim <= 0 || headDim > 1024 || rotaryDim < 0 || rotaryDim > headDim || rotaryDim%2 != 0 ||
		kv == nil || kv.be != v || layer < 0 || layer >= len(kv.K) || layer >= len(kv.Kraw) || layer >= len(kv.V) {
		return Tensor{}, false, nil
	}
	qw, kw := int64(qHeads)*int64(headDim), int64(kHeads)*int64(headDim)
	end := int64(startPos) + int64(tokens)
	maxFloats := min(maxIndex, int64(^uint(0)>>1)/4)
	if qw > maxFloats || kw > maxFloats || int64(tokens) > maxFloats/qw || end > maxFloats/kw {
		return Tensor{}, false, nil
	}
	valid := func(t Tensor, rows, width int) bool {
		b, ok := t.buf.(*vulkanBuf)
		return ok && b != nil && b.ptr != nil && t.Dtype == F32 && len(t.Shape) == 2 && t.Shape[0] == rows && t.Shape[1] == width
	}
	validNorm := func(t Tensor) bool {
		b, ok := t.buf.(*vulkanBuf)
		return ok && b != nil && b.ptr != nil && t.Dtype == F32 && len(t.Shape) == 1 && t.Shape[0] == headDim
	}
	if !valid(q, tokens, int(qw)) || !valid(k, tokens, int(kw)) || !valid(values, tokens, int(kw)) ||
		!validNorm(qNorm) || !validNorm(kNorm) ||
		C.fvk_qwen35_attention_postprocess_supported(C.int(tokens), C.int(startPos), C.int(qHeads), C.int(kHeads),
			C.int(headDim), C.int(rotaryDim), C.double(theta), C.float(eps)) == 0 {
		return Tensor{}, false, nil
	}
	caches := [...]*vslice{&kv.Kraw[layer], &kv.K[layer], &kv.V[layer]}
	oldLen, added := startPos*int(kw), tokens*int(kw)
	for _, cache := range caches {
		if cache.len != oldLen {
			return Tensor{}, false, nil
		}
	}
	// Capacity reservation alone does not authorize overwriting another fork's
	// tail. Reuse the same admission as growAppend, including shared highWater.
	for _, cache := range caches {
		v.prepareVSliceAppend(cache, added, "Qwen attention postprocess KV append")
	}
	transientStart := len(v.transient)
	rotatedQ, _ = v.devTr([]int{tokens, int(qw)}, F32)
	status := int(C.fvk_qwen35_attention_postprocess_f32(v.vp(q), v.vp(k), v.vp(values), v.vp(qNorm), v.vp(kNorm),
		v.vp(rotatedQ), caches[0].ptr, caches[1].ptr, caches[2].ptr,
		C.int(tokens), C.int(startPos), C.int(qHeads), C.int(kHeads), C.int(headDim), C.int(rotaryDim), C.double(theta), C.float(eps)))
	if status != 0 {
		v.qwen35VulkanSequenceReleaseLocked(transientStart)
		return Tensor{}, false, fmt.Errorf("compute: Vulkan Qwen attention postprocess failed closed (code %d)", status)
	}
	// Metadata publication follows successful checked command recording. Its
	// existing batch barrier/fence owns visibility and deferred allocation frees.
	for _, cache := range caches {
		cache.publishAppend(added)
	}
	return rotatedQ, true, nil
}
