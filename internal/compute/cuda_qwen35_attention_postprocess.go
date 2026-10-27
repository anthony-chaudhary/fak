//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
int fak_qwen35_attention_postprocess_capability(int tokens);
int fak_qwen35_attention_postprocess_f32(
    const float *Q, const float *K, const float *V,
    const float *QW, const float *KW, float *QOut,
    float *KRawCache, float *KCache, float *VCache,
    int tokens, int start, int rotary, float eps, double theta);
unsigned long long fak_qwen35_attention_postprocess_launch_count(void);
*/
import "C"

import (
	"fmt"
	"math"
	"os"
)

// Experimental, read once. No public setter, decode change, or default promotion.
var cudaQwen35AttentionPostprocessEnabled = os.Getenv("FAK_CUDA_QWEN35_ATTN_POSTPROCESS") == "1"

// Caller holds cudaMu and has reserved the existing privately owned KV backing.
// CUDA Clone deep-copies its prefix; unlike Vulkan there is no shared COW frontier
// to detach here. Declines have no effects; after selection any error is fatal to
// this sequence and must never retry the old composition over a partially written KV.
func (c *cudaBackend) qwen35SequenceAttentionPostprocessLocked(q, k, value, qWeight, kWeight Tensor, kv *cudaKV, layer, tokens, start, rotary int, eps float32, theta float64) (Tensor, *cudaBuf, bool, error) {
	if !cudaQwen35AttentionPostprocessEnabled || c.capturing || qWeight.Buf() == nil {
		return Tensor{}, nil, false, nil
	}
	fail := func(reason string) (Tensor, *cudaBuf, bool, error) {
		return Tensor{}, nil, true, &Qwen35SequenceError{Stage: "attention-postprocess", Layer: layer, Reason: reason}
	}
	if kv == nil || kv.be != c || layer < 0 || layer >= len(kv.Kraw) || layer >= len(kv.K) || layer >= len(kv.V) {
		return fail("KV cache does not own the requested compact attention layer")
	}
	if err := validateQwen35CausalAttentionPanelGeometry(tokens, start, 24, kv.cfg.NumKVHeads, kv.cfg.HeadDim); err != nil {
		return Tensor{}, nil, true, err
	}
	if rotary <= 0 || rotary > 256 || rotary%2 != 0 || !(eps > 0) || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) || !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return fail("invalid rotary dimension, RMS epsilon, or RoPE theta")
	}
	for _, operand := range []struct {
		name   string
		tensor Tensor
		shape  []int
	}{
		{"query", q, []int{tokens, 6144}}, {"key", k, []int{tokens, 1024}}, {"value", value, []int{tokens, 1024}},
		{"query norm", qWeight, []int{256}}, {"key norm", kWeight, []int{256}},
	} {
		if err := c.validateQwen35SequenceTensor(operand.name, operand.tensor, false, operand.shape...); err != nil {
			return Tensor{}, nil, true, &Qwen35SequenceError{Stage: "attention-postprocess", Layer: layer, Cause: err}
		}
	}
	want, end := start*1024, (start+tokens)*1024
	for _, dst := range []*dslice{&kv.Kraw[layer], &kv.K[layer], &kv.V[layer]} {
		if dst.ptr == nil || dst.len != want || dst.cap < end {
			return fail("KV cache is not reserved at the exact append frontier")
		}
	}
	capability := int(C.fak_qwen35_attention_postprocess_capability(C.int(tokens)))
	if capability == 0 {
		return Tensor{}, nil, false, nil
	}
	if capability != 1 {
		return fail(fmt.Sprintf("CUDA capability status %d", capability))
	}
	out, buf, err := c.qwen35SequenceAllocLocked([]int{tokens, 6144}, "attention-postprocess-query")
	if err != nil {
		return Tensor{}, nil, true, err
	}
	status := int(C.fak_qwen35_attention_postprocess_f32(c.cf(q), c.cf(k), c.cf(value), c.cf(qWeight), c.cf(kWeight), c.cf(out), (*C.float)(kv.Kraw[layer].ptr), (*C.float)(kv.K[layer].ptr), (*C.float)(kv.V[layer].ptr), C.int(tokens), C.int(start), C.int(rotary), C.float(eps), C.double(theta)))
	if status != 0 {
		// Match sequence lifetime ownership: drain before releasing the transient;
		// a drain fault is surfaced and latched, never treated as permission to retry.
		if drainErr := c.qwen35SequenceSyncLocked(); drainErr != nil {
			c.faultLatch.ObserveError(drainErr, "attention-postprocess-drain")
			return Tensor{}, nil, true, drainErr
		}
		c.releaseTransientBuffers([]*cudaBuf{buf})
		return fail(fmt.Sprintf("CUDA status %d", status))
	}
	// Exactly the existing AppendKV publication point: all three writes enqueued
	// on the one stream before any metadata advances. The sequence final fence
	// still establishes completion and poisons state on asynchronous failure.
	for _, dst := range []*dslice{&kv.Kraw[layer], &kv.K[layer], &kv.V[layer]} {
		dst.len = end
	}
	if layer == 0 {
		for token := 0; token < tokens; token++ {
			kv.pos = append(kv.pos, start+token)
		}
	}
	return out, buf, true, nil
}

func cudaQwen35AttentionPostprocessLaunchesLocked() uint64 {
	return uint64(C.fak_qwen35_attention_postprocess_launch_count())
}
