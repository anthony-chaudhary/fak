//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
int fak_qwen35_vllm_qk_capability(int nq, int nk, int hd, int rotary);
int fak_qwen35_vllm_qk_f32(
    const float *Q, const float *K, const float *V,
    const float *QW, const float *KW, float *QOut,
    float *KRawCache, float *KCache, float *VCache,
    int row, int position, int nq, int nk, int hd, int rotary,
    float eps, double theta);
unsigned long long fak_qwen35_vllm_qk_launch_count(void);
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"runtime"
)

// Read once. This is a source-delivered numerical candidate, not a qualified
// default or a claim that the donor's half-precision kernel matches F32 bits.
var cudaQwen35VLLMQKEnabled = os.Getenv("FAK_CUDA_QWEN35_VLLM_QK") == "1"

// AppendQKNormRoPE optionally replaces the ordinary single-token QK norm, RoPE
// and KV append. False/nil means no candidate effects. A selected error is fatal
// to the forward; the caller must not retry the ordinary append.
//
// Only existing owned F32 capacity is admitted. Initial and growth-boundary
// tokens use AppendKV, which remains the sole owner of allocation/growth policy.
func (k *cudaKV) AppendQKNormRoPE(q, key, value, qw, kw Tensor, layer, position, heads, rotary int, eps float32, theta float64) (Tensor, bool, error) {
	if !cudaQwen35VLLMQKEnabled {
		return Tensor{}, false, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	defer runtime.KeepAlive([5]Tensor{q, key, value, qw, kw})
	fail := func(reason string) (Tensor, bool, error) {
		return Tensor{}, true, fmt.Errorf("cuda: Qwen vLLM QK layer %d: %s", layer, reason)
	}
	if k == nil || k.be == nil || k.be != cudaDev {
		return fail("KV cache is not owned by the registered CUDA backend")
	}
	c := k.be
	if err := c.faultLatch.Admit("qwen35-vllm-qk"); err != nil {
		return Tensor{}, true, err
	}
	if c.capturing {
		return Tensor{}, false, nil
	}
	if layer < 0 || layer >= len(k.Kraw) || layer >= len(k.K) || layer >= len(k.V) {
		return fail("compact attention layer is outside the KV cache")
	}
	nk, hd := k.cfg.NumKVHeads, k.cfg.HeadDim
	if nk <= 0 || heads <= 0 || heads%nk != 0 ||
		(hd != 64 && hd != 128 && hd != 192 && hd != 256) ||
		heads > math.MaxInt32/hd || nk > math.MaxInt32/hd ||
		int64(heads)+2*int64(nk) > math.MaxInt32 ||
		rotary < 0 || rotary > hd || rotary%8 != 0 {
		return Tensor{}, false, nil
	}
	if position < 0 || position > math.MaxInt32 || !(eps > 0) ||
		math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) ||
		!(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return fail("invalid absolute position, RMS epsilon or RoPE theta")
	}
	for _, plane := range []dslice{k.Kraw[layer], k.K[layer], k.V[layer]} {
		if plane.dtype != F32 {
			return Tensor{}, false, nil
		}
	}
	width := nk * hd
	frontier := k.Kraw[layer].len
	if frontier < 0 || frontier > math.MaxInt32-width || frontier%width != 0 ||
		k.K[layer].len != frontier || k.V[layer].len != frontier {
		return fail("cache planes do not have one valid append frontier")
	}
	row := frontier / width
	wantRows := len(k.pos)
	if layer != 0 {
		wantRows--
	}
	if row != wantRows {
		return fail("layer frontier does not match the published position sequence")
	}
	for _, plane := range []dslice{k.Kraw[layer], k.K[layer], k.V[layer]} {
		if plane.cap < frontier+width || plane.ptr == nil {
			return Tensor{}, false, nil
		}
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		n    int
	}{
		{"query", q, heads * hd}, {"key", key, width}, {"value", value, width},
		{"query gain", qw, hd}, {"key gain", kw, hd},
	} {
		if err := c.validateQwen35SequenceTensor(operand.name, operand.t, false, operand.n); err != nil {
			return Tensor{}, true, err
		}
	}
	capability := int(C.fak_qwen35_vllm_qk_capability(C.int(heads), C.int(nk), C.int(hd), C.int(rotary)))
	if capability == 0 {
		return Tensor{}, false, nil
	}
	if capability != 1 {
		c.faultLatch.Observe(DeviceFaultExecution, "qwen35-vllm-qk-admission", capability)
		return fail(fmt.Sprintf("native admission status %d", capability))
	}
	out, _, err := c.devTrDeviceOnly([]int{heads * hd}, F32, "qwen35-vllm-qk-query")
	if err != nil {
		return Tensor{}, true, err
	}
	status := int(C.fak_qwen35_vllm_qk_f32(c.cf(q), c.cf(key), c.cf(value), c.cf(qw), c.cf(kw), c.cf(out),
		(*C.float)(k.Kraw[layer].ptr), (*C.float)(k.K[layer].ptr), (*C.float)(k.V[layer].ptr),
		C.int(row), C.int(position), C.int(heads), C.int(nk), C.int(hd), C.int(rotary), C.float(eps), C.double(theta)))
	if status != 0 {
		// The native launcher drains a failed submission and closes the existing
		// cleanup latch if completion is uncertain. Keep output ownership in the
		// ordinary transient lifecycle; never release or replay it here.
		if status > 0 {
			c.faultLatch.Observe(DeviceFaultExecution, "qwen35-vllm-qk-submit", status)
		}
		return fail(fmt.Sprintf("native submission status %d", status))
	}
	// As in AppendKV, same-stream writes precede host metadata publication.
	// The ordinary final logits read owns completion; this is not a new fence.
	k.Kraw[layer].len = frontier + width
	k.K[layer].len = frontier + width
	k.V[layer].len = frontier + width
	if layer == 0 {
		k.pos = append(k.pos, position)
	}
	return out, true, nil
}

func cudaQwen35VLLMQKLaunches() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fak_qwen35_vllm_qk_launch_count())
}
