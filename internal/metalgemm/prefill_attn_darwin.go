//go:build darwin && arm64 && cgo

package metalgemm

/*
int mg_prefill_attn_supported(int hd, int nH, int nKV);
int mg_prefill_attn(const float* q, const float* k, const float* v, float* out,
                    int P, int kv_len, int nH, int nKV, int hd, int window, float scale,
                    double* out_gpu_ms, double* out_wait_ms);
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// prefillAttnMu serializes PrefillAttention: the native side reuses one set of scratch buffers.
var prefillAttnMu sync.Mutex

// PrefillAttentionSupported reports whether the tiled GQA prefill attention kernel (fak#13695)
// has a compiled pipeline for this head geometry (head_dim 64/128/256, nH a multiple of nKV).
func PrefillAttentionSupported(headDim, nH, nKV int) bool {
	if !Available() {
		return false
	}
	return C.mg_prefill_attn_supported(C.int(headDim), C.int(nH), C.int(nKV)) != 0
}

// PrefillAttentionTiming is the execution receipt of one PrefillAttention call.
type PrefillAttentionTiming struct {
	GPUMilliseconds  float64 // cb.GPUEndTime-cb.GPUStartTime
	WaitMilliseconds float64 // host commit-to-completion wait (bounded)
}

// PrefillAttention computes causal GQA attention for a prefill panel of P query rows on the GPU,
// the device twin of the host attnPrefillInto: out [P, nH*hd] receives softmax(scale*q.k^T)v per
// (row, head), where q is [P, nH*hd] and k/v are the layer's whole KV cache [kvLen, nKV*hd] with the
// panel's own keys last (kvLen = cached + P). Query row t sits at absolute position kvLen-P+t and
// attends keys (t-window, t] when window > 0, else every key <= t. It returns an error, leaving out
// untouched, for an unsupported geometry, an allocation failure, or a command buffer that did not
// complete within the bounded wait.
func PrefillAttention(out, q, k, v []float32, P, kvLen, nH, nKV, headDim, window int, scale float32) (PrefillAttentionTiming, error) {
	var timing PrefillAttentionTiming
	if P <= 0 || kvLen < P || nH <= 0 || nKV <= 0 || headDim <= 0 ||
		len(q) < P*nH*headDim || len(out) < P*nH*headDim ||
		len(k) < kvLen*nKV*headDim || len(v) < kvLen*nKV*headDim {
		return timing, fmt.Errorf("metalgemm: invalid prefill attention shape P=%d kvLen=%d nH=%d nKV=%d hd=%d", P, kvLen, nH, nKV, headDim)
	}
	var gpuMs, waitMs C.double
	prefillAttnMu.Lock()
	rc := C.mg_prefill_attn((*C.float)(unsafe.Pointer(&q[0])), (*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])), (*C.float)(unsafe.Pointer(&out[0])),
		C.int(P), C.int(kvLen), C.int(nH), C.int(nKV), C.int(headDim), C.int(window), C.float(scale),
		&gpuMs, &waitMs)
	prefillAttnMu.Unlock()
	timing = PrefillAttentionTiming{GPUMilliseconds: float64(gpuMs), WaitMilliseconds: float64(waitMs)}
	if rc == -10 {
		return timing, commandBufferStallError(timing.WaitMilliseconds, 0, 0, "", "prefill attention")
	}
	if rc != 0 {
		return timing, fmt.Errorf("metalgemm: prefill attention failed (code %d)", int(rc))
	}
	return timing, nil
}
