//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// tryQwen35Q8PairLocked reuses only the existing noncooperative single-window
// Q8 pair. It is sequence-private, default-off and never changes dtype admission,
// the public decode selector or the existing cooperative/chunked precision path.
func (v *vulkanBackend) tryQwen35Q8PairLocked(x, w0, w1 Tensor, tokens int) (y0, y1 Tensor, selected bool, err error) {
	const maxIndex = int64(1<<31 - 1)
	maxFloatElements := int64(^uint(0)>>1) / 4
	if !v.haveQ8SequencePair || !v.haveQ8 || tokens <= 1 || int64(tokens) > maxIndex || v.q8CooperativeMatrixActive(tokens) ||
		x.Dtype != F32 || len(x.Shape) != 2 || x.Shape[0] != tokens || x.Backend() != Backend(v) || !x.Ready() {
		return Tensor{}, Tensor{}, false, nil
	}
	in := x.Shape[1]
	xb, ok := x.buf.(*vulkanBuf)
	if !ok || xb == nil || xb.ptr == nil || in <= 0 || in > 2048 || in%32 != 0 ||
		int64(tokens)*int64(in) > maxIndex || int64(tokens)*int64(in) > maxFloatElements {
		return Tensor{}, Tensor{}, false, nil
	}
	weights := [...]Tensor{w0, w1}
	var bufs [2]*vulkanBuf
	var total int64
	for i, w := range weights {
		b, ok := w.buf.(*vulkanBuf)
		if !ok || b == nil || b.ptr == nil || b.scalePtr == nil || len(b.q8Chunks) != 0 ||
			w.Dtype != Q8_0 || w.Quant == nil || w.Quant.Block != 32 || len(w.Shape) != 2 || w.Shape[1] != in ||
			w.Shape[0] <= 0 || int64(w.Shape[0]) > maxIndex || w.Backend() != Backend(v) || !w.Ready() {
			return Tensor{}, Tensor{}, false, nil
		}
		out := int64(w.Shape[0])
		if out*int64(in) > maxIndex || out*int64(tokens) > maxIndex || out*int64(tokens) > maxFloatElements {
			return Tensor{}, Tensor{}, false, nil
		}
		total += out
		bufs[i] = b
	}
	if total > maxIndex || C.fvk_qwen35_q8_pair_supported(C.int(w0.Shape[0]), C.int(w1.Shape[0]), C.int(in), C.int(tokens)) == 0 {
		return Tensor{}, Tensor{}, false, nil
	}
	start := len(v.transient)
	y0, _ = v.devTr([]int{tokens, w0.Shape[0]}, F32)
	y1, _ = v.devTr([]int{tokens, w1.Shape[0]}, F32)
	status := int(C.fvk_qwen35_q8_pair_f32(bufs[0].ptr, bufs[0].scalePtr, bufs[1].ptr, bufs[1].scalePtr,
		xb.ptr, v.vp(y0), v.vp(y1), C.int(w0.Shape[0]), C.int(w1.Shape[0]), C.int(in), C.int(tokens)))
	if status != 0 {
		v.qwen35VulkanSequenceReleaseLocked(start)
		return Tensor{}, Tensor{}, false, fmt.Errorf("compute: Vulkan Qwen Q8 sequence pair failed closed (code %d)", status)
	}
	return y0, y1, true, nil
}
