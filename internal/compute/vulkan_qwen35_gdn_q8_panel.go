//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// tryQwen35GDNQ8PanelLocked is the sequence-only four-projection candidate.
// Its separate native capability is opt-in at backend initialization; it never
// widens the decode selector or substitutes Q8 activation arithmetic for the
// cooperative-matrix path. The caller holds vulkanMu and owns x and the weights.
// Refusal is mutation-free; a selected operation's error must not replay work.
func (v *vulkanBackend) tryQwen35GDNQ8PanelLocked(
	x, w0, w1, w2, w3 Tensor, tokens int,
) (y0, y1, y2, y3 Tensor, fused bool, err error) {
	const maxIndex = int64(1<<31 - 1)
	maxFloatElements := int64(^uint(0)>>1) / 4
	if !v.haveQ8GDNInProjPanel || !v.haveQ8 || tokens <= 1 || int64(tokens) > maxIndex || v.q8CooperativeMatrixActive(tokens) ||
		x.Dtype != F32 || len(x.Shape) != 2 || x.Shape[0] != tokens {
		return Tensor{}, Tensor{}, Tensor{}, Tensor{}, false, nil
	}
	in := x.Shape[1]
	xb, ok := x.buf.(*vulkanBuf)
	if !ok || xb == nil || xb.ptr == nil || in <= 0 || int64(in) > maxIndex || in%32 != 0 ||
		int64(tokens)*int64(in) > maxIndex || int64(tokens)*int64(in) > maxFloatElements {
		return Tensor{}, Tensor{}, Tensor{}, Tensor{}, false, nil
	}
	weights := [...]Tensor{w0, w1, w2, w3}
	var bufs [4]*vulkanBuf
	var totalOut int64
	for i, weight := range weights {
		buf, ok := weight.buf.(*vulkanBuf)
		if !ok || buf == nil || weight.Dtype != Q8_0 || weight.Quant == nil || weight.Quant.Block != 32 ||
			len(weight.Shape) != 2 || weight.Shape[1] != in || weight.Shape[0] <= 0 ||
			int64(weight.Shape[0]) > maxIndex || buf.ptr == nil || buf.scalePtr == nil || len(buf.q8Chunks) != 0 {
			return Tensor{}, Tensor{}, Tensor{}, Tensor{}, false, nil
		}
		out := int64(weight.Shape[0])
		if out*int64(in) > maxIndex || out*int64(tokens) > maxIndex || out*int64(tokens) > maxFloatElements {
			return Tensor{}, Tensor{}, Tensor{}, Tensor{}, false, nil
		}
		totalOut += out
		bufs[i] = buf
	}
	if totalOut > maxIndex-7 || C.fvk_qwen35_gdn_q8_in_proj_panel_supported(
		C.int(w0.Shape[0]), C.int(w1.Shape[0]), C.int(w2.Shape[0]), C.int(w3.Shape[0]), C.int(in), C.int(tokens),
	) == 0 {
		return Tensor{}, Tensor{}, Tensor{}, Tensor{}, false, nil
	}

	start := len(v.transient)
	y0, _ = v.devTr([]int{tokens, w0.Shape[0]}, F32)
	y1, _ = v.devTr([]int{tokens, w1.Shape[0]}, F32)
	y2, _ = v.devTr([]int{tokens, w2.Shape[0]}, F32)
	y3, _ = v.devTr([]int{tokens, w3.Shape[0]}, F32)
	status := int(C.fvk_qwen35_gdn_q8_in_proj_panel_f32(
		bufs[0].ptr, bufs[0].scalePtr, bufs[1].ptr, bufs[1].scalePtr,
		bufs[2].ptr, bufs[2].scalePtr, bufs[3].ptr, bufs[3].scalePtr,
		xb.ptr, v.vp(y0), v.vp(y1), v.vp(y2), v.vp(y3),
		C.int(w0.Shape[0]), C.int(w1.Shape[0]), C.int(w2.Shape[0]), C.int(w3.Shape[0]), C.int(in), C.int(tokens),
	))
	if status != 0 {
		v.qwen35VulkanSequenceReleaseLocked(start)
		return Tensor{}, Tensor{}, Tensor{}, Tensor{}, false,
			fmt.Errorf("compute: Vulkan Qwen GDN Q8 panel projection failed closed (code %d)", status)
	}
	return y0, y1, y2, y3, true, nil
}
