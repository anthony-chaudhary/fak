//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// tryQwen35ResidualNormLocked is one sequence-only operation family for the
// post-branch residual Add -> PostNorm chain, irrespective of projection dtype.
// Refusal is mutation-free. Once selected, failure aborts the sequence because
// x may already have been updated; replay is never permitted. vulkanMu is held.
func (v *vulkanBackend) tryQwen35ResidualNormLocked(x, delta, weight Tensor, rows int, eps float32) (y Tensor, used bool, err error) {
	const maxIndex = int64(1<<31 - 1)
	if !v.haveQwenResidualNorm || rows <= 1 || int64(rows) > maxIndex || len(x.Shape) != 2 || x.Shape[0] != rows {
		return Tensor{}, false, nil
	}
	width := x.Shape[1]
	maxFloats := min(maxIndex, int64(^uint(0)>>1)/4)
	if width <= 0 || int64(width) > maxFloats || int64(rows) > maxFloats/int64(width) {
		return Tensor{}, false, nil
	}
	valid := func(t Tensor) bool {
		b, ok := t.buf.(*vulkanBuf)
		return ok && b != nil && b.ptr != nil && t.Dtype == F32 && t.Backend() == Backend(v)
	}
	if !valid(x) || !valid(delta) || !valid(weight) || len(delta.Shape) != 2 || delta.Shape[0] != rows || delta.Shape[1] != width ||
		len(weight.Shape) != 1 || weight.Shape[0] != width ||
		C.fvk_qwen35_residual_norm_supported(C.int(rows), C.int(width), C.float(eps)) == 0 {
		return Tensor{}, false, nil
	}
	start := len(v.transient)
	y, _ = v.devTr([]int{rows, width}, F32)
	status := int(C.fvk_qwen35_residual_norm_f32(v.vp(x), v.vp(delta), v.vp(weight), v.vp(y), C.int(rows), C.int(width), C.float(eps)))
	if status != 0 {
		v.qwen35VulkanSequenceReleaseLocked(start)
		return Tensor{}, false, fmt.Errorf("compute: Vulkan Qwen residual norm failed closed (code %d)", status)
	}
	return y, true, nil
}
