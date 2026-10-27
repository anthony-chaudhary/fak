//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// tryQ6KPanelLocked is called only after the ordinary Q6_K dtype, backend,
// packed payload, shape, output ownership and legacy grid validation. It never
// broadens decode or V4.1's independent scalar shared-head selector.
func (v *vulkanBackend) tryQ6KPanelLocked(w, x, y Tensor, out, in, tokens int) bool {
	if !v.haveQ6KPanel || tokens < 16 ||
		C.fvk_q6k_matmul_panel_supported(C.int(out), C.int(in), C.int(tokens)) == 0 {
		return false
	}
	status := int(C.fvk_q6k_matmul_panel_f32(v.vp(w), v.vp(x), v.vp(y), C.int(out), C.int(in), C.int(tokens)))
	if status != 0 {
		panic(fmt.Sprintf("compute: Vulkan Q6_K panel failed closed (code %d)", status))
	}
	return true
}

// q6KPanelReferenceForTest keeps the independent ordinary scalar-panel oracle
// callable without mutating a global capability or widening production gates.
func (v *vulkanBackend) q6KPanelReferenceForTest(w, x Tensor, tokens int) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := v.validateQ6KMatMulInputs(w, x, tokens)
	y, _ := v.devTr([]int{tokens, out}, F32)
	C.fvk_q6k_matmul_f32(v.vp(w), v.vp(x), v.vp(y), C.int(out), C.int(in), C.int(tokens))
	return y
}
