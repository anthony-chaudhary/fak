//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// tryQ5KPanelLocked is called only after the ordinary Q5_K dtype, backend,
// packed payload, shape, output ownership and legacy grid validation. It never
// broadens decode or V4.1's independent scalar shared-head selector.
func (v *vulkanBackend) tryQ5KPanelLocked(w, x, y Tensor, out, in, tokens int) bool {
	if !v.haveQ5KPanel || tokens < 16 ||
		C.fvk_q5k_matmul_panel_supported(C.int(out), C.int(in), C.int(tokens)) == 0 {
		return false
	}
	status := int(C.fvk_q5k_matmul_panel_f32(v.vp(w), v.vp(x), v.vp(y), C.int(out), C.int(in), C.int(tokens)))
	if status != 0 {
		panic(fmt.Sprintf("compute: Vulkan Q5_K panel failed closed (code %d)", status))
	}
	return true
}

// q5KPanelReferenceForTest keeps the independent ordinary scalar-panel oracle
// callable without mutating a global capability or widening production gates.
func (v *vulkanBackend) q5KPanelReferenceForTest(w, x Tensor, tokens int) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := v.validateQ5KMatMulInputs(w, x, tokens)
	y, _ := v.devTr([]int{tokens, out}, F32)
	C.fvk_q5k_matmul_f32(v.vp(w), v.vp(x), v.vp(y), C.int(out), C.int(in), C.int(tokens))
	return y
}
