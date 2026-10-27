//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

// These witness adapters do not enable the candidate or mutate the native proof.
func vulkanNativeHazardsEnabled() bool  { return C.fvk_native_batch_hazards_enabled() != 0 }
func vulkanNativeHazardsSelfCheck() int { return int(C.fvk_native_batch_hazards_self_check()) }

// The copy witness uses the real ordinary recorder; no synthetic ordinal update.
func (v *vulkanBackend) nativeHazardCopyForTest(dst, src Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_d2d(v.vp(dst), v.vp(src), C.size_t(src.Numel()*4))
}
