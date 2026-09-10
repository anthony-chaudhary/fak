//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdint.h>
int fvk_debug_gdn_prefill_tiled_available(void);
void fvk_debug_gdn_prefill_tiled_mode(int mode);
void fvk_debug_gdn_prefill_tiled_reset(void);
void fvk_debug_gdn_prefill_tiled_snapshot(uint64_t* tiled, uint64_t* scalar);
*/
import "C"

// VulkanDebugGDNTiledPrefillAvailable reports usable optional pipelines and the
// device's Wave32 shuffle, workgroup, and shared-memory capabilities. It does
// not establish qualification or select the candidate for a particular call.
func (v *vulkanBackend) VulkanDebugGDNTiledPrefillAvailable() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_debug_gdn_prefill_tiled_available() != 0
}

// VulkanDebugSetGDNTiledPrefillMode selects -1 for the normal default-off
// FAK_VULKAN_GDN_PREFILL_TILED policy, 0 for the original kernel, or 1 for the
// candidate on supported shapes. Device and shape gates apply in every mode.
// This debug override is process-wide, like the native Vulkan device.
func (v *vulkanBackend) VulkanDebugSetGDNTiledPrefillMode(mode int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_debug_gdn_prefill_tiled_mode(C.int(mode))
}

// VulkanDebugResetGDNTiledPrefillProfile resets the process-wide route counters.
func (v *vulkanBackend) VulkanDebugResetGDNTiledPrefillProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_debug_gdn_prefill_tiled_reset()
}

// VulkanDebugGDNTiledPrefillProfileSnapshot counts successfully recorded
// preprojected calls, not elapsed GPU work or full-model requests.
func (v *vulkanBackend) VulkanDebugGDNTiledPrefillProfileSnapshot() (tiledCalls, scalarCalls int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var tiled, scalar C.uint64_t
	C.fvk_debug_gdn_prefill_tiled_snapshot(&tiled, &scalar)
	return int64(tiled), int64(scalar)
}
