//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdint.h>
int fvk_debug_gdn_prefill_tiled_available(void);
void fvk_debug_gdn_prefill_tiled_mode(int mode);
void fvk_debug_gdn_prefill_tiled_reset(void);
void fvk_debug_gdn_prefill_tiled_snapshot(uint64_t* tiled, uint64_t* scalar);
// Issue-local adapter: the verify-width self-check ABI lands beside its only Go
// consumer before it widens the shared header.
int fvk_debug_gdn_verify_tiled_available(void);
void fvk_debug_gdn_verify_tiled_mode(int mode);
void fvk_debug_gdn_verify_tiled_reset(void);
void fvk_debug_gdn_verify_tiled_snapshot(uint64_t* verify, uint64_t* scalar);
int fvk_debug_gdn_verify_tiled_verdict(float* deviation);
int fvk_debug_gdn_tiled_verdict(float* deviation);
int fvk_debug_gdn_verify_tiled_checked(void);
void fvk_debug_gdn_verify_tiled_force_disagree(int enabled);
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

// VulkanDebugGDNVerifyTiledAvailable reports whether the verify-width (1..7
// token) register-resident pipeline built on this device.
func (v *vulkanBackend) VulkanDebugGDNVerifyTiledAvailable() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_debug_gdn_verify_tiled_available() != 0
}

// VulkanDebugSetGDNVerifyTiledMode selects -1 for the verdict-admitted policy,
// 0 to force the stock kernel, or 1 to request the candidate (which still
// requires a true verdict). Process-wide, like the native Vulkan device.
func (v *vulkanBackend) VulkanDebugSetGDNVerifyTiledMode(mode int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_debug_gdn_verify_tiled_mode(C.int(mode))
}

// VulkanDebugResetGDNVerifyTiledProfile resets the process-wide verify-width
// route counter. The stock counter is shared with the tiled profile.
func (v *vulkanBackend) VulkanDebugResetGDNVerifyTiledProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_debug_gdn_verify_tiled_reset()
}

// VulkanDebugGDNVerifyTiledProfileSnapshot returns the verify-width candidate
// dispatch count and the shared stock-kernel dispatch count.
func (v *vulkanBackend) VulkanDebugGDNVerifyTiledProfileSnapshot() (verifyCalls, scalarCalls int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var verify, scalar C.uint64_t
	C.fvk_debug_gdn_verify_tiled_snapshot(&verify, &scalar)
	return int64(verify), int64(scalar)
}

// VulkanDebugGDNVerifyTiledVerdict returns the cached on-device self-check
// verdict and the measured normwise max relative deviation. checked is false
// only if the self-check has never run.
func (v *vulkanBackend) VulkanDebugGDNVerifyTiledVerdict() (admitted bool, deviation float32, checked bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var dev C.float
	admitted = C.fvk_debug_gdn_verify_tiled_verdict(&dev) != 0
	checked = C.fvk_debug_gdn_verify_tiled_checked() != 0
	return admitted, float32(dev), checked
}

// VulkanDebugSetGDNVerifyTiledForceDisagree installs a forced-disagreement hook
// for tests: the self-check re-runs and reports failure, so the stock route must
// be selected and the verify-width counter must stay at zero.
func (v *vulkanBackend) VulkanDebugSetGDNVerifyTiledForceDisagree(enabled bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	flag := C.int(0)
	if enabled {
		flag = 1
	}
	C.fvk_debug_gdn_verify_tiled_force_disagree(flag)
}

// VulkanDebugGDNTiledVerdict returns the cached self-check verdict and measured
// normwise max relative deviation for the prefill-tiled class (>=8 tokens),
// checked in the same pre-forward self-check as the verify-width class.
func (v *vulkanBackend) VulkanDebugGDNTiledVerdict() (admitted bool, deviation float32) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var dev C.float
	admitted = C.fvk_debug_gdn_tiled_verdict(&dev) != 0
	return admitted, float32(dev)
}
