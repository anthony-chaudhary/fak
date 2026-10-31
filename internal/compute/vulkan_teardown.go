package compute

import "fmt"

// vulkanTeardownAfterFlush keeps Go ownership intact when the native fence did
// not establish completion. The caller holds vulkanMu across the plain flush,
// sticky-status read and release; this gate does not reset or retry a fault.
// A successful gate preserves the existing release path, not a guarantee that
// every later native free or trim succeeds.
func vulkanTeardownAfterFlush(status int, release func()) error {
	if status != 0 {
		class, cause := VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
		if status == -4 { // VK_ERROR_DEVICE_LOST
			class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
		}
		return &BackendError{
			Backend: "vulkan", Site: "TeardownResources", Class: class, Err: cause,
			Message: fmt.Sprintf("native submission fault %d; teardown ownership retained", status),
		}
	}
	release()
	return nil
}
