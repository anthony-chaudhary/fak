//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"
import "unsafe"

func (v *vulkanBackend) v41TryPoolNormPanel(original Backend, gain Tensor, owned []float32, rows, width int, eps float32) (bool, error) {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil {
		return false, nil
	}
	if rows < 2 || rows > 64 || width <= 0 || width > 2147483647 || width > int(^uint(0)>>1)/(4*rows) || len(owned) != rows*width {
		return false, nil
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	requireVulkanSubmissionHealthyLocked("FFN panel transport")
	b, ok := gain.buf.(*vulkanBuf)
	if !ok || b == nil || b.ptr == nil || gain.Backend() != v || gain.Dtype != F32 || len(gain.Shape) != 1 || gain.Shape[0] != width || gain.Numel() != width {
		return false, nil
	}
	var status C.int
	selected := int(C.fvk_v41_try_pool_norm_panel((*C.float)(unsafe.Pointer(&owned[0])), b.ptr, C.int(rows), C.int(width), C.float(eps), &status))
	if selected == 0 && status == 0 {
		return false, nil
	}
	if selected != 1 || status != 0 {
		err := vulkanReadError(int(status))
		err.Site = "FFN panel transport"
		return true, err
	}
	return true, nil
}
