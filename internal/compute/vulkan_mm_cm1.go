//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

// mmCM1TryLocked records Y = W X for tokens rows through the prefill GEMM over the packed
// weight at wp, or returns the sentinel-wrapped reason it did not; a refusal records nothing.
func (v *vulkanBackend) mmCM1TryLocked(cfg VulkanMMCM1Config, dt Dtype, wp unsafe.Pointer, x, y Tensor, out, in, tokens int) error {
	if err := vulkanMMCM1Decide(cfg, dt, tokens); err != nil {
		return err
	}
	f := vulkanMMCM1Formats[dt]
	return vulkanMMCM1StatusError(int(C.fvk_mm_cm1_f32(C.int(f.id), wp, v.vp(x), v.vp(y),
		C.int(out), C.int(in), C.int(tokens))))
}

// mmCM1MatMulLocked reports whether the prefill GEMM served the call; on false the caller
// runs fallback, and the reason has been counted (and announced once for device refusals).
func (v *vulkanBackend) mmCM1MatMulLocked(w, x, y Tensor, out, in, tokens int, fallback string) bool {
	return v.mmCM1MatMulPtrLocked(w.Dtype, v.vp(w), x, y, out, in, tokens, fallback)
}

func (v *vulkanBackend) mmCM1MatMulPtrLocked(dt Dtype, wp unsafe.Pointer, x, y Tensor, out, in, tokens int, fallback string) bool {
	err := v.mmCM1TryLocked(vulkanMMCM1CurrentConfig(), dt, wp, x, y, out, in, tokens)
	recordVulkanMMCM1(dt, err, fallback)
	return err == nil
}

// vulkanMMCM1DeviceStatus returns nil when dt's GEMM kernel is servable on this device.
func vulkanMMCM1DeviceStatus(dt Dtype) error {
	f, ok := vulkanMMCM1Formats[dt]
	if !ok {
		return ErrVulkanMMCM1UnsupportedDtype
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanMMCM1StatusError(int(C.fvk_mm_cm1_status(C.int(f.id))))
}
