//go:build wip_v41_shared_head_qualification && vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// VulkanQ6KSharedHeadQualification describes the selected device's necessary
// shape/resource conditions. It is not a scalar/panel parity qualification and
// deliberately does not implement V41SharedHeadAvailable.
type VulkanQ6KSharedHeadQualification struct {
	Device                                                            string
	VendorID, DeviceID, DriverVersion, APIVersion, DeviceType         uint32
	MaxGroupsX                                                        uint64
	MaxBufferBytes, MaxStorageBufferRange, MaxMemoryAllocationSize    int64
	PackedWeightBytes, MaxInputBytes, MaxOutputBytes, RequiredGroupsX uint64
	MaxRows                                                           int
}

// VulkanQ6KSharedHeadQualificationPreflight queries already-initialized state
// under the existing global command-stream mutex. It never allocates, uploads,
// submits, opens a recording window, or installs a capability provider.
// A successful result establishes necessary resource conditions only. Physical
// scalar/panel and model-level witnesses remain separate gates.
func (v *vulkanBackend) VulkanQ6KSharedHeadQualificationPreflight(dtype Dtype, out, in, maxRows int) (VulkanQ6KSharedHeadQualification, error) {
	r := VulkanQ6KSharedHeadQualification{}
	if v == nil || dtype != Q6_K || out != 129280 || in != 5120 || maxRows < 1 || maxRows > 4 {
		return r, fmt.Errorf("shared-head qualification requires Q6_K [129280,5120] and rows in [1,4]")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if v != vulkanDev || C.fvk_have_q6k_matmul() == 0 || C.fvk_submission_status() != 0 {
		return r, fmt.Errorf("shared-head qualification requires the healthy registered Q6_K Vulkan backend")
	}
	var name [256]C.char
	var vendor, device, driver, api C.uint32_t
	if C.fvk_device_identity(&name[0], 256, &vendor, &device, &driver, &api) == 0 {
		return r, fmt.Errorf("shared-head qualification physical device identity is unavailable")
	}
	r.Device = C.GoString(&name[0])
	r.VendorID, r.DeviceID, r.DriverVersion, r.APIVersion = uint32(vendor), uint32(device), uint32(driver), uint32(api)
	r.DeviceType = uint32(C.fvk_device_type())
	if r.VendorID != vulkanAMDVendorID || r.DeviceID == 0 || r.DeviceType != 1 {
		return r, fmt.Errorf("shared-head qualification requires an identified physical integrated AMD device; got %+v", r)
	}
	r.MaxRows = maxRows
	r.PackedWeightBytes = uint64(out) * uint64(in/256) * 210
	r.MaxInputBytes = uint64(maxRows) * uint64(in) * 4
	r.MaxOutputBytes = uint64(maxRows) * uint64(out) * 4
	r.RequiredGroupsX = (uint64(maxRows)*uint64(out) + 63) / 64
	r.MaxGroupsX = uint64(C.fvk_max_compute_work_group_count_x())
	r.MaxBufferBytes, r.MaxStorageBufferRange, r.MaxMemoryAllocationSize = v.maxBufferBytes, v.maxStorageBufferRange, v.maxMemoryAllocationSize
	if r.MaxGroupsX == 0 || r.RequiredGroupsX > r.MaxGroupsX {
		return r, fmt.Errorf("shared-head qualification X grid %d exceeds known device limit %d", r.RequiredGroupsX, r.MaxGroupsX)
	}
	// These exact dimensions satisfy the existing C-int ABI and uint32 shader
	// indexing checks. The packed payload is already four-byte aligned. Every
	// smaller survivor count needs fewer groups and no larger input/output.
	for _, limit := range []int64{r.MaxBufferBytes, r.MaxStorageBufferRange, r.MaxMemoryAllocationSize} {
		if limit <= 0 || r.PackedWeightBytes > uint64(limit) || r.MaxInputBytes > uint64(limit) || r.MaxOutputBytes > uint64(limit) {
			return r, fmt.Errorf("shared-head qualification requires known per-buffer limits covering all operands: %+v", r)
		}
	}
	return r, nil
}
