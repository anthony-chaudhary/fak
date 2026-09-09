//go:build !(darwin && arm64 && cgo)

package hil

import (
	"os"
	"runtime"
	"strconv"
)

func probePlatformHardware() HardwareInfo {
	info := baseHardwareInfo()
	info.Platform = runtime.GOOS
	info.Architecture = runtime.GOARCH

	// Check CUDA presence via environment or driver files
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("FAK_CUDA") == "1" {
		info.Kind = HardwareCUDA
		info.DeviceName = "NVIDIA CUDA Device (Environment Configured)"
		info.PhysicalAvailable = true
		info.Details["cuda_visible_devices"] = os.Getenv("CUDA_VISIBLE_DEVICES")
		return info
	}

	// Check Vulkan presence via environment
	if os.Getenv("VK_ICD_FILENAMES") != "" || os.Getenv("FAK_VULKAN") == "1" {
		info.Kind = HardwareVulkan
		info.DeviceName = "Vulkan Compute Device"
		info.PhysicalAvailable = true
		return info
	}

	// Fallback to host CPU SIMD
	info.Kind = HardwareCPUSIMD
	info.DeviceName = "Host CPU (Reference SIMD)"
	info.PhysicalAvailable = false
	info.Details["runtime_num_cpu"] = strconv.Itoa(runtime.NumCPU())
	return info
}
