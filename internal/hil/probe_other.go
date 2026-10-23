//go:build !(darwin && arm64 && cgo)

package hil

import (
	"os"
	"runtime"
	"strconv"
	"strings"
)

func probePlatformHardware() HardwareInfo {
	info := baseHardwareInfo()
	info.Platform = runtime.GOOS
	info.Architecture = runtime.GOARCH

	// This build executes only runHostReferenceMicroDose. Environment settings
	// configure potential accelerators; they do not prove discovery or execution.
	info.Kind = HardwareCPUSIMD
	info.DeviceName = "Host CPU (Reference SIMD)"
	info.PhysicalAvailable = false
	info.Details["runtime_num_cpu"] = strconv.Itoa(runtime.NumCPU())
	info.Details["execution_mode"] = "host_cpu_reference"
	// Literal reads keep each key visible to the envconfiglint freeze ledger.
	for _, hint := range []struct{ key, value string }{
		{"CUDA_VISIBLE_DEVICES", os.Getenv("CUDA_VISIBLE_DEVICES")},
		{"FAK_CUDA", os.Getenv("FAK_CUDA")},
		{"VK_ICD_FILENAMES", os.Getenv("VK_ICD_FILENAMES")},
		{"FAK_VULKAN", os.Getenv("FAK_VULKAN")},
	} {
		if hint.value != "" {
			info.Details["configuration_hint_"+strings.ToLower(hint.key)] = hint.value
		}
	}
	return info
}
