//go:build darwin && arm64 && cgo

package hil

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func probePlatformHardware() HardwareInfo {
	info := baseHardwareInfo()
	info.Platform = "darwin"
	info.Architecture = "arm64"

	// Read CPU model via sysctl
	if out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
		info.Details["cpu_brand"] = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("sysctl", "-n", "hw.model").Output(); err == nil {
		info.Details["hw_model"] = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("sysctl", "-n", "hw.ncpu").Output(); err == nil {
		info.Details["cpu_cores"] = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("sysctl", "-n", "hw.memsize").Output(); err == nil {
		var mem uint64
		if _, err := fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &mem); err == nil {
			info.MemoryTotalBytes = mem
			info.MemoryUnified = true
		}
	}

	// Probe Apple Silicon Metal device
	if metalgemm.Available() {
		info.Kind = HardwareMetal
		info.PhysicalAvailable = true
		devName := metalgemm.DeviceName()
		if devName != "" {
			info.DeviceName = devName
		} else {
			info.DeviceName = "Apple Silicon Metal Device"
		}
		if tot, ok := metalgemm.DeviceMemoryTotal(); ok && tot > 0 {
			info.Details["metal_working_set_bytes"] = fmt.Sprintf("%d", tot)
		}
		info.Details["mps_available"] = fmt.Sprintf("%t", metalgemm.MPSAvailable())
	} else {
		info.Kind = HardwareCPUSIMD
		info.PhysicalAvailable = false
		info.DeviceName = info.Details["cpu_brand"]
		if info.DeviceName == "" {
			info.DeviceName = "Apple Silicon Host (Metal Unavailable)"
		}
	}

	return info
}
