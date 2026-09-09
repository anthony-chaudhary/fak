package hil

import (
	"runtime"
)

// ProbeHardware discovers the physical compute capabilities of the local host.
func ProbeHardware() HardwareInfo {
	return probePlatformHardware()
}

func baseHardwareInfo() HardwareInfo {
	return HardwareInfo{
		Kind:              HardwareUnknown,
		DeviceName:        "Generic Host",
		Architecture:      runtime.GOARCH,
		Platform:          runtime.GOOS,
		MemoryTotalBytes:  0,
		MemoryUnified:     false,
		PhysicalAvailable: false,
		Details:           make(map[string]string),
	}
}
