// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	// VendorIDAMD is the AMD PCI vendor ID (0x1002).
	VendorIDAMD uint32 = 0x1002

	// DeviceIDGFX1151 is the GFX1151 RDNA 3.5 PCI device ID (0x1586).
	DeviceIDGFX1151 uint32 = 0x1586

	// DefaultDRMSysfsRoot is the default Linux DRM sysfs directory.
	DefaultDRMSysfsRoot = "/sys/class/drm"

	// CanonicalDeviceNameStrixHalo is the standard display name for Strix Halo APU silicon.
	CanonicalDeviceNameStrixHalo = "AMD Ryzen AI Max+ 395 (GFX1151)"
)

// IsStrixHaloArch reports whether arch or deviceName indicates AMD Strix Halo silicon (GFX1151).
func IsStrixHaloArch(arch string) bool {
	lower := strings.ToLower(strings.TrimSpace(arch))
	if lower == "" {
		return false
	}
	return strings.Contains(lower, "gfx1151") ||
		strings.Contains(lower, "strix halo") ||
		strings.Contains(lower, "strix-halo") ||
		strings.Contains(lower, "strix_halo") ||
		strings.Contains(lower, "ryzen ai max") ||
		strings.Contains(lower, "8060s") ||
		strings.Contains(lower, "8050s")
}

// DetectGFX1151 checks whether AMD Strix Halo GFX1151 silicon is present via environment override
// or Linux DRM sysfs inspection.
func DetectGFX1151(sysfsRoot string) (bool, string, error) {
	return DetectGFX1151WithDeviceName(sysfsRoot, "")
}

// DetectGFX1151WithDeviceName checks environment override, device name string, and DRM sysfs.
func DetectGFX1151WithDeviceName(sysfsRoot, deviceName string) (bool, string, error) {
	// 1. Environment override (highest priority)
	if override := os.Getenv("FAK_STRIX_GFX1151_OVERRIDE"); override != "" {
		trimmed := strings.TrimSpace(override)
		if trimmed == "1" || strings.EqualFold(trimmed, "true") {
			return true, "AMD Ryzen AI Max+ 395 (GFX1151 override)", nil
		}
		if trimmed == "0" || strings.EqualFold(trimmed, "false") {
			return false, "", nil
		}
	}

	// 2. Device / backend name string check
	if deviceName != "" && IsStrixHaloArch(deviceName) {
		return true, deviceName, nil
	}

	// 3. DRM sysfs detection
	if sysfsRoot == "" {
		if envPath := os.Getenv("FAK_DRM_SYSFS_PATH"); envPath != "" {
			sysfsRoot = envPath
		} else {
			sysfsRoot = DefaultDRMSysfsRoot
		}
	}

	// Direct check if sysfsRoot itself has vendor & device (e.g. /sys/class/drm/card0/device)
	if vendorData, err := os.ReadFile(filepath.Join(sysfsRoot, "vendor")); err == nil {
		if isMatchingPCIID(string(vendorData), VendorIDAMD) {
			if deviceData, err := os.ReadFile(filepath.Join(sysfsRoot, "device")); err == nil {
				if isMatchingPCIID(string(deviceData), DeviceIDGFX1151) {
					return true, CanonicalDeviceNameStrixHalo, nil
				}
			}
		}
	}

	// Glob scan sysfsRoot/card*/device/vendor
	pattern := filepath.Join(sysfsRoot, "card*", "device", "vendor")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return false, "", fmt.Errorf("strix: failed to glob DRM sysfs vendor files: %w", err)
	}

	for _, vendorPath := range matches {
		vendorData, err := os.ReadFile(vendorPath)
		if err != nil {
			continue
		}
		if !isMatchingPCIID(string(vendorData), VendorIDAMD) {
			continue
		}

		cardDeviceDir := filepath.Dir(vendorPath) // .../cardX/device
		devicePath := filepath.Join(cardDeviceDir, "device")
		deviceData, err := os.ReadFile(devicePath)
		if err != nil {
			continue
		}
		if isMatchingPCIID(string(deviceData), DeviceIDGFX1151) {
			return true, CanonicalDeviceNameStrixHalo, nil
		}
	}

	return false, "", nil
}

func isMatchingPCIID(content string, expected uint32) bool {
	s := strings.TrimSpace(content)
	s = strings.TrimPrefix(strings.ToLower(s), "0x")
	val, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return false
	}
	return uint32(val) == expected
}
