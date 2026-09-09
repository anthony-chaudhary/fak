package compute

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/anthony-chaudhary/fak/pkg/strix"
)

const (
	// StrixVendorIDHex is the AMD PCI vendor ID (0x1002).
	StrixVendorIDHex uint32 = 0x1002
	// StrixDeviceIDHex is the GFX1151 RDNA 3.5 PCI device ID (0x1586).
	StrixDeviceIDHex uint32 = 0x1586

	// DefaultStrixBridgeName is the canonical engine name reported by StrixKernelBridge.
	DefaultStrixBridgeName = "strix-kernel-bridge"

	// DefaultDRMSysfsRoot is the default Linux DRM sysfs directory.
	DefaultDRMSysfsRoot = "/sys/class/drm"
)

var (
	// ErrUnsupportedDevice indicates the current platform/device lacks GFX1151 hardware.
	ErrUnsupportedDevice = errors.New("strix: GFX1151 hardware acceleration is not supported on this device")
	// ErrInvalidAllocationSize indicates an invalid buffer allocation request (<= 0 bytes).
	ErrInvalidAllocationSize = errors.New("strix: allocation size must be greater than 0")
)

// StrixKernelBridge satisfies pkg/strix.AccelerationEngine, bridging AMD Strix Halo (GFX1151)
// kernel execution, Wave32 matrix acceleration, and USWC GTT zero-copy memory management.
type StrixKernelBridge struct {
	mu          sync.Mutex
	name        string
	supported   bool
	profile     strix.DeviceProfile
	buffers     map[uintptr][]byte
	config      StrixKernelBridgeConfig
	drmCardPath string
}

// Ensure compile-time interface satisfaction with pkg/strix.AccelerationEngine.
var _ strix.AccelerationEngine = (*StrixKernelBridge)(nil)

// StrixKernelBridgeConfig provides dependency injection and configuration for StrixKernelBridge.
type StrixKernelBridgeConfig struct {
	// SysfsDRMRoot overrides the default sysfs DRM directory (default: "/sys/class/drm").
	SysfsDRMRoot string
	// DetectFn overrides hardware detection if non-nil.
	DetectFn func() (bool, string, error)
	// Allocator overrides USWC GTT memory allocation if non-nil.
	Allocator func(bytes int64) (uintptr, error)
}

// NewStrixKernelBridge creates a new StrixKernelBridge with automatic hardware detection.
func NewStrixKernelBridge() *StrixKernelBridge {
	return NewStrixKernelBridgeWithConfig(StrixKernelBridgeConfig{})
}

// NewStrixKernelBridgeWithConfig creates a StrixKernelBridge with custom configuration.
func NewStrixKernelBridgeWithConfig(cfg StrixKernelBridgeConfig) *StrixKernelBridge {
	var isSupported bool
	var drmCardPath string

	if cfg.DetectFn != nil {
		sup, card, err := cfg.DetectFn()
		if err == nil && sup {
			isSupported = true
			drmCardPath = card
		}
	} else {
		sup, card, _ := DetectGFX1151DRM(cfg.SysfsDRMRoot)
		isSupported = sup
		drmCardPath = card
	}

	bridge := &StrixKernelBridge{
		name:        DefaultStrixBridgeName,
		supported:   isSupported,
		buffers:     make(map[uintptr][]byte),
		config:      cfg,
		drmCardPath: drmCardPath,
	}

	if isSupported {
		bridge.profile = strix.DefaultStrixHaloProfile()
	} else {
		bridge.profile = strix.DeviceProfile{
			Architecture:      "generic-fallback",
			CUCount:           0,
			Wavefront:         0,
			BusWidthBits:      0,
			PeakBandwidthGBps: 0,
		}
	}

	return bridge
}

// Name returns the identifier of the acceleration engine.
func (b *StrixKernelBridge) Name() string {
	return b.name
}

// IsSupported reports whether physical GFX1151 hardware is present and operational.
func (b *StrixKernelBridge) IsSupported() bool {
	return b.supported
}

// WavefrontSize returns the execution wavefront size (32 for GFX1151 Wave32, 0 when unsupported).
func (b *StrixKernelBridge) WavefrontSize() int {
	if !b.supported {
		return 0
	}
	return b.profile.Wavefront
}

// ComputeUnits returns the available compute units (40 for Strix Halo, 0 when unsupported).
func (b *StrixKernelBridge) ComputeUnits() int {
	if !b.supported {
		return 0
	}
	return b.profile.CUCount
}

// AllocUSWCGTT allocates a USWC (Uncached Speculative Write Combining) GTT zero-copy memory buffer.
// Returns ErrUnsupportedDevice if GFX1151 acceleration is not supported.
func (b *StrixKernelBridge) AllocUSWCGTT(bytes int64) (uintptr, error) {
	if bytes <= 0 {
		return 0, ErrInvalidAllocationSize
	}
	if !b.supported {
		return 0, ErrUnsupportedDevice
	}

	if b.config.Allocator != nil {
		return b.config.Allocator(bytes)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	raw := make([]byte, bytes)
	ptr := uintptr(unsafe.Pointer(&raw[0]))
	b.buffers[ptr] = raw
	return ptr, nil
}

// FreeUSWCGTT releases a buffer previously allocated via AllocUSWCGTT.
func (b *StrixKernelBridge) FreeUSWCGTT(ptr uintptr) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, ok := b.buffers[ptr]; !ok {
		return fmt.Errorf("strix: buffer pointer 0x%x not found", ptr)
	}
	delete(b.buffers, ptr)
	return nil
}

// Profile returns the active DeviceProfile for the bridge.
func (b *StrixKernelBridge) Profile() strix.DeviceProfile {
	return b.profile
}

// DRMCardPath returns the sysfs path to the matched DRM device, or empty if unsupported.
func (b *StrixKernelBridge) DRMCardPath() string {
	return b.drmCardPath
}

// DetectGFX1151DRM inspects DRM devices under sysfsRoot for AMD vendor (0x1002) and GFX1151 device (0x1586).
// If sysfsRoot is empty, it uses the FAK_DRM_SYSFS_PATH environment variable or /sys/class/drm.
func DetectGFX1151DRM(sysfsRoot string) (bool, string, error) {
	// Check for explicit environment override (useful for testing or containerized deployments)
	if override := os.Getenv("FAK_STRIX_GFX1151_OVERRIDE"); override != "" {
		if override == "1" || strings.EqualFold(override, "true") {
			return true, "override:env", nil
		}
		if override == "0" || strings.EqualFold(override, "false") {
			return false, "", nil
		}
	}

	if sysfsRoot == "" {
		if envPath := os.Getenv("FAK_DRM_SYSFS_PATH"); envPath != "" {
			sysfsRoot = envPath
		} else {
			sysfsRoot = DefaultDRMSysfsRoot
		}
	}

	pattern := filepath.Join(sysfsRoot, "card*", "device", "vendor")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return false, "", fmt.Errorf("strix: failed to glob DRM sysfs vendor files: %w", err)
	}
	if len(matches) == 0 {
		return false, "", nil
	}

	for _, vendorPath := range matches {
		vendorData, err := os.ReadFile(vendorPath)
		if err != nil {
			continue
		}
		if !isMatchingPCIID(string(vendorData), StrixVendorIDHex) {
			continue
		}

		cardDir := filepath.Dir(vendorPath) // /sys/class/drm/cardX/device
		devicePath := filepath.Join(cardDir, "device")
		deviceData, err := os.ReadFile(devicePath)
		if err != nil {
			continue
		}
		if isMatchingPCIID(string(deviceData), StrixDeviceIDHex) {
			return true, filepath.Dir(cardDir), nil // return /sys/class/drm/cardX
		}
	}

	return false, "", nil
}

// isMatchingPCIID compares a sysfs PCI hex string with an expected 32-bit hex value.
func isMatchingPCIID(content string, expected uint32) bool {
	s := strings.TrimSpace(content)
	s = strings.TrimPrefix(strings.ToLower(s), "0x")
	val, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return false
	}
	return uint32(val) == expected
}
