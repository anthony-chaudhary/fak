// Package strix defines the exported public hardware acceleration interfaces,
// device profiles, and capabilities for AMD Strix Halo (GFX1151) APUs.
//
// This package is part of the public fak SDK and has zero dependencies on
// proprietary private platform code, preserving Gate 2 encapsulation.
package strix

// AccelerationEngine defines the public hardware acceleration contract for AMD Strix Halo (GFX1151) APUs.
type AccelerationEngine interface {
	Name() string
	IsSupported() bool
	WavefrontSize() int
	ComputeUnits() int
	AllocUSWCGTT(bytes int64) (uintptr, error)
}

// DeviceProfile describes physical silicon specifications and bandwidth limits for GFX1151 compute units.
type DeviceProfile struct {
	Architecture      string // "gfx1151"
	CUCount           int    // 40
	Wavefront         int    // 32
	BusWidthBits      int    // 256
	PeakBandwidthGBps float64
}

// DefaultStrixHaloProfile returns the canonical hardware profile for AMD Strix Halo (GFX1151).
func DefaultStrixHaloProfile() DeviceProfile {
	return DeviceProfile{
		Architecture:      "gfx1151",
		CUCount:           40,
		Wavefront:         32,
		BusWidthBits:      256,
		PeakBandwidthGBps: 273.056,
	}
}
