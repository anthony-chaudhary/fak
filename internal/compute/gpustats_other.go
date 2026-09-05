//go:build !darwin

package compute

// AppleSiliconGPUStats returns (nil, false) on non-Darwin platforms.
func AppleSiliconGPUStats() (stats []GPUStat, present bool) {
	return nil, false
}
