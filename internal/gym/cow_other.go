//go:build !linux && !darwin && !windows

package gym

// newOSOverlay creates a fallback userspace overlay on other platforms.
func newOSOverlay(lowerDir, tempBase string) (CoWOverlay, error) {
	return newUserspaceOverlay(lowerDir, tempBase)
}
