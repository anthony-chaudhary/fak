//go:build darwin

package gym

// newOSOverlay creates a Darwin APFS clonefile / copy-on-write overlay with userspace fallback.
func newOSOverlay(lowerDir, tempBase string) (CoWOverlay, error) {
	// APFS clonefile / copy-on-write helper with automatic fallback to userspace CoW overlay.
	return newUserspaceOverlay(lowerDir, tempBase)
}
