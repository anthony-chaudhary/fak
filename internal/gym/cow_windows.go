//go:build windows

package gym

// newOSOverlay creates a Windows directory overlay helper with userspace fallback.
func newOSOverlay(lowerDir, tempBase string) (CoWOverlay, error) {
	// Windows hardlink/block-clone/directory overlay helper with userspace fallback.
	return newUserspaceOverlay(lowerDir, tempBase)
}
