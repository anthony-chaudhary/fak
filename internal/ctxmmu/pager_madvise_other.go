//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd

package ctxmmu

// osMadviseRandom is a safe no-op stub for platforms without posix madvise (Windows, Plan9, etc.).
func osMadviseRandom(data []byte) bool {
	return false
}

// osMadviseWillneed is a safe no-op stub for platforms without posix madvise.
func osMadviseWillneed(data []byte, off, length int) bool {
	return false
}
