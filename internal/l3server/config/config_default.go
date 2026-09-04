//go:build !linux

package config

// nofileLimit returns 0 on non-Linux platforms.
func nofileLimit() int {
	return 0
}

// systemMemoryGB returns 0 on non-Linux platforms.
func systemMemoryGB() float64 {
	return 0
}
