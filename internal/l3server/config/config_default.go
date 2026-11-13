//go:build !linux

package config

// nofileLimit returns 0 on non-Linux platforms.
func nofileLimit() int {
	return 0
}

// systemMemoryGB returns 0 on non-Linux platforms. It is a function variable
// (matching the Linux definition) so tests can substitute a fixed value.
var systemMemoryGB = func() float64 {
	return 0
}
