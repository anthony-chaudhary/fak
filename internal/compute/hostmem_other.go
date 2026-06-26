//go:build !windows && !linux && !darwin

package compute

func hostSystemMemory() (total, free int64, known bool) {
	return 0, FreeUnknown, false
}
