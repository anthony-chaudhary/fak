//go:build !windows && !linux && !darwin

package compute

// diskInfo reports disk total/free bytes for path on unsupported platforms.
func diskInfo(path string) (total, free int64, known bool) {
	return 0, FreeUnknown, false
}