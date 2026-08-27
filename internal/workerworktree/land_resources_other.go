//go:build !windows && !linux && !darwin && !freebsd && !netbsd && !openbsd

package workerworktree

// currentLandResources deliberately reports unavailable on platforms where the
// standard library has no portable process CPU/peak-RSS probe.
func currentLandResources() landResourceSample {
	return landResourceSample{reason: "process CPU and peak RSS are unsupported on this operating system"}
}
