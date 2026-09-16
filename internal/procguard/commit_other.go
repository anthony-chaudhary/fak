//go:build !windows && !darwin && !linux

package procguard

// collectMemorySnapshot is the honest typed-unsupported path for GOOS values with
// no native collector (freebsd, openbsd, ...). supported=false with a non-empty
// detail is distinct from a live zero-byte tree: callers must SKIP the memory
// invariant here, never read the zero as a measurement. Linux is NOT this path;
// see commit_linux.go.
func collectMemorySnapshot(rootPID int) (MemorySnapshot, bool, string) {
	return MemorySnapshot{RootPID: rootPID}, false, "memory accounting unsupported on this platform"
}

func hostPhysicalMemoryBytes() (uint64, string) {
	return 0, "physical memory accounting unsupported on this platform"
}
