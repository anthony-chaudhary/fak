//go:build !windows && !darwin

package procguard

func collectSystemMemorySnapshot() (MemorySnapshot, bool, string) {
	return MemorySnapshot{}, false, "system commit accounting unsupported on this platform"
}

func collectMemorySnapshot(rootPID int) (MemorySnapshot, bool, string) {
	return MemorySnapshot{RootPID: rootPID}, false, "memory accounting unsupported on this platform"
}

func hostPhysicalMemoryBytes() (uint64, string) {
	return 0, "physical memory accounting unsupported on this platform"
}
