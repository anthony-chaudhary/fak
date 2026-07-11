//go:build !linux

package compute

// NUMA memory-policy confinement and cheap RSS introspection are Linux-only; on
// other platforms the loader's memory visibility degrades to nothing rather than
// guessing.
func readMemPolicy() memPolicy { return memPolicy{} }

func policyFreeBytes(nodes []int) (int64, bool) { return 0, false }

func processRSSBytes() int64 { return 0 }
