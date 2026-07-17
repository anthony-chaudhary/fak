//go:build !linux

package compute

// onlineNUMANodes returns nil off Linux: there is no /sys NUMA topology to read, and the
// planner refuses (unsupported) before any node list would matter.
func onlineNUMANodes() []int { return nil }
