//go:build linux

package compute

import "os"

// onlineNUMANodes reads the kernel's online NUMA node ids from
// /sys/devices/system/node/online (e.g. "0-7" ⇒ [0..7]). Returns nil when the file is
// absent or unparseable — the planner then reports single_node and declines to interleave.
func onlineNUMANodes() []int {
	b, err := os.ReadFile("/sys/devices/system/node/online")
	if err != nil {
		return nil
	}
	return parseNodeList(string(b))
}
