//go:build !linux

package modelperfobs

import "fmt"

func DiscoverNUMATopology() (NUMATopology, error) {
	return NUMATopology{}, fmt.Errorf("NUMA topology discovery is supported only on Linux sysfs hosts")
}
