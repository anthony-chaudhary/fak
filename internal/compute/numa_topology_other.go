//go:build !linux

package compute

func detectHostNUMATopologyPlatform() []NUMANodeTopology {
	return nil
}
