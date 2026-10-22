//go:build vulkan && (windows || linux) && cgo

package main

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func vulkanMemoryCapacitySnapshot(be compute.Backend) (any, error) {
	reporter, ok := be.(interface {
		VulkanMemoryCapacitySnapshot() (compute.VulkanMemoryCapacitySnapshot, error)
	})
	if !ok {
		return nil, errors.New("selected backend does not expose Vulkan memory capacity")
	}
	return reporter.VulkanMemoryCapacitySnapshot()
}
