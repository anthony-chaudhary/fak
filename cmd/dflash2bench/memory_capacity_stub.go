//go:build !vulkan || (!windows && !linux) || !cgo

package main

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func vulkanMemoryCapacitySnapshot(compute.Backend) (any, error) {
	return nil, errors.New("Vulkan memory capacity unavailable in this build")
}
