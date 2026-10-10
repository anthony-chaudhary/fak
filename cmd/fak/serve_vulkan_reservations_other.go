//go:build !vulkan || (!windows && !linux) || !cgo

package main

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func serveVulkanPostLoadReservations(*fakmodel.Model, compute.Backend, compute.MemoryPlan) (string, bool) {
	return "", false
}
