//go:build windows

package main

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// probeVulkanLoaderFacts runs the cgo-free Windows Vulkan loader probe and maps
// its receipt or typed error into the pure readiness fact. This is the only live
// I/O seam: it lives here (platform-split) so buildServeReadiness stays a pure
// fold and tests never need a Vulkan runtime. A missing or broken loader is
// reported as an Applicable-but-unavailable fact, not an error, so the doctor can
// render a typed, actionable warn instead of failing the serve-readiness check.
func probeVulkanLoaderFacts() *vulkanLoaderFacts {
	r, err := compute.ProbeVulkanPureGoLoader()
	if err != nil {
		f := &vulkanLoaderFacts{Applicable: true, Available: false, Detail: err.Error()}
		var typed *compute.VulkanPureGoLoaderError
		if errors.As(err, &typed) {
			f.Kind = string(typed.Kind)
			if typed.Loader != "" {
				f.Detail = fmt.Sprintf("loader=%s symbol=%s: %v", typed.Loader, typed.Symbol, err)
			}
		}
		return f
	}
	return &vulkanLoaderFacts{
		Applicable: true,
		Available:  true,
		Version:    fmt.Sprintf("%d.%d.%d", r.Major, r.Minor, r.Patch),
		Detail:     r.Loader,
	}
}
