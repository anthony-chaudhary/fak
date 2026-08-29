package compute

type vulkanQ4KConfigurer interface {
	configureVulkanQ4K(profile, stage bool)
}

// ConfigureVulkanQ4K applies explicit diagnostic/staging settings to a selected Vulkan
// backend. It returns false for nil or non-Vulkan backends so operator-facing callers can
// fail loudly instead of claiming an inert configuration.
func ConfigureVulkanQ4K(backend Backend, profile, stage bool) bool {
	cfg, ok := backend.(vulkanQ4KConfigurer)
	if !ok {
		return false
	}
	cfg.configureVulkanQ4K(profile, stage)
	return true
}
