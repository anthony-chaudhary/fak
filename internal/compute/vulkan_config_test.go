package compute

import "testing"

type recordingVulkanQ4KBackend struct {
	Backend
	profile bool
	stage   bool
}

func (b *recordingVulkanQ4KBackend) configureVulkanQ4K(profile, stage bool) {
	b.profile, b.stage = profile, stage
}

func TestConfigureVulkanQ4KReachesSelectedBackend(t *testing.T) {
	b := &recordingVulkanQ4KBackend{}
	if !ConfigureVulkanQ4K(b, true, true) || !b.profile || !b.stage {
		t.Fatalf("Vulkan Q4_K config did not reach backend: %+v", b)
	}
	if ConfigureVulkanQ4K(nil, true, true) {
		t.Fatal("nil backend accepted Vulkan Q4_K config")
	}
}
