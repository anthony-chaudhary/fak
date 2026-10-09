package computebuild

import (
	"fmt"
	"sort"
)

// VulkanShaderRegistryV4ID identifies the coupled shared-attention native archive
// and exact 61-module bundle. V2/59 and V3/60 remain historical identities.
const VulkanShaderRegistryV4ID = "fak.vulkan-shader-registry.v4.current61"
const vulkanV4ModuleCount = 61

// CurrentVulkanShaderRegistryV4 returns an owned copy of the trusted V4 set.
// CurrentVulkanShaderRegistry deliberately remains frozen at V3/60.
func CurrentVulkanShaderRegistryV4() []string {
	return append(CurrentVulkanShaderRegistry(), "v41_shared_attention")
}

func currentVulkanRegistryV4Identity() (VulkanShaderRegistryIdentity, error) {
	stems := CurrentVulkanShaderRegistryV4()
	sort.Strings(stems)
	digest, err := hashJSON(stems)
	if err != nil {
		return VulkanShaderRegistryIdentity{}, err
	}
	return VulkanShaderRegistryIdentity{ID: VulkanShaderRegistryV4ID, SHA256: digest, ModuleCount: vulkanV4ModuleCount}, nil
}

// The supplied list is only a redundant assertion against V4 policy. Neither
// receipt contents nor caller assertions select the filesystem census.
func validateCurrentVulkanRegistryV4(stems []string) error {
	if len(stems) != vulkanV4ModuleCount {
		return fmt.Errorf("V4 SPIR-V registry must contain exactly %d modules", vulkanV4ModuleCount)
	}
	expected := make(map[string]bool, vulkanV4ModuleCount)
	for _, stem := range CurrentVulkanShaderRegistryV4() {
		expected[stem] = true
	}
	seen := make(map[string]bool, vulkanV4ModuleCount)
	for _, stem := range stems {
		if seen[stem] {
			return fmt.Errorf("duplicate V4 SPIR-V registry module %q", stem)
		}
		if !expected[stem] {
			return fmt.Errorf("unknown V4 SPIR-V registry module %q", stem)
		}
		seen[stem] = true
	}
	return nil
}
