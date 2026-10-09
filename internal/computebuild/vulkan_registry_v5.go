package computebuild

import (
	"fmt"
	"sort"
)

// VulkanShaderRegistryV5ID identifies the coupled indexer-score native archive
// and exact 62-module bundle. V2/59, V3/60, and V4/61 remain historical identities.
const VulkanShaderRegistryV5ID = "fak.vulkan-shader-registry.v5.current62"
const vulkanV5ModuleCount = 62

// CurrentVulkanShaderRegistryV5 returns an owned copy of the trusted V5 set.
// Earlier accessors deliberately retain their historical membership.
func CurrentVulkanShaderRegistryV5() []string {
	return append(CurrentVulkanShaderRegistryV4(), "v41_indexer_score")
}

func currentVulkanRegistryV5Identity() (VulkanShaderRegistryIdentity, error) {
	stems := CurrentVulkanShaderRegistryV5()
	sort.Strings(stems)
	digest, err := hashJSON(stems)
	if err != nil {
		return VulkanShaderRegistryIdentity{}, err
	}
	return VulkanShaderRegistryIdentity{ID: VulkanShaderRegistryV5ID, SHA256: digest, ModuleCount: vulkanV5ModuleCount}, nil
}

// The supplied list is only a redundant assertion against V5 policy. Neither
// receipt contents nor caller assertions select the filesystem census.
func validateCurrentVulkanRegistryV5(stems []string) error {
	if len(stems) != vulkanV5ModuleCount {
		return fmt.Errorf("V5 SPIR-V registry must contain exactly %d modules", vulkanV5ModuleCount)
	}
	expected := make(map[string]bool, vulkanV5ModuleCount)
	for _, stem := range CurrentVulkanShaderRegistryV5() {
		expected[stem] = true
	}
	seen := make(map[string]bool, vulkanV5ModuleCount)
	for _, stem := range stems {
		if seen[stem] {
			return fmt.Errorf("duplicate V5 SPIR-V registry module %q", stem)
		}
		if !expected[stem] {
			return fmt.Errorf("unknown V5 SPIR-V registry module %q", stem)
		}
		seen[stem] = true
	}
	return nil
}
