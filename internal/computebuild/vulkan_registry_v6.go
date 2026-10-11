package computebuild

import (
	"fmt"
	"sort"
)

// VulkanShaderRegistryV6ID identifies the exact 63-module bundle that adds the
// V4.1 mHC stream module to V5/62. V2/59, V3/60, V4/61 and V5/62 remain
// historical identities.
const VulkanShaderRegistryV6ID = "fak.vulkan-shader-registry.v6.current63"
const vulkanV6ModuleCount = 63

// CurrentVulkanShaderRegistryV6 returns an owned copy of the trusted V6 set.
// Earlier accessors deliberately retain their historical membership.
func CurrentVulkanShaderRegistryV6() []string {
	return append(CurrentVulkanShaderRegistryV5(), "v41_mhc")
}

func currentVulkanRegistryV6Identity() (VulkanShaderRegistryIdentity, error) {
	stems := CurrentVulkanShaderRegistryV6()
	sort.Strings(stems)
	digest, err := hashJSON(stems)
	if err != nil {
		return VulkanShaderRegistryIdentity{}, err
	}
	return VulkanShaderRegistryIdentity{ID: VulkanShaderRegistryV6ID, SHA256: digest, ModuleCount: vulkanV6ModuleCount}, nil
}

// The supplied list is only a redundant assertion against V6 policy. Neither
// receipt contents nor caller assertions select the filesystem census.
func validateCurrentVulkanRegistryV6(stems []string) error {
	if len(stems) != vulkanV6ModuleCount {
		return fmt.Errorf("V6 SPIR-V registry must contain exactly %d modules", vulkanV6ModuleCount)
	}
	expected := make(map[string]bool, vulkanV6ModuleCount)
	for _, stem := range CurrentVulkanShaderRegistryV6() {
		expected[stem] = true
	}
	seen := make(map[string]bool, vulkanV6ModuleCount)
	for _, stem := range stems {
		if seen[stem] {
			return fmt.Errorf("duplicate V6 SPIR-V registry module %q", stem)
		}
		if !expected[stem] {
			return fmt.Errorf("unknown V6 SPIR-V registry module %q", stem)
		}
		seen[stem] = true
	}
	return nil
}
