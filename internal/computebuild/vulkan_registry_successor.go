package computebuild

import (
	"fmt"
	"sort"
)

// Successor registries (V5 onward) share one receipt envelope: the registry
// identity, the native archive and the indexer-score structural contract. They
// differ only in schema, identity and module membership, so each is one row
// here and every receipt operation is a method on that row. Admitting a module
// is a new row whose added list names it; earlier rows keep their membership.
const (
	// VulkanShaderRegistryV5ID identifies the coupled indexer-score native archive
	// and exact 62-module bundle. V2/59, V3/60, and V4/61 remain historical identities.
	VulkanShaderRegistryV5ID = "fak.vulkan-shader-registry.v5.current62"
	// VulkanShaderRegistryV6ID identifies the exact 63-module bundle that adds the
	// V4.1 mHC stream module to V5/62.
	VulkanShaderRegistryV6ID = "fak.vulkan-shader-registry.v6.current63"
	// VulkanShaderRegistryV7ID identifies the exact 66-module bundle that adds the
	// V4.1 MoE router top-k, index-select and Q2_K embedding-gather modules to V6/63.
	VulkanShaderRegistryV7ID = "fak.vulkan-shader-registry.v7.current66"
)

type vulkanSuccessorRegistry struct {
	version string
	schema  string
	id      string
	count   int
	parent  *vulkanSuccessorRegistry
	added   []string
}

var (
	vulkanRegistryV5 = &vulkanSuccessorRegistry{version: "V5", schema: VulkanBuildReceiptSchemaV5, id: VulkanShaderRegistryV5ID, count: 62, added: []string{"v41_indexer_score"}}
	vulkanRegistryV6 = &vulkanSuccessorRegistry{version: "V6", schema: VulkanBuildReceiptSchemaV6, id: VulkanShaderRegistryV6ID, count: 63, parent: vulkanRegistryV5, added: []string{"v41_mhc"}}
	vulkanRegistryV7 = &vulkanSuccessorRegistry{version: "V7", schema: VulkanBuildReceiptSchemaV7, id: VulkanShaderRegistryV7ID, count: 66, parent: vulkanRegistryV6, added: []string{"v41_router_topk", "v41_index_select", "v41_q2k_embed_gather"}}

	// vulkanSuccessorRegistries is every successor schema a receipt may name.
	vulkanSuccessorRegistries = []*vulkanSuccessorRegistry{vulkanRegistryV5, vulkanRegistryV6, vulkanRegistryV7}
	// vulkanCurrentRegistry is the registry the builder compiles and stamps.
	vulkanCurrentRegistry = vulkanRegistryV7
)

func vulkanSuccessorForSchema(schema string) *vulkanSuccessorRegistry {
	for _, r := range vulkanSuccessorRegistries {
		if r.schema == schema {
			return r
		}
	}
	return nil
}

// stems returns an owned copy of the trusted membership in build order.
func (r *vulkanSuccessorRegistry) stems() []string {
	var base []string
	if r.parent == nil {
		base = CurrentVulkanShaderRegistryV4()
	} else {
		base = r.parent.stems()
	}
	return append(base, r.added...)
}

func (r *vulkanSuccessorRegistry) identity() (VulkanShaderRegistryIdentity, error) {
	stems := r.stems()
	if len(stems) != r.count {
		return VulkanShaderRegistryIdentity{}, fmt.Errorf("%s SPIR-V registry policy lists %d modules, want %d", r.version, len(stems), r.count)
	}
	sort.Strings(stems)
	digest, err := hashJSON(stems)
	if err != nil {
		return VulkanShaderRegistryIdentity{}, err
	}
	return VulkanShaderRegistryIdentity{ID: r.id, SHA256: digest, ModuleCount: r.count}, nil
}

// validateStems treats the supplied list only as a redundant assertion against
// policy. Neither receipt contents nor caller assertions select the census.
func (r *vulkanSuccessorRegistry) validateStems(stems []string) error {
	if len(stems) != r.count {
		return fmt.Errorf("%s SPIR-V registry must contain exactly %d modules", r.version, r.count)
	}
	expected := make(map[string]bool, r.count)
	for _, stem := range r.stems() {
		expected[stem] = true
	}
	seen := make(map[string]bool, r.count)
	for _, stem := range stems {
		if seen[stem] {
			return fmt.Errorf("duplicate %s SPIR-V registry module %q", r.version, stem)
		}
		if !expected[stem] {
			return fmt.Errorf("unknown %s SPIR-V registry module %q", r.version, stem)
		}
		seen[stem] = true
	}
	return nil
}

// CurrentVulkanShaderRegistryV5 returns an owned copy of the trusted V5 set.
func CurrentVulkanShaderRegistryV5() []string { return vulkanRegistryV5.stems() }

// CurrentVulkanShaderRegistryV6 returns an owned copy of the trusted V6 set.
func CurrentVulkanShaderRegistryV6() []string { return vulkanRegistryV6.stems() }

// CurrentVulkanShaderRegistryV7 returns an owned copy of the trusted V7 set.
func CurrentVulkanShaderRegistryV7() []string { return vulkanRegistryV7.stems() }

// CurrentVulkanShaderRegistryLatest returns an owned copy of the set the
// builder compiles and stamps today.
func CurrentVulkanShaderRegistryLatest() []string { return vulkanCurrentRegistry.stems() }
