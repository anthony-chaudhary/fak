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
	// VulkanShaderRegistryV8ID identifies the exact 67-module bundle that adds the
	// V4.1 bit-exact RMSNorm module (compressor and index-key norms) to V7/66.
	VulkanShaderRegistryV8ID = "fak.vulkan-shader-registry.v8.current67"
	// VulkanShaderRegistryV9ID identifies the exact 68-module ordinary bundle
	// adding checked attention BF16 publication to the historical V8/67 set.
	VulkanShaderRegistryV9ID = "fak.vulkan-shader-registry.v9.current68"
	// VulkanShaderRegistryV10ID identifies the exact 69-module bundle that adds the
	// V4.1 routed-expert activation module (BF16 clamp SwiGLU route weight) to V9/68.
	VulkanShaderRegistryV10ID = "fak.vulkan-shader-registry.v10.current69"
	// VulkanShaderRegistryV11ID identifies the exact 71-module bundle that adds the
	// V4.1 Engram gate/add and compressor gated-pool modules to V10/69.
	VulkanShaderRegistryV11ID = "fak.vulkan-shader-registry.v11.current71"
	// VulkanShaderRegistryV12ID identifies the exact 72-module bundle that adds the
	// V4.1 strict binary32 tail-RoPE module (compressed publication RoPE) to V11/71.
	VulkanShaderRegistryV12ID = "fak.vulkan-shader-registry.v12.current72"
	// VulkanShaderRegistryV13ID identifies the exact 73-module bundle that adds the
	// V4.1 strict binary32 scalar-multiply module (indexer weight scale) to V12/72.
	VulkanShaderRegistryV13ID = "fak.vulkan-shader-registry.v13.current73"
	// VulkanShaderRegistryV14ID identifies the exact 74-module bundle that adds the
	// V4.1 Engram table-row decode module to V13/73.
	VulkanShaderRegistryV14ID = "fak.vulkan-shader-registry.v14.current74"
	// VulkanShaderRegistryV15ID identifies the exact 78-module bundle that adds the
	// four V4.1 routed-expert id modules (Q2_K/Q3_K id matvec, id SwiGLU, publish sum) to V14/74.
	VulkanShaderRegistryV15ID = "fak.vulkan-shader-registry.v15.current78"
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
	vulkanRegistryV8 = &vulkanSuccessorRegistry{version: "V8", schema: VulkanBuildReceiptSchemaV8, id: VulkanShaderRegistryV8ID, count: 67, parent: vulkanRegistryV7, added: []string{"v41_exact_rmsnorm"}}

	vulkanRegistryV9  = &vulkanSuccessorRegistry{version: "V9", schema: VulkanBuildReceiptSchemaV9, id: VulkanShaderRegistryV9ID, count: 68, parent: vulkanRegistryV8, added: []string{"v41_bf16_checked"}}
	vulkanRegistryV10 = &vulkanSuccessorRegistry{version: "V10", schema: VulkanBuildReceiptSchemaV10, id: VulkanShaderRegistryV10ID, count: 69, parent: vulkanRegistryV9, added: []string{"v41_routed_swiglu"}}

	vulkanRegistryV11 = &vulkanSuccessorRegistry{version: "V11", schema: VulkanBuildReceiptSchemaV11, id: VulkanShaderRegistryV11ID, count: 71, parent: vulkanRegistryV10, added: []string{"v41_engram_gate", "v41_compressor_pool"}}

	vulkanRegistryV12 = &vulkanSuccessorRegistry{version: "V12", schema: VulkanBuildReceiptSchemaV12, id: VulkanShaderRegistryV12ID, count: 72, parent: vulkanRegistryV11, added: []string{"v41_tail_rope_exact"}}

	vulkanRegistryV13 = &vulkanSuccessorRegistry{version: "V13", schema: VulkanBuildReceiptSchemaV13, id: VulkanShaderRegistryV13ID, count: 73, parent: vulkanRegistryV12, added: []string{"v41_scale_exact"}}

	vulkanRegistryV14 = &vulkanSuccessorRegistry{version: "V14", schema: VulkanBuildReceiptSchemaV14, id: VulkanShaderRegistryV14ID, count: 74, parent: vulkanRegistryV13, added: []string{"v41_engram_row_decode"}}

	vulkanRegistryV15 = &vulkanSuccessorRegistry{version: "V15", schema: VulkanBuildReceiptSchemaV15, id: VulkanShaderRegistryV15ID, count: 78, parent: vulkanRegistryV14, added: []string{"v41_q2k_matvec_id", "v41_q3k_matvec_id", "v41_routed_swiglu_id", "v41_routed_publish_sum"}}

	// vulkanSuccessorRegistries is every successor schema a receipt may name.
	vulkanSuccessorRegistries = []*vulkanSuccessorRegistry{vulkanRegistryV5, vulkanRegistryV6, vulkanRegistryV7, vulkanRegistryV8, vulkanRegistryV9, vulkanRegistryV10, vulkanRegistryV11, vulkanRegistryV12, vulkanRegistryV13, vulkanRegistryV14, vulkanRegistryV15}
	// vulkanCurrentRegistry is the registry the builder compiles and stamps.
	vulkanCurrentRegistry = vulkanRegistryV15
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

// CurrentVulkanShaderRegistryV8 returns an owned copy of the trusted V8 set.
func CurrentVulkanShaderRegistryV8() []string { return vulkanRegistryV8.stems() }

// CurrentVulkanShaderRegistryV9 returns an owned copy of the trusted V9 set.
func CurrentVulkanShaderRegistryV9() []string { return vulkanRegistryV9.stems() }

// CurrentVulkanShaderRegistryV10 returns an owned copy of the trusted V10 set.
func CurrentVulkanShaderRegistryV10() []string { return vulkanRegistryV10.stems() }

// CurrentVulkanShaderRegistryV11 returns an owned copy of the trusted V11 set.
func CurrentVulkanShaderRegistryV11() []string { return vulkanRegistryV11.stems() }

// CurrentVulkanShaderRegistryV12 returns an owned copy of the trusted V12 set.
func CurrentVulkanShaderRegistryV12() []string { return vulkanRegistryV12.stems() }

// CurrentVulkanShaderRegistryV13 returns an owned copy of the trusted V13 set.
func CurrentVulkanShaderRegistryV13() []string { return vulkanRegistryV13.stems() }

// CurrentVulkanShaderRegistryV14 returns an owned copy of the trusted V14 set.
func CurrentVulkanShaderRegistryV14() []string { return vulkanRegistryV14.stems() }

// CurrentVulkanShaderRegistryV15 returns an owned copy of the trusted V15 set.
func CurrentVulkanShaderRegistryV15() []string { return vulkanRegistryV15.stems() }

// CurrentVulkanShaderRegistryLatest returns an owned copy of the set the
// builder compiles and stamps today.
func CurrentVulkanShaderRegistryLatest() []string { return vulkanCurrentRegistry.stems() }

// CurrentVulkanBuildIdentity returns the receipt schema and registry identity
// the builder stamps today.
func CurrentVulkanBuildIdentity() (string, VulkanShaderRegistryIdentity, error) {
	identity, err := vulkanCurrentRegistry.identity()
	if err != nil {
		return "", VulkanShaderRegistryIdentity{}, err
	}
	return vulkanCurrentRegistry.schema, identity, nil
}
