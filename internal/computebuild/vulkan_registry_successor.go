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
	// VulkanShaderRegistryV16ID identifies the exact 79-module bundle that adds the
	// run-to-run V4.1 shared attention module to V15/78.
	VulkanShaderRegistryV16ID = "fak.vulkan-shader-registry.v16.current79"
	// VulkanShaderRegistryV17ID identifies the exact 81-module bundle that adds the
	// two V4.1 fused routed-FFN modules (gate+up+SwiGLU+mid, down+sum) to V16/79.
	VulkanShaderRegistryV17ID = "fak.vulkan-shader-registry.v17.current81"
	// VulkanShaderRegistryV18ID adds the opt-in same-weight Q2_K token tile.
	VulkanShaderRegistryV18ID = "fak.vulkan-shader-registry.v18.current82"
	// VulkanShaderRegistryV19ID adds same-weight Q3_K token sharing.
	VulkanShaderRegistryV19ID = "fak.vulkan-shader-registry.v19.current83"
	// VulkanShaderRegistryV20ID adds the optional row-addressed routed SwiGLU module.
	VulkanShaderRegistryV20ID = "fak.vulkan-shader-registry.v20.current84"
	// VulkanShaderRegistryV21ID adds optional row-addressed routed down publication.
	VulkanShaderRegistryV21ID = "fak.vulkan-shader-registry.v21.current85"
	// VulkanShaderRegistryV22ID adds optional fused index-scale BF16 publication.
	VulkanShaderRegistryV22ID = "fak.vulkan-shader-registry.v22.current86"
	// VulkanShaderRegistryV23ID preserves V22 membership and binds canonical materialization.
	VulkanShaderRegistryV23ID = "fak.vulkan-shader-registry.v23.current86"
	// VulkanShaderRegistryV24ID adds the canonical routed activation chunk.
	VulkanShaderRegistryV24ID = "fak.vulkan-shader-registry.v24.current87"
	// VulkanShaderRegistryV25ID adds canonical row-local down publication chunks.
	VulkanShaderRegistryV25ID = "fak.vulkan-shader-registry.v25.current88"
	VulkanShaderRegistryV26ID = "fak.vulkan-shader-registry.v26.current89"
	// VulkanShaderRegistryV27ID adds the nine K-quant/i-quant prefill coopmat GEMM modules.
	VulkanShaderRegistryV27ID = "fak.vulkan-shader-registry.v27.current98"
	// VulkanShaderRegistryV28ID adds the grouped-query split-KV decode attention module.
	VulkanShaderRegistryV28ID = "fak.vulkan-shader-registry.v28.current99"
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

	vulkanRegistryV16 = &vulkanSuccessorRegistry{version: "V16", schema: VulkanBuildReceiptSchemaV16, id: VulkanShaderRegistryV16ID, count: 79, parent: vulkanRegistryV15, added: []string{"v41_shared_attention_fast"}}

	vulkanRegistryV17 = &vulkanSuccessorRegistry{version: "V17", schema: VulkanBuildReceiptSchemaV17, id: VulkanShaderRegistryV17ID, count: 81, parent: vulkanRegistryV16, added: []string{"v41_routed_gate_up_swiglu_id", "v41_routed_down_sum"}}

	vulkanRegistryV18 = &vulkanSuccessorRegistry{version: "V18", schema: VulkanBuildReceiptSchemaV18, id: VulkanShaderRegistryV18ID, count: 82, parent: vulkanRegistryV17, added: []string{"v41_q2k_token_panel"}}

	vulkanRegistryV19 = &vulkanSuccessorRegistry{version: "V19", schema: VulkanBuildReceiptSchemaV19, id: VulkanShaderRegistryV19ID, count: 83, parent: vulkanRegistryV18, added: []string{"v41_q3k_token_panel"}}

	vulkanRegistryV20 = &vulkanSuccessorRegistry{version: "V20", schema: VulkanBuildReceiptSchemaV20, id: VulkanShaderRegistryV20ID, count: 84, parent: vulkanRegistryV19, added: []string{"v41_routed_swiglu_panel_row"}}

	vulkanRegistryV21 = &vulkanSuccessorRegistry{version: "V21", schema: VulkanBuildReceiptSchemaV21, id: VulkanShaderRegistryV21ID, count: 85, parent: vulkanRegistryV20, added: []string{"v41_routed_down_panel_row"}}

	vulkanRegistryV22 = &vulkanSuccessorRegistry{version: "V22", schema: VulkanBuildReceiptSchemaV22, id: VulkanShaderRegistryV22ID, count: 86, parent: vulkanRegistryV21, added: []string{"v41_index_scale_bf16"}}

	vulkanRegistryV23 = &vulkanSuccessorRegistry{version: "V23", schema: VulkanBuildReceiptSchemaV23, id: VulkanShaderRegistryV23ID, count: 86, parent: vulkanRegistryV22}

	// vulkanSuccessorRegistries is every successor schema a receipt may name.
	vulkanRegistryV24 = &vulkanSuccessorRegistry{version: "V24", schema: VulkanBuildReceiptSchemaV24, id: VulkanShaderRegistryV24ID, count: 87, parent: vulkanRegistryV23, added: []string{"v41_routed_swiglu_panel_chunk"}}

	vulkanRegistryV25 = &vulkanSuccessorRegistry{version: "V25", schema: VulkanBuildReceiptSchemaV25, id: VulkanShaderRegistryV25ID, count: 88, parent: vulkanRegistryV24, added: []string{"v41_routed_down_panel_chunk"}}

	vulkanRegistryV26 = &vulkanSuccessorRegistry{version: "V26", schema: VulkanBuildReceiptSchemaV26, id: VulkanShaderRegistryV26ID, count: 89, parent: vulkanRegistryV25, added: []string{"v41_q2k_token_pair"}}

	vulkanRegistryV27 = &vulkanSuccessorRegistry{version: "V27", schema: VulkanBuildReceiptSchemaV27, id: VulkanShaderRegistryV27ID, count: 98, parent: vulkanRegistryV26, added: []string{"mm_cm1_q2k", "mm_cm1_iq2xxs", "mm_cm1_iq2xs", "mm_cm1_iq2s", "mm_cm1_iq3xxs", "mm_cm1_iq3s", "mm_cm1_iq1s", "mm_cm1_iq4xs", "mm_cm1_q4k"}}

	vulkanRegistryV28 = &vulkanSuccessorRegistry{version: "V28", schema: VulkanBuildReceiptSchemaV28, id: VulkanShaderRegistryV28ID, count: 99, parent: vulkanRegistryV27, added: []string{"attention_split_kv"}}

	vulkanSuccessorRegistries = []*vulkanSuccessorRegistry{vulkanRegistryV5, vulkanRegistryV6, vulkanRegistryV7, vulkanRegistryV8, vulkanRegistryV9, vulkanRegistryV10, vulkanRegistryV11, vulkanRegistryV12, vulkanRegistryV13, vulkanRegistryV14, vulkanRegistryV15, vulkanRegistryV16, vulkanRegistryV17, vulkanRegistryV18, vulkanRegistryV19, vulkanRegistryV20, vulkanRegistryV21, vulkanRegistryV22, vulkanRegistryV23, vulkanRegistryV24, vulkanRegistryV25, vulkanRegistryV26, vulkanRegistryV27, vulkanRegistryV28}
	// vulkanCurrentRegistry is the registry the builder compiles and stamps.
	vulkanCurrentRegistry = vulkanRegistryV28
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

// CurrentVulkanShaderRegistryV16 returns an owned copy of the trusted V16 set.
func CurrentVulkanShaderRegistryV16() []string { return vulkanRegistryV16.stems() }

// CurrentVulkanShaderRegistryV17 returns an owned copy of the trusted V17 set.
func CurrentVulkanShaderRegistryV17() []string { return vulkanRegistryV17.stems() }

// CurrentVulkanShaderRegistryV18 returns the exact experimental tile bundle.
func CurrentVulkanShaderRegistryV18() []string { return vulkanRegistryV18.stems() }

// CurrentVulkanShaderRegistryV19 returns the exact Q3_K token-tile bundle.
func CurrentVulkanShaderRegistryV19() []string { return vulkanRegistryV19.stems() }

// CurrentVulkanShaderRegistryV20 returns the exact optional routed-row bundle.
func CurrentVulkanShaderRegistryV20() []string { return vulkanRegistryV20.stems() }

// CurrentVulkanShaderRegistryV21 returns the exact optional down-row bundle.
func CurrentVulkanShaderRegistryV21() []string { return vulkanRegistryV21.stems() }

// CurrentVulkanShaderRegistryV22 returns the exact optional index-scale bundle.
func CurrentVulkanShaderRegistryV22() []string { return vulkanRegistryV22.stems() }

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

// CurrentVulkanBuildSchema returns the receipt schema the builder stamps today.
func CurrentVulkanBuildSchema() string { return vulkanCurrentRegistry.schema }
