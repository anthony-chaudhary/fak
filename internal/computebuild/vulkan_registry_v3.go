package computebuild

import (
	"fmt"
	"sort"
	"strings"
)

// VulkanShaderRegistryV3ID identifies the coupled current native archive and bundle.
// Historical V2 evidence does not authorize running legacy59 with the current backend.
const VulkanShaderRegistryV3ID = "fak.vulkan-shader-registry.v3.current60"
const vulkanV3ModuleCount = 60

// This literal freezes the historical order and membership independently of the
// exported compatibility slice. Every accessor returns owned storage.
const historicalVulkanV2Registry = `matmul
matmul_add
matmul_argmax
matmul_argmax_blocks
matmul2
matmul3
rmsnorm
rmsnorm_matmul
rmsnorm_matmul2
rmsnorm_matmul3
rmsnorm_matmul_argmax_blocks
rope
swiglu
swiglu_matmul_add
add
add_bias
attention
argmax
argmax_pairs
q8_matmul
q8_matmul2
q8_matmul3
rmsnorm_q8_matmul2
rmsnorm_q8_matmul3
swiglu_q8_matmul_add
qwen35_gdn_q8_in_proj
qwen35_gdn_conv
qwen35_gdn_recurrent
q4k_matmul
q4k_matmul_wave32
q4k_matmul_coopmat
q6k_matmul
q5k_matmul
q3k_matmul
q2k_matmul
qwen35_split_qg_panel
qwen35_partial_rope_panel
qwen35_causal_attention_panel
sigmoid_mul
q8_matmul_decode
glm_kda_recurrent_reread
glm_kda_recurrent_wave32
flash_attn_dequant
qwen35_gdn_tiled_transpose
coopmat_wave32_wmma
rmsnorm_q4k_matmul2
swiglu_q4k_matmul_add
qwen35_gdn_prefill_tiled
qwen35_gdn_prefill_norm
qwen35_gdn_verify_tiled
q2k_matvec
rmsnorm_q8_matmul2_coop
iq4xs_matvec
iq3xxs_matvec
iq2s_matvec
iq3s_matvec
iq2xxs_matvec
iq2xs_matvec
iq1s_matvec`

func historicalVulkanV2Shaders() []string {
	return strings.Fields(historicalVulkanV2Registry)
}

// CurrentVulkanShaderRegistry returns a fresh copy of the trusted current set.
// Callers cannot change build or verification policy by mutating this slice.
func CurrentVulkanShaderRegistry() []string {
	return append(historicalVulkanV2Shaders(), "v41_tail_rope_qk")
}

// VulkanShaderRegistryIdentity declares the trusted set, not a caller-provided list.
// SHA256 hashes the sorted canonical stems as a JSON string array.
type VulkanShaderRegistryIdentity struct {
	ID          string `json:"id"`
	SHA256      string `json:"sha256"`
	ModuleCount int    `json:"module_count"`
}

func currentVulkanRegistryIdentity() (VulkanShaderRegistryIdentity, error) {
	stems := CurrentVulkanShaderRegistry()
	sort.Strings(stems)
	digest, err := hashJSON(stems)
	if err != nil {
		return VulkanShaderRegistryIdentity{}, err
	}
	return VulkanShaderRegistryIdentity{ID: VulkanShaderRegistryV3ID, SHA256: digest, ModuleCount: vulkanV3ModuleCount}, nil
}

// A supplied evidence list is only a redundant assertion against trusted policy.
// Neither receipt data nor the assertion chooses the expected filesystem census.
func validateCurrentVulkanRegistry(stems []string) error {
	if len(stems) != vulkanV3ModuleCount {
		return fmt.Errorf("current SPIR-V registry must contain exactly %d modules", vulkanV3ModuleCount)
	}
	expected := make(map[string]bool, vulkanV3ModuleCount)
	for _, stem := range CurrentVulkanShaderRegistry() {
		expected[stem] = true
	}
	seen := make(map[string]bool, vulkanV3ModuleCount)
	for _, stem := range stems {
		if seen[stem] {
			return fmt.Errorf("duplicate current SPIR-V registry module %q", stem)
		}
		if !expected[stem] {
			return fmt.Errorf("unknown current SPIR-V registry module %q", stem)
		}
		seen[stem] = true
	}
	return nil
}
