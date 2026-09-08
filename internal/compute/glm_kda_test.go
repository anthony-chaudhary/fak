package compute

import (
	"os"
	"strings"
	"testing"
)

func TestGLMKDAWave32OwnershipIsBijectiveAndBounded(t *testing.T) {
	const heads = 64
	stateOwners := make([]uint8, heads*GLMKDAHeadDim*GLMKDAHeadDim)
	outputOwners := make([]uint8, heads*GLMKDAHeadDim)
	for h := 0; h < heads; h++ {
		for j := 0; j < GLMKDAHeadDim; j++ {
			outputOwners[h*GLMKDAHeadDim+j]++
			for i := 0; i < GLMKDAHeadDim; i++ {
				idx := h*GLMKDAHeadDim*GLMKDAHeadDim + i*GLMKDAHeadDim + j
				if idx < 0 || idx >= len(stateOwners) {
					t.Fatalf("state index %d outside [0,%d)", idx, len(stateOwners))
				}
				stateOwners[idx]++
			}
		}
	}
	for idx, owners := range stateOwners {
		if owners != 1 {
			t.Fatalf("state[%d] owners=%d, want exactly 1", idx, owners)
		}
	}
	for idx, owners := range outputOwners {
		if owners != 1 {
			t.Fatalf("output[%d] owners=%d, want exactly 1", idx, owners)
		}
	}
}

func TestGLMKDAShadersPreserveMatchedOneLeverContract(t *testing.T) {
	read := func(name string) string {
		raw, err := os.ReadFile("shaders/" + name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return string(raw)
	}
	parent := read("glm_kda_recurrent_reread.comp")
	candidate := read("glm_kda_recurrent_wave32.comp")
	for name, src := range map[string]string{"parent": parent, "candidate": candidate} {
		for _, token := range []string{
			"layout(local_size_x = 128", "const int D = 128", "int h = int(gl_WorkGroupID.x)",
			"int j = int(gl_LocalInvocationID.x)", "stateBase + i * D + j",
			"outputData[vectorBase + j] = acc", "for (int i = 0; i < D; ++i)",
		} {
			if !strings.Contains(src, token) {
				t.Errorf("%s shader missing contract token %q", name, token)
			}
		}
		if strings.Contains(src, "barrier()") {
			t.Errorf("%s shader contains an unnecessary cross-invocation barrier", name)
		}
	}
	if !strings.Contains(parent, "state[si] = decayed") {
		t.Error("matched parent does not materialize the decayed state before reread")
	}
	if !strings.Contains(candidate, "float decayed[D]") || strings.Contains(candidate, "state[si] = decayed") {
		t.Error("candidate does not isolate private decayed-column retention")
	}
}

func TestGLMKDAPipelineRequiresWave32InsteadOfInferringIt(t *testing.T) {
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatalf("read vulkan_shim.cpp: %v", err)
	}
	src := string(raw)
	for _, token := range []string{
		"VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME",
		"requiredSubgroup.requiredSubgroupSize = subgroupSize",
		"P(\"glm_kda_recurrent_reread.spv\"), 7, sizeof(int), 32",
		"P(\"glm_kda_recurrent_wave32.spv\"), 7, sizeof(int), 32",
	} {
		if !strings.Contains(src, token) {
			t.Errorf("Vulkan shim missing explicit Wave32 contract token %q", token)
		}
	}
}

func TestGLMKDAVariantValuesMatchCABI(t *testing.T) {
	if VulkanGLMKDAReread != 0 || VulkanGLMKDAWave32Retain != 1 {
		t.Fatalf("variant values=(%d,%d), want C ABI values (0,1)", VulkanGLMKDAReread, VulkanGLMKDAWave32Retain)
	}
}
