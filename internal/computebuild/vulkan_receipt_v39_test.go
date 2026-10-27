package computebuild

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms
// Source/receipt registration only; does not compile or qualify the candidate.
func TestVulkanReceiptV39QwenAttentionQG(t *testing.T) {
	const stem = "qwen35_attention_qg"
	want := append(vulkanRegistryV38.stems(), stem)
	if len(want) != 110 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V39 must add exactly the optional Qwen interleaved QG postprocess module to historical V38")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV39 || id.ID != VulkanShaderRegistryV39ID || id.ModuleCount != 110 {
		t.Fatalf("V39 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV38.validateStems(want) == nil || vulkanRegistryV39.validateStems(vulkanRegistryV38.stems()) == nil {
		t.Fatal("historical V38 and panel V39 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV39) != vulkanRegistryV39 ||
		!vulkanRegistryV39.usesQualifiedQ2TokenPair() || !vulkanRegistryV39.usesQualifiedIndexScale() {
		t.Fatal("V39 must preserve inherited qualification")
	}
	for _, file := range []string{filepath.Join("shaders", stem+".comp"), "vulkan_shim.cpp", "build_vulkan.ps1"} {
		data, err := os.ReadFile(filepath.Join("..", "compute", file))
		if err != nil {
			t.Fatal(err)
		}
		if file == "vulkan_shim.cpp" && !strings.Contains(string(data), `P("`+stem+`.spv")`) {
			t.Fatal("native optional loader must consume the canonical registered module")
		}
		if file == "build_vulkan.ps1" && !strings.Contains(string(data), `$shaders += "`+stem+`"`) {
			t.Fatal("legacy shader build must also include the panel module")
		}
	}
}
