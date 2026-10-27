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
func TestVulkanReceiptV35QwenAttentionPostprocess(t *testing.T) {
	const stem = "qwen35_attention_postprocess"
	want := append(vulkanRegistryV34.stems(), stem)
	if len(want) != 106 || !reflect.DeepEqual(want, vulkanRegistryV35.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V35 must add exactly the optional Qwen attention postprocess module to historical V34")
	}
	schema := vulkanRegistryV35.schema
	id, err := vulkanRegistryV35.identity()
	if err != nil || schema != VulkanBuildReceiptSchemaV35 || id.ID != VulkanShaderRegistryV35ID || id.ModuleCount != 106 {
		t.Fatalf("V35 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV34.validateStems(want) == nil || vulkanRegistryV35.validateStems(vulkanRegistryV34.stems()) == nil {
		t.Fatal("historical V34 and panel V35 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV35) != vulkanRegistryV35 ||
		!vulkanRegistryV35.usesQualifiedQ2TokenPair() || !vulkanRegistryV35.usesQualifiedIndexScale() {
		t.Fatal("V35 must preserve inherited qualification")
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
