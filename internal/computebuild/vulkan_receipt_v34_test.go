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
func TestVulkanReceiptV34QwenGDNQ8Panel(t *testing.T) {
	const stem = "qwen35_gdn_q8_in_proj_panel"
	want := append(vulkanRegistryV33.stems(), stem)
	if len(want) != 105 || !reflect.DeepEqual(want, vulkanRegistryV34.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V34 must add exactly the optional Qwen GDN Q8 panel to historical V33")
	}
	schema := vulkanRegistryV34.schema
	id, err := vulkanRegistryV34.identity()
	if err != nil || schema != VulkanBuildReceiptSchemaV34 || id.ID != VulkanShaderRegistryV34ID || id.ModuleCount != 105 {
		t.Fatalf("V34 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV33.validateStems(want) == nil || vulkanRegistryV34.validateStems(vulkanRegistryV33.stems()) == nil {
		t.Fatal("historical V33 and panel V34 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV34) != vulkanRegistryV34 ||
		!vulkanRegistryV34.usesQualifiedQ2TokenPair() || !vulkanRegistryV34.usesQualifiedIndexScale() {
		t.Fatal("V34 must preserve inherited qualification")
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
