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
func TestVulkanReceiptV36Q6KPanel(t *testing.T) {
	const stem = "q6k_matmul_panel"
	want := append(vulkanRegistryV35.stems(), stem)
	if len(want) != 107 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V36 must add exactly the optional Q6_K panel tile to historical V35")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV36 || id.ID != VulkanShaderRegistryV36ID || id.ModuleCount != 107 {
		t.Fatalf("V36 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV35.validateStems(want) == nil || vulkanRegistryV36.validateStems(vulkanRegistryV35.stems()) == nil {
		t.Fatal("historical V35 and panel V36 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV36) != vulkanRegistryV36 ||
		!vulkanRegistryV36.usesQualifiedQ2TokenPair() || !vulkanRegistryV36.usesQualifiedIndexScale() {
		t.Fatal("V36 must preserve inherited qualification")
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
