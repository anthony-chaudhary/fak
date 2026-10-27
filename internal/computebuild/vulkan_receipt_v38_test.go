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
func TestVulkanReceiptV38Q5KPanel(t *testing.T) {
	const stem = "q5k_matmul_panel"
	want := append(vulkanRegistryV37.stems(), stem)
	if len(want) != 109 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V38 must add exactly the optional Q5_K companion panel tile to historical V37")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV38 || id.ID != VulkanShaderRegistryV38ID || id.ModuleCount != 109 {
		t.Fatalf("V38 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV37.validateStems(want) == nil || vulkanRegistryV38.validateStems(vulkanRegistryV37.stems()) == nil {
		t.Fatal("historical V37 and panel V38 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV38) != vulkanRegistryV38 ||
		!vulkanRegistryV38.usesQualifiedQ2TokenPair() || !vulkanRegistryV38.usesQualifiedIndexScale() {
		t.Fatal("V38 must preserve inherited qualification")
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
