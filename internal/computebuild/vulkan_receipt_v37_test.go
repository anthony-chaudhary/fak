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
func TestVulkanReceiptV37QwenResidualNorm(t *testing.T) {
	const stem = "qwen35_residual_norm"
	want := append(vulkanRegistryV36.stems(), stem)
	if len(want) != 108 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V37 must add exactly the optional Qwen residual norm module to historical V36")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV37 || id.ID != VulkanShaderRegistryV37ID || id.ModuleCount != 108 {
		t.Fatalf("V37 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV36.validateStems(want) == nil || vulkanRegistryV37.validateStems(vulkanRegistryV36.stems()) == nil {
		t.Fatal("historical V36 and panel V37 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV37) != vulkanRegistryV37 ||
		!vulkanRegistryV37.usesQualifiedQ2TokenPair() || !vulkanRegistryV37.usesQualifiedIndexScale() {
		t.Fatal("V37 must preserve inherited qualification")
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
