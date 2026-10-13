package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV19TokenPanelBoundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV18(), "v41_q3k_token_panel")
	if len(want) != 83 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV19(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder must add exactly the Q3_K token-panel module to historical V18")
	}
	identity, err := vulkanRegistryV19.identity()
	if err != nil {
		t.Fatal(err)
	}
	if identity.ID != "fak.vulkan-shader-registry.v19.current83" || identity.ModuleCount != 83 || identity.SHA256 != "cc9d6705bb8bb5625448419075b70167584ba18c2b7c513c62ff343c4e66f3e6" {
		t.Fatalf("V19 exact83 identity differs from independent JSON vector: %+v", identity)
	}
	if vulkanRegistryV18.validateStems(want) == nil || vulkanRegistryV19.validateStems(CurrentVulkanShaderRegistryV18()) == nil {
		t.Fatal("historical V18 and token-panel V19 must not be interchangeable")
	}
	schema, current, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV19 || !reflect.DeepEqual(current, identity) || vulkanSuccessorForSchema(schema) != vulkanRegistryV19 {
		t.Fatalf("current receipt does not select V19: %q %+v %v", schema, current, err)
	}
}
