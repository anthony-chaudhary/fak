package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV18TokenPanelBoundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV17(), "v41_q2k_token_panel")
	if len(want) != 82 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV18(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder must add exactly the Q2_K token-panel module to historical V17")
	}
	identity, err := vulkanRegistryV18.identity()
	if err != nil {
		t.Fatal(err)
	}
	if identity.ID != "fak.vulkan-shader-registry.v18.current82" || identity.ModuleCount != 82 || identity.SHA256 != "a6e2b8424c6d79db126e3707df6aa030e0ee5990de0fdaad70ded5f4d1bf7db9" {
		t.Fatalf("V18 exact82 identity differs from independent JSON vector: %+v", identity)
	}
	if vulkanRegistryV17.validateStems(want) == nil || vulkanRegistryV18.validateStems(CurrentVulkanShaderRegistryV17()) == nil {
		t.Fatal("historical V17 and token-panel V18 must not be interchangeable")
	}
	schema, current, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV18 || !reflect.DeepEqual(current, identity) || vulkanSuccessorForSchema(schema) != vulkanRegistryV18 {
		t.Fatalf("current receipt does not select V18: %q %+v %v", schema, current, err)
	}
}
