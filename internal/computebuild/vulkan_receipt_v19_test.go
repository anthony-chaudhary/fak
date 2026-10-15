package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV19TokenPanelBoundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV18(), "v41_q3k_token_panel")
	if len(want) != 83 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV19(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("historical V19 must add exactly the Q3_K token-panel module to historical V18")
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
}

// fak-test:runtime fast est=5ms
// Pins only packaging policy; no native archive or hardware qualification is implied.
func TestVulkanReceiptV20RoutedRowBoundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV19(), "v41_routed_swiglu_panel_row")
	if len(want) != 84 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV20(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("historical V20 must add exactly the optional routed-row module to historical V19")
	}
	identity, err := vulkanRegistryV20.identity()
	if err != nil {
		t.Fatal(err)
	}
	if identity.ID != "fak.vulkan-shader-registry.v20.current84" || identity.ModuleCount != 84 || identity.SHA256 != "833a9ccf384394d224ec76097423acd37026ddba9ee93aeec97a37d13e06c6a3" {
		t.Fatalf("V20 exact84 identity differs from independent JSON vector: %+v", identity)
	}
	if vulkanRegistryV19.validateStems(want) == nil || vulkanRegistryV20.validateStems(CurrentVulkanShaderRegistryV19()) == nil {
		t.Fatal("historical V19 and routed-row V20 must not be interchangeable")
	}
}

// fak-test:runtime fast est=5ms
// Pins only packaging policy; no native archive or hardware qualification is implied.
func TestVulkanReceiptV21DownRowBoundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV20(), "v41_routed_down_panel_row")
	if len(want) != 85 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV21(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder must add exactly the optional routed-row module to historical V20")
	}
	identity, err := vulkanRegistryV21.identity()
	if err != nil {
		t.Fatal(err)
	}
	if identity.ID != "fak.vulkan-shader-registry.v21.current85" || identity.ModuleCount != 85 || identity.SHA256 != "c42b852570f83fa81b0e744328cfe297b42493777f6ff8d9987f6fbf5976daa3" {
		t.Fatalf("V21 exact85 identity differs from independent JSON vector: %+v", identity)
	}
	if vulkanRegistryV20.validateStems(want) == nil || vulkanRegistryV21.validateStems(CurrentVulkanShaderRegistryV20()) == nil {
		t.Fatal("historical V20 and routed-row V21 must not be interchangeable")
	}
	schema, current, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV21 || !reflect.DeepEqual(current, identity) || vulkanSuccessorForSchema(schema) != vulkanRegistryV21 || vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV20) != vulkanRegistryV20 {
		t.Fatalf("current receipt does not select V21 or historical V20 is unavailable: %q %+v %v", schema, current, err)
	}
}
