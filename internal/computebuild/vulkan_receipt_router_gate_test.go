package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV31RouterGate(t *testing.T) {
	want := append(vulkanRegistryV30.stems(), "v41_router_gate_topk")
	if len(want) != 102 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V31 must add exactly the fused router gate module to V30")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV31 || id.ID != VulkanShaderRegistryV31ID || id.ModuleCount != 102 {
		t.Fatalf("V31 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV30.validateStems(want) == nil || vulkanRegistryV31.validateStems(vulkanRegistryV30.stems()) == nil {
		t.Fatal("historical V30 and router gate V31 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV31) != vulkanRegistryV31 ||
		!vulkanRegistryV31.usesQualifiedQ2TokenPair() || !vulkanRegistryV31.usesQualifiedIndexScale() {
		t.Fatal("V31 must remain a registered successor with inherited qualification")
	}
}
