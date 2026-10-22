package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV33SharedAttentionSplit(t *testing.T) {
	want := append(vulkanRegistryV32.stems(), "v41_shared_attention_split")
	if len(want) != 104 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V33 must add exactly the V4.1 split-k shared attention module to V32")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV33 || id.ID != VulkanShaderRegistryV33ID || id.ModuleCount != 104 {
		t.Fatalf("V33 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV32.validateStems(want) == nil || vulkanRegistryV33.validateStems(vulkanRegistryV32.stems()) == nil {
		t.Fatal("historical V32 and split-k V33 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV33) != vulkanRegistryV33 ||
		!vulkanRegistryV33.usesQualifiedQ2TokenPair() || !vulkanRegistryV33.usesQualifiedIndexScale() {
		t.Fatal("V33 must remain a registered successor with inherited qualification")
	}
}
