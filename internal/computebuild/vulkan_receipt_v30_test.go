package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV30ValueHeadTile(t *testing.T) {
	want := append(vulkanRegistryV29.stems(), "qwen35_value_head_tile")
	if len(want) != 101 || !reflect.DeepEqual(want, vulkanRegistryV30.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V30 must add exactly the GDN value-head tiling module to V29")
	}
	id, err := vulkanRegistryV30.identity()
	schema := vulkanRegistryV30.schema
	if err != nil || schema != VulkanBuildReceiptSchemaV30 || id.ID != VulkanShaderRegistryV30ID || id.ModuleCount != 101 {
		t.Fatalf("V30 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV29.validateStems(want) == nil || vulkanRegistryV30.validateStems(vulkanRegistryV29.stems()) == nil {
		t.Fatal("historical V29 and value-head tiling V30 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV30) != vulkanRegistryV30 ||
		!vulkanRegistryV30.usesQualifiedQ2TokenPair() || !vulkanRegistryV30.usesQualifiedIndexScale() {
		t.Fatal("V30 must remain a registered successor with inherited qualification")
	}
}
