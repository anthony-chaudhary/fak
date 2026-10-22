package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV32CompressorRing(t *testing.T) {
	want := append(vulkanRegistryV31.stems(), "v41_compressor_ring")
	if len(want) != 103 || !reflect.DeepEqual(want, vulkanRegistryV32.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V32 must add exactly the V4.1 compressor ring module to V31")
	}
	id, err := vulkanRegistryV32.identity()
	schema := vulkanRegistryV32.schema
	if err != nil || schema != VulkanBuildReceiptSchemaV32 || id.ID != VulkanShaderRegistryV32ID || id.ModuleCount != 103 {
		t.Fatalf("V32 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV31.validateStems(want) == nil || vulkanRegistryV32.validateStems(vulkanRegistryV31.stems()) == nil {
		t.Fatal("historical V31 and compressor ring V32 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV32) != vulkanRegistryV32 ||
		!vulkanRegistryV32.usesQualifiedQ2TokenPair() || !vulkanRegistryV32.usesQualifiedIndexScale() {
		t.Fatal("V32 must remain a registered successor with inherited qualification")
	}
}
