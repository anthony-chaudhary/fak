package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV28SplitKVAttention(t *testing.T) {
	want := append(vulkanRegistryV27.stems(), "attention_split_kv")
	if len(want) != 99 || !reflect.DeepEqual(want, vulkanRegistryV28.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("V28 must add exactly the split-KV decode attention module to V27")
	}
	id, err := vulkanRegistryV28.identity()
	schema := vulkanRegistryV28.schema
	if err != nil || schema != VulkanBuildReceiptSchemaV28 || id.ID != VulkanShaderRegistryV28ID || id.ModuleCount != 99 {
		t.Fatalf("V28 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV27.validateStems(want) == nil || vulkanRegistryV28.validateStems(vulkanRegistryV27.stems()) == nil {
		t.Fatal("historical V27 and split-KV V28 must not be interchangeable")
	}
	if !vulkanRegistryV28.usesQualifiedQ2TokenPair() || !vulkanRegistryV28.usesQualifiedIndexScale() {
		t.Fatal("V28 must keep every inherited canonical-module verification")
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanReceiptV29RawFiniteStatus(t *testing.T) {
	want := append(vulkanRegistryV28.stems(), "v41_f32_finite")
	if len(want) != 100 || !reflect.DeepEqual(want, vulkanRegistryV29.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST versioned receipt membership is immutable
		t.Fatal("V29 must add exactly the optional finite-status module to V28")
	}
	id, err := vulkanRegistryV29.identity()
	schema := vulkanRegistryV29.schema
	if err != nil || schema != VulkanBuildReceiptSchemaV29 || id.ID != VulkanShaderRegistryV29ID || id.ModuleCount != 100 {
		t.Fatalf("V29 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV28.validateStems(want) == nil || vulkanRegistryV29.validateStems(vulkanRegistryV28.stems()) == nil {
		t.Fatal("historical V28 and finite-status V29 must not be interchangeable")
	}
	if vulkanSuccessorForSchema(VulkanBuildReceiptSchemaV29) != vulkanRegistryV29 ||
		!vulkanRegistryV29.usesQualifiedQ2TokenPair() || !vulkanRegistryV29.usesQualifiedIndexScale() {
		t.Fatal("V29 must remain a registered successor with inherited qualification")
	}
}
