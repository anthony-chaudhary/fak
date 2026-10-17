package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV28SplitKVAttention(t *testing.T) {
	want := append(vulkanRegistryV27.stems(), "attention_split_kv")
	if len(want) != 99 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("V28 must add exactly the split-KV decode attention module to V27")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
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
