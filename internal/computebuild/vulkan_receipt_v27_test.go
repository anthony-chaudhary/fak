package computebuild

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms
// Pins only the source/receipt census. No shader or native archive is compiled.
func TestVulkanReceiptV27PrefillGEMM(t *testing.T) {
	want := append(vulkanRegistryV26.stems(), "mm_cm1_q2k", "mm_cm1_iq2xxs", "mm_cm1_iq2xs", "mm_cm1_iq2s", "mm_cm1_iq3xxs", "mm_cm1_iq3s", "mm_cm1_iq1s", "mm_cm1_iq4xs", "mm_cm1_q4k")
	if len(want) != 98 || !reflect.DeepEqual(want, vulkanRegistryV27.stems()) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("V27 must add exactly the nine prefill GEMM modules to V26")
	}
	schema := vulkanRegistryV27.schema
	id, err := vulkanRegistryV27.identity()
	if err != nil || schema != VulkanBuildReceiptSchemaV27 || id.ID != VulkanShaderRegistryV27ID || id.ModuleCount != 98 {
		t.Fatalf("V27 identity differs: %s %+v %v", schema, id, err)
	}
	if vulkanRegistryV26.validateStems(want) == nil || vulkanRegistryV27.validateStems(vulkanRegistryV26.stems()) == nil {
		t.Fatal("historical V26 and prefill-GEMM V27 must not be interchangeable")
	}
	if !vulkanRegistryV27.usesQualifiedQ2TokenPair() || !vulkanRegistryV27.usesQualifiedIndexScale() {
		t.Fatal("V27 must keep every inherited canonical-module verification")
	}
}
