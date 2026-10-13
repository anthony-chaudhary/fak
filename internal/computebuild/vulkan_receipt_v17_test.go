package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime medium est=6s lane=default
// Ref #12640. Fixtures exercise identity/structure only, not a compiler,
// executable SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV17Current81Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV16(), "v41_routed_gate_up_swiglu_id", "v41_routed_down_sum")
	if len(want) != 81 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV17(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("historical V17 is not V16 plus the two fused routed-FFN modules exactly once")
	}
	registry, err := vulkanRegistryV17.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v17.current81" || registry.ModuleCount != 81 || registry.SHA256 != "c06385812cd06425e906c1ab1e469cf29a42e54481483d00fd9c44ab6ff7198b" {
		t.Fatalf("V17 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV16.validateStems(want) == nil || vulkanRegistryV17.validateStems(CurrentVulkanShaderRegistryV16()) == nil {
		t.Fatal("historical and current registries are interchangeable")
	}

	ctx := context.Background()
	fixture := newVulkanVerifierFixture(t)
	evidence := fixture.evidence
	for _, stem := range want[59:] {
		if stem != "v41_indexer_score" {
			writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stem+".spv"), "inert historical shader\n")
		}
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv"), string(vulkanV5StructuralFixture()))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal", "compute", "libfakvulkan.a"), "inert native archive identity\n")
	runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V17 exact81 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v17-receipt.json")
	if _, err := vulkanRegistryV16.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V16 policy admitted the exact81 census")
	}
	observed, err := vulkanRegistryV17.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV17, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV17.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V17 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV16.decodeStrict(raw); err == nil {
		t.Fatal("V16 decoder admitted a V17 receipt")
	}
	decoded, err := vulkanRegistryV17.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V17 round trip: %v", err)
	}
	asV16 := decoded
	asV16.Schema = VulkanBuildReceiptSchemaV16
	if CompareReceiptProvenance(&asV16, &receipt) == nil || vulkanRegistryV16.validateSuccess(&asV16) == nil {
		t.Fatal("V17 evidence accepted under the V16 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_shared_attention_fast.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV17.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V16 bundle without v41_shared_attention_fast accepted")
	}
}
