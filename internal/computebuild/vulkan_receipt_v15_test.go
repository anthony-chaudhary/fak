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
// Ref #13668. Fixtures exercise identity/structure only, not a compiler,
// executable SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV15Current78Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV14(), "v41_q2k_matvec_id", "v41_q3k_matvec_id", "v41_routed_swiglu_id", "v41_routed_publish_sum")
	if len(want) != 78 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV15(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder does not use V14 plus the four routed-expert id modules exactly once")
	}
	registry, err := vulkanRegistryV15.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v15.current78" || registry.ModuleCount != 78 || registry.SHA256 != "815c3a8f7a562e13cc9819c2ba5163d6acc68f2c62e6233fa55627c9f8a7b127" {
		t.Fatalf("V15 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV14.validateStems(want) == nil || vulkanRegistryV15.validateStems(CurrentVulkanShaderRegistryV14()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V15 exact78 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v15-receipt.json")
	if _, err := vulkanRegistryV14.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V14 policy admitted the exact78 census")
	}
	observed, err := vulkanRegistryV15.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV15, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV15.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V15 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV14.decodeStrict(raw); err == nil {
		t.Fatal("V14 decoder admitted a V15 receipt")
	}
	decoded, err := vulkanRegistryV15.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V15 round trip: %v", err)
	}
	asV14 := decoded
	asV14.Schema = VulkanBuildReceiptSchemaV14
	if CompareReceiptProvenance(&asV14, &receipt) == nil || vulkanRegistryV14.validateSuccess(&asV14) == nil {
		t.Fatal("V15 evidence accepted under the V14 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_routed_publish_sum.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV15.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V15 bundle without v41_routed_publish_sum accepted")
	}
}
