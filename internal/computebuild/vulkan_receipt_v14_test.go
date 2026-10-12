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
func TestVulkanReceiptV14Current74Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV13(), "v41_engram_row_decode")
	if len(want) != 74 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV14(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder does not use V13 plus the Engram row-decode module exactly once")
	}
	registry, err := vulkanRegistryV14.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v14.current74" || registry.ModuleCount != 74 || registry.SHA256 != "ddffbd3c5d28bbd69452ccbd2710df68bcc52cd4da0c9192e8661cec3b724151" {
		t.Fatalf("V14 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV13.validateStems(want) == nil || vulkanRegistryV14.validateStems(CurrentVulkanShaderRegistryV13()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V14 exact74 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v14-receipt.json")
	if _, err := vulkanRegistryV13.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V13 policy admitted the exact74 census")
	}
	observed, err := vulkanRegistryV14.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV14, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV14.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V14 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV13.decodeStrict(raw); err == nil {
		t.Fatal("V13 decoder admitted a V14 receipt")
	}
	decoded, err := vulkanRegistryV14.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V14 round trip: %v", err)
	}
	asV13 := decoded
	asV13.Schema = VulkanBuildReceiptSchemaV13
	if CompareReceiptProvenance(&asV13, &receipt) == nil || vulkanRegistryV13.validateSuccess(&asV13) == nil {
		t.Fatal("V14 evidence accepted under the V13 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_engram_row_decode.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV14.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V14 bundle without v41_engram_row_decode accepted")
	}
}
