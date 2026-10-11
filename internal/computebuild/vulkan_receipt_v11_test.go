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
// Refs #13811, #13812. Fixtures exercise identity/structure only, not a compiler,
// executable SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV11Current71Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV10(), "v41_engram_gate", "v41_compressor_pool")
	if len(want) != 71 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV11(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) {
		t.Fatal("current builder does not use V10 plus the Engram gate and compressor pool modules exactly once")
	}
	registry, err := vulkanRegistryV11.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v11.current71" || registry.ModuleCount != 71 || registry.SHA256 != "c032225a0fa215d5cea54ffd8d2c5bdc1b14c66685959ae4913e6521ca5cde2a" {
		t.Fatalf("V11 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV10.validateStems(want) == nil || vulkanRegistryV11.validateStems(CurrentVulkanShaderRegistryV10()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V11 exact71 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v11-receipt.json")
	if _, err := vulkanRegistryV10.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V10 policy admitted the exact71 census")
	}
	observed, err := vulkanRegistryV11.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV11, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV11.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V11 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV10.decodeStrict(raw); err == nil {
		t.Fatal("V10 decoder admitted a V11 receipt")
	}
	decoded, err := vulkanRegistryV11.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V11 round trip: %v", err)
	}
	asV10 := decoded
	asV10.Schema = VulkanBuildReceiptSchemaV10
	if CompareReceiptProvenance(&asV10, &receipt) == nil || vulkanRegistryV10.validateSuccess(&asV10) == nil {
		t.Fatal("V11 evidence accepted under the V10 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_engram_gate.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV11.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V11 bundle without v41_engram_gate accepted")
	}
}
