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
// Ref #13815. Fixtures exercise identity/structure only, not a compiler,
// executable SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV10Current69Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV9(), "v41_routed_swiglu")
	if len(want) != 69 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV10(), want) || !reflect.DeepEqual(VulkanShaders[:69], want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest()[:69], want) {
		t.Fatal("V10 is not V9 plus the routed SwiGLU module as the current builder prefix")
	}
	registry, err := vulkanRegistryV10.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v10.current69" || registry.ModuleCount != 69 || registry.SHA256 != "e8a725e0b2cdb34cbbef3983460784d512d5106f6135765eca3d5942c3aa4d4a" {
		t.Fatalf("V10 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV9.validateStems(want) == nil || vulkanRegistryV10.validateStems(CurrentVulkanShaderRegistryV9()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V10 exact69 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v10-receipt.json")
	if _, err := vulkanRegistryV9.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V9 policy admitted the exact69 census")
	}
	observed, err := vulkanRegistryV10.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV10, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV10.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V10 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV9.decodeStrict(raw); err == nil {
		t.Fatal("V9 decoder admitted a V10 receipt")
	}
	decoded, err := vulkanRegistryV10.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V10 round trip: %v", err)
	}
	asV9 := decoded
	asV9.Schema = VulkanBuildReceiptSchemaV9
	if CompareReceiptProvenance(&asV9, &receipt) == nil || vulkanRegistryV9.validateSuccess(&asV9) == nil {
		t.Fatal("V10 evidence accepted under the V9 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_routed_swiglu.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV10.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V10 bundle without v41_routed_swiglu accepted")
	}
}
