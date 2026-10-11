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
func TestVulkanReceiptV8Current67Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV7(), "v41_exact_rmsnorm")
	if len(want) != 67 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV8(), want) || !reflect.DeepEqual(VulkanShaders, want) {
		t.Fatal("current builder does not use the exact67 registry")
	}
	registry, err := vulkanRegistryV8.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v8.current67" || registry.ModuleCount != 67 || registry.SHA256 != "13644de2f4f58b564d1241174b7d5ae1995c36f179867a1d86c00ea0000d47e9" {
		t.Fatalf("V8 registry identity differs from independent JSON vector: %+v", registry)
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V8 exact67 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v8-receipt.json")
	if _, err := vulkanRegistryV7.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V7 policy admitted the exact67 census")
	}
	observed, err := vulkanRegistryV8.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV8, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV8.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V8 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV7.decodeStrict(raw); err == nil {
		t.Fatal("V7 decoder admitted a V8 receipt")
	}
	decoded, err := vulkanRegistryV8.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V8 round trip: %v", err)
	}
	asV7 := decoded
	asV7.Schema = VulkanBuildReceiptSchemaV7
	if CompareReceiptProvenance(&asV7, &receipt) == nil || vulkanRegistryV7.validateSuccess(&asV7) == nil {
		t.Fatal("V8 evidence accepted under the V7 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_exact_rmsnorm.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV8.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V8 bundle without v41_exact_rmsnorm accepted")
	}
}
