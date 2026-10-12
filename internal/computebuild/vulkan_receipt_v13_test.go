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
func TestVulkanReceiptV13Current73Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV12(), "v41_scale_exact")
	if len(want) != 73 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV13(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder does not use V12 plus the strict scalar-multiply module exactly once")
	}
	registry, err := vulkanRegistryV13.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v13.current73" || registry.ModuleCount != 73 || registry.SHA256 != "dd7028349a8865b897849f319e9ebc06601d5dc3e5b820b12478c15c1e9a182b" {
		t.Fatalf("V13 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV12.validateStems(want) == nil || vulkanRegistryV13.validateStems(CurrentVulkanShaderRegistryV12()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V13 exact73 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v13-receipt.json")
	if _, err := vulkanRegistryV12.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V12 policy admitted the exact73 census")
	}
	observed, err := vulkanRegistryV13.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV13, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV13.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V13 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV12.decodeStrict(raw); err == nil {
		t.Fatal("V12 decoder admitted a V13 receipt")
	}
	decoded, err := vulkanRegistryV13.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V13 round trip: %v", err)
	}
	asV12 := decoded
	asV12.Schema = VulkanBuildReceiptSchemaV12
	if CompareReceiptProvenance(&asV12, &receipt) == nil || vulkanRegistryV12.validateSuccess(&asV12) == nil {
		t.Fatal("V13 evidence accepted under the V12 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_scale_exact.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV13.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V13 bundle without v41_scale_exact accepted")
	}
}
