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
// Ref #13816. Fixtures exercise identity/structure only, not a compiler,
// executable SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV12Current72Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV11(), "v41_tail_rope_exact")
	if len(want) != 72 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV12(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder does not use V11 plus the strict tail-RoPE module exactly once")
	}
	registry, err := vulkanRegistryV12.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v12.current72" || registry.ModuleCount != 72 || registry.SHA256 != "0863e4b09e7ebda6085b4914cae7430640a760649de5445c7220af8fbb46d97a" {
		t.Fatalf("V12 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV11.validateStems(want) == nil || vulkanRegistryV12.validateStems(CurrentVulkanShaderRegistryV11()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V12 exact72 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v12-receipt.json")
	if _, err := vulkanRegistryV11.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V11 policy admitted the exact72 census")
	}
	observed, err := vulkanRegistryV12.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV12, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV12.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V12 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV11.decodeStrict(raw); err == nil {
		t.Fatal("V11 decoder admitted a V12 receipt")
	}
	decoded, err := vulkanRegistryV12.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V12 round trip: %v", err)
	}
	asV11 := decoded
	asV11.Schema = VulkanBuildReceiptSchemaV11
	if CompareReceiptProvenance(&asV11, &receipt) == nil || vulkanRegistryV11.validateSuccess(&asV11) == nil {
		t.Fatal("V12 evidence accepted under the V11 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_tail_rope_exact.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV12.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V12 bundle without v41_tail_rope_exact accepted")
	}
}
