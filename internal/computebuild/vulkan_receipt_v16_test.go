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
func TestVulkanReceiptV16Current79Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV15(), "v41_shared_attention_fast")
	if len(want) != 79 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV16(), want) || !reflect.DeepEqual(VulkanShaders, want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("current builder does not use V15 plus the run-to-run shared attention module exactly once")
	}
	registry, err := vulkanRegistryV16.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v16.current79" || registry.ModuleCount != 79 || registry.SHA256 != "be6571e97ca0e946cedf8b9f5d858cac5065b35fda2204ce7a23318cbd322a2a" {
		t.Fatalf("V16 registry identity differs from independent JSON vector: %+v", registry)
	}
	if vulkanRegistryV15.validateStems(want) == nil || vulkanRegistryV16.validateStems(CurrentVulkanShaderRegistryV15()) == nil {
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V16 exact79 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v16-receipt.json")
	if _, err := vulkanRegistryV15.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V15 policy admitted the exact79 census")
	}
	observed, err := vulkanRegistryV16.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV16, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV16.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V16 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV15.decodeStrict(raw); err == nil {
		t.Fatal("V15 decoder admitted a V16 receipt")
	}
	decoded, err := vulkanRegistryV16.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V16 round trip: %v", err)
	}
	asV15 := decoded
	asV15.Schema = VulkanBuildReceiptSchemaV15
	if CompareReceiptProvenance(&asV15, &receipt) == nil || vulkanRegistryV15.validateSuccess(&asV15) == nil {
		t.Fatal("V16 evidence accepted under the V15 schema")
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_shared_attention_fast.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV16.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V16 bundle without v41_shared_attention_fast accepted")
	}
}
