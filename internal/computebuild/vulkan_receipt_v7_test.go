package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
// Each successor row is self-consistent and admitting a module only appends it;
// adding one never needs a per-version receipt implementation.
func TestVulkanSuccessorRegistryTable(t *testing.T) {
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	if vulkanCurrentRegistry != vulkanSuccessorRegistries[len(vulkanSuccessorRegistries)-1] {
		t.Fatal("current registry is not the newest successor row")
	}
	if !reflect.DeepEqual(VulkanShaders, vulkanCurrentRegistry.stems()) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest(), VulkanShaders) {
		t.Fatal("builder snapshot is not the current successor registry")
	}
	schemas, ids := map[string]bool{}, map[string]bool{}
	prior := CurrentVulkanShaderRegistryV4()
	for _, r := range vulkanSuccessorRegistries {
		stems := r.stems()
		if len(stems) != r.count || r.validateStems(stems) != nil {
			t.Fatalf("%s lists %d modules, policy count %d", r.version, len(stems), r.count)
		}
		if !slices.Equal(stems[:len(prior)], prior) || !slices.Equal(stems[len(prior):], r.added) {
			t.Fatalf("%s does not append %v to its predecessor", r.version, r.added)
		}
		if schemas[r.schema] || ids[r.id] || vulkanSuccessorForSchema(r.schema) != r {
			t.Fatalf("%s schema or id is not unique", r.version)
		}
		schemas[r.schema], ids[r.id] = true, true
		for _, stem := range r.added {
			if !FileExists(filepath.Join(repoRoot, "internal", "compute", "shaders", stem+".comp")) {
				t.Fatalf("%s admits %s without shader source", r.version, stem)
			}
		}
		prior = stems
	}
}

// fak-test:runtime medium est=6s lane=default
// Fixtures exercise identity/structure only, not a compiler, executable
// SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV7Current66Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV6(), "v41_router_topk", "v41_index_select", "v41_q2k_embed_gather")
	if len(want) != 66 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV7(), want) || !reflect.DeepEqual(VulkanShaders[:66], want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("V7 is not the exact66 registry prefix of the current builder")
	}
	registry, err := vulkanRegistryV7.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v7.current66" || registry.ModuleCount != 66 || registry.SHA256 != "722908f36014a28c16b2dcac9e2d807cd13c16c42dc966c927fef07c80feb9df" {
		t.Fatalf("V7 registry identity differs from independent JSON vector: %+v", registry)
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
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V7 exact66 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v7-receipt.json")
	if _, err := vulkanRegistryV6.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("V6 policy admitted the exact66 census")
	}
	observed, err := vulkanRegistryV7.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV7, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV7.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("V7 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV6.decodeStrict(raw); err == nil {
		t.Fatal("V6 decoder admitted a V7 receipt")
	}
	decoded, err := vulkanRegistryV7.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V7 round trip: %v", err)
	}
	asV6 := decoded
	asV6.Schema = VulkanBuildReceiptSchemaV6
	if CompareReceiptProvenance(&asV6, &receipt) == nil || vulkanRegistryV6.validateSuccess(&asV6) == nil {
		t.Fatal("V7 evidence accepted under the V6 schema")
	}
	for _, stem := range []string{"v41_router_topk", "v41_index_select", "v41_q2k_embed_gather"} {
		path := filepath.Join(evidence.SPIRVRoot, stem+".spv")
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := vulkanRegistryV7.hashBundle(evidence.SPIRVRoot); err == nil {
			t.Fatalf("V7 bundle without %s accepted", stem)
		}
		writeFixtureFile(t, path, "inert historical shader\n")
	}
}
