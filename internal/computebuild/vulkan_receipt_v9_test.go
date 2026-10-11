package computebuild

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime medium est=8s lane=default
// The estimate is unmeasured. Inert artifacts exercise canonical identity
// admission, not shader semantics, build causality, or device qualification.
func TestVulkanReceiptV9Current68Boundary(t *testing.T) {
	want := append(CurrentVulkanShaderRegistryV8(), "v41_bf16_checked")
	if len(want) != 68 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV9(), want) || !reflect.DeepEqual(VulkanShaders[:68], want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryLatest()[:68], want) { //boundarylint:ignore CHANGE_DETECTOR_TEST a versioned receipt registry is immutable by design; the count is its identity
		t.Fatal("V9 is not V8 plus the checked BF16 module as the current builder prefix")
	}
	registry, err := vulkanRegistryV9.identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v9.current68" || registry.ModuleCount != 68 || registry.SHA256 != "57c7bd4cc6b6f2ad4a3ecfc4498e758a0f5de1b4d4d620bb749ccbaa5a659344" {
		t.Fatalf("V9 identity differs from the independent sorted-name JSON vector: %+v", registry)
	}
	historical, err := vulkanRegistryV8.identity()
	if err != nil || historical.ID != "fak.vulkan-shader-registry.v8.current67" || historical.ModuleCount != 67 || historical.SHA256 != "13644de2f4f58b564d1241174b7d5ae1995c36f179867a1d86c00ea0000d47e9" {
		t.Fatalf("historical V8 identity changed: %+v, %v", historical, err)
	}
	copy := CurrentVulkanShaderRegistryV9()
	copy[len(copy)-1] = "forged"
	if !reflect.DeepEqual(CurrentVulkanShaderRegistryV9(), want) || !reflect.DeepEqual(CurrentVulkanShaderRegistryV8(), want[:67]) {
		t.Fatal("caller mutation changed a trusted registry")
	}
	for _, tc := range []struct {
		name string
		edit func([]string) []string
	}{
		{"missing", func(s []string) []string { return s[:67] }},
		{"duplicate", func(s []string) []string { s[67] = s[66]; return s }},
		{"substituted witness module", func(s []string) []string { s[67] = "v41_bf16_boundary"; return s }},
		{"extra", func(s []string) []string { return append(s, "v41_bf16_boundary") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if vulkanRegistryV9.validateStems(tc.edit(append([]string(nil), want...))) == nil {
				t.Fatal("noncanonical V9 membership accepted")
			}
		})
	}
	if vulkanRegistryV8.validateStems(want) == nil || vulkanRegistryV9.validateStems(CurrentVulkanShaderRegistryV8()) == nil {
		t.Fatal("historical and current registries are interchangeable")
	}

	ctx := context.Background()
	fixture := newVulkanVerifierFixture(t)
	evidence := fixture.evidence
	for _, stem := range want[59:] {
		if stem != "v41_indexer_score" {
			writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stem+".spv"), "inert shader identity\n")
		}
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv"), string(vulkanV5StructuralFixture()))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal", "compute", "libfakvulkan.a"), "inert ordinary native archive identity\n")
	runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V9 exact68 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v9-receipt.json")
	if _, err := vulkanRegistryV8.observeEvidence(ctx, evidence, fixture.gitRun); err == nil {
		t.Fatal("historical V8 verifier admitted the current68 bundle")
	}
	observed, err := vulkanRegistryV9.observeEvidence(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV9, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan: &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount,
			Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand,
			BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore,
		Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV9.verifyIdentityWithRunners(ctx, evidence, fixture.gitRun, nil); err != nil {
		t.Fatalf("current V9 receipt did not verify: %v", err)
	}
	if _, err := vulkanRegistryV8.decodeStrict(raw); err == nil {
		t.Fatal("historical V8 decoder admitted V9 evidence")
	}
	decoded, err := vulkanRegistryV9.decodeStrict(raw)
	if err != nil || CompareReceiptProvenance(&decoded, &receipt) != nil {
		t.Fatalf("V9 round trip failed: %v", err)
	}
	clone := func(t *testing.T) ComputeBuildReceipt {
		t.Helper()
		c, err := vulkanRegistryV9.decodeStrict(raw)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	rebind := func(t *testing.T, c *ComputeBuildReceipt) {
		t.Helper()
		p := c.Vulkan
		var err error
		p.BuildCommandSHA256, err = hashJSON(p.NormalizedBuildCommand)
		if err == nil {
			p.StableIdentitySHA256, err = vulkanRegistryV9.stableIdentitySHA(p.Source, p.SPIRVBundleSHA256, p.SPIRVModuleCount, p.ToolchainSHA256, p.BuildCommandSHA256, c.Artifact.SHA256, *c.VulkanNativeArchive, *c.VulkanIndexerScore)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, profile := range []struct{ define, tag, diagnostic string }{
		{"FAK_V41_INDEXER_SCORE_WITNESS", "v41_indexer_witness", "test-only indexer witness"},
		{"FAK_V41_BF16_BOUNDARY_WITNESS", "v41_bf16_boundary_witness", "test-only BF16 boundary witness"},
		{"FAK_V41_TAIL_ROPE_EXACT_WITNESS", "v41_tail_rope_exact_witness", "test-only exact tail RoPE witness"},
		{"FAK_V41_LOGIT_SCALE_WITNESS", "v41_logit_scale_witness", "test-only logit scale witness"},
	} {
		for _, command := range []string{"cxx|-D" + profile.define, "go|build|-tags|vulkan," + profile.tag, "link|-lfakvulkan_" + profile.tag} {
			t.Run("witness "+command, func(t *testing.T) {
				c := clone(t)
				c.Vulkan.NormalizedBuildCommand = append(c.Vulkan.NormalizedBuildCommand, command)
				rebind(t, &c)
				if err := vulkanRegistryV9.validateSuccess(&c); err == nil || !strings.Contains(err.Error(), profile.diagnostic) {
					t.Fatalf("witness profile not specifically refused with repaired digests: %v", err)
				}
			})
		}
	}
	for _, tc := range []struct {
		name string
		edit func(*ComputeBuildReceipt)
	}{
		{"historical schema", func(c *ComputeBuildReceipt) { c.Schema = VulkanBuildReceiptSchemaV8 }},
		{"historical registry", func(c *ComputeBuildReceipt) { *c.VulkanRegistry = historical }},
		{"wrong registry digest", func(c *ComputeBuildReceipt) { c.VulkanRegistry.SHA256 = strings.Repeat("0", 64) }},
		{"wrong archive digest", func(c *ComputeBuildReceipt) { c.VulkanNativeArchive.SHA256 = strings.Repeat("0", 64) }},
		{"wrong binary digest", func(c *ComputeBuildReceipt) { c.Artifact.SHA256 = strings.Repeat("0", 64) }},
		{"wrong bundle digest", func(c *ComputeBuildReceipt) { c.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := clone(t)
			tc.edit(&c)
			rebind(t, &c)
			if err := vulkanRegistryV9.compareToObservation(c, observed); err == nil {
				t.Fatal("changed artifact identity accepted despite fixed observed bytes")
			}
		})
	}
	path := filepath.Join(evidence.SPIRVRoot, "v41_bf16_checked.spv")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := vulkanRegistryV9.hashBundle(evidence.SPIRVRoot); err == nil {
		t.Fatal("V9 bundle missing checked BF16 accepted")
	}
}
