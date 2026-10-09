package computebuild

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2s lane=default
// Refs #13668. These are identity fixtures, never compiled shader or GPU evidence.
func TestVulkanReceiptV3CurrentRegistryBoundary(t *testing.T) {
	legacy := newVulkanVerifierFixture(t)
	legacyRaw, err := os.ReadFile(legacy.evidence.ReceiptPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := verifyVulkanFixture(context.Background(), legacy, legacy.evidence); err != nil {
		t.Fatalf("historical V2/59 fixture: %v", err)
	}
	if _, err := decodeStrictVulkanReceiptV3(legacyRaw); err == nil {
		t.Fatal("historical V2 envelope accepted as current V3")
	}

	// Extend only this isolated fixture; never rewrite the historical fixture constructor.
	writeFixtureFile(t, filepath.Join(legacy.evidence.SPIRVRoot, "v41_tail_rope_qk.spv"), "inert v41 fixture bytes\n")
	runFixtureGit(t, legacy.evidence.SourceRoot, "add", "-f", ".")
	runFixtureGit(t, legacy.evidence.SourceRoot, "commit", "-q", "-m", "current60 fixture")
	evidence := legacy.evidence
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = CurrentVulkanShaderRegistry()
	observed, err := observeVulkanReceiptEvidenceV3(context.Background(), evidence, legacy.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := currentVulkanRegistryIdentity()
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV3, Backend: "vulkan", Command: "binary", Outcome: "success",
		ReceiptPath: evidence.ReceiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan: &VulkanBuildProvenance{
			Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount,
			Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256,
			NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256,
			StableIdentitySHA256: observed.StableIdentitySHA256,
		},
		VulkanRegistry: &registry, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	if err := WriteReceiptAtomic(evidence.ReceiptPath, &receipt); err != nil {
		t.Fatal(err)
	}
	verify := func(e VulkanBinaryReceiptEvidence) error {
		if runtime.GOOS == "linux" {
			_, err := VerifyVulkanBinaryReceiptIdentityForSchema(context.Background(), VulkanBuildReceiptSchemaV3, e)
			return err
		}
		_, err := verifyVulkanBinaryReceiptV3IdentityWithRunners(context.Background(), e, legacy.gitRun, legacy.gitRun)
		return err
	}
	if err := verify(evidence); err != nil {
		t.Fatalf("current V3/60 fixture: %v", err)
	}
	currentRaw, err := os.ReadFile(evidence.ReceiptPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeStrictVulkanReceipt(currentRaw); err == nil {
		t.Fatal("V2 strict decoder accepted V3-only registry metadata")
	}
	if _, err := verifyVulkanFixture(context.Background(), legacy, evidence); err == nil {
		t.Fatal("V3/60 passed the historical V2 verifier")
	}
	if registry.ModuleCount != 60 || receipt.Vulkan.SPIRVModuleCount != 60 {
		t.Fatalf("current registry or observation is not 60: %+v", registry)
	}
	t.Run("generic and nested serialization", func(t *testing.T) {
		for _, value := range []any{receipt, &receipt, VulkanBinaryReceiptIdentityVerification{Receipt: receipt}} {
			raw, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(raw, []byte(`"shader_registry":`)) || !bytes.Contains(raw, []byte(VulkanShaderRegistryV3ID)) {
				t.Fatal("V3 registry lost during generic JSON serialization")
			}
		}
	})
	t.Run("Windows exact registry parity", func(t *testing.T) {
		raw, err := os.ReadFile(filepath.Join("..", "compute", "build_vulkan.ps1"))
		if err != nil {
			t.Fatal(err)
		}
		parts := strings.SplitN(string(raw), "function Build-Shaders {", 2)
		if len(parts) != 2 {
			t.Fatal("Windows Build-Shaders registry not found")
		}
		block := strings.SplitN(parts[1], "foreach ($s in $shaders)", 2)[0]
		var windows []string
		for _, match := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllStringSubmatch(block, -1) {
			windows = append(windows, match[1])
		}
		if err := validateCurrentVulkanRegistry(windows); err != nil {
			t.Fatalf("Windows and native Go current registry differ: %v", err)
		}
	})
	t.Run("registry is a copy", func(t *testing.T) {
		stems := CurrentVulkanShaderRegistry()
		stems[0] = "forged"
		if CurrentVulkanShaderRegistry()[0] == "forged" {
			t.Fatal("caller mutated the trusted registry")
		}
	})
	for _, tc := range []struct {
		name string
		edit func(*ComputeBuildReceipt)
	}{
		{"missing registry", func(r *ComputeBuildReceipt) { r.VulkanRegistry = nil }},
		{"wrong registry", func(r *ComputeBuildReceipt) { r.VulkanRegistry.ID += ".forged" }},
		{"wrong registry digest", func(r *ComputeBuildReceipt) { r.VulkanRegistry.SHA256 = strings.Repeat("0", 64) }},
		{"wrong registry count", func(r *ComputeBuildReceipt) { r.VulkanRegistry.ModuleCount-- }},
		{"wrong bundle digest", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64) }},
		{"schema mismatch", func(r *ComputeBuildReceipt) { r.Schema = VulkanBuildReceiptSchema }},
		{"wrong source", func(r *ComputeBuildReceipt) { r.Vulkan.Source.GitTree = strings.Repeat("0", 40) }},
		{"wrong tool", func(r *ComputeBuildReceipt) { r.Vulkan.Toolchain[0].SHA256 = strings.Repeat("0", 64) }},
		{"wrong command", func(r *ComputeBuildReceipt) { r.Vulkan.NormalizedBuildCommand[0] += "|forged" }},
		{"wrong binary", func(r *ComputeBuildReceipt) { r.Artifact.SHA256 = strings.Repeat("0", 64) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed, err := decodeStrictVulkanReceiptV3(currentRaw)
			if err != nil {
				t.Fatal(err)
			}
			tc.edit(&changed)
			// Repair downstream claims as an attacker could. Independent observations,
			// rather than merely a stale checksum, must reject the forged identity.
			forgeVerifierReceiptDownstream(t, &changed, evidence)
			changed.Vulkan.StableIdentitySHA256, err = vulkanStableIdentityV3SHA(changed.Vulkan.Source,
				changed.Vulkan.SPIRVBundleSHA256, changed.Vulkan.SPIRVModuleCount,
				changed.Vulkan.ToolchainSHA256, changed.Vulkan.BuildCommandSHA256, changed.Artifact.SHA256)
			if err != nil {
				t.Fatal(err)
			}
			if err := WriteReceiptAtomic(evidence.ReceiptPath, &changed); err != nil {
				t.Fatal(err)
			}
			defer os.WriteFile(evidence.ReceiptPath, currentRaw, 0644)
			if err := verify(evidence); err == nil {
				t.Fatal("forged current envelope accepted")
			}
		})
	}
	for _, tc := range []struct {
		name string
		edit func([]string) []string
	}{
		{"missing", func(s []string) []string { return s[:len(s)-1] }},
		{"additional", func(s []string) []string { return append(s, "forged") }},
		{"duplicate", func(s []string) []string { s[len(s)-1] = s[0]; return s }},
		{"substituted", func(s []string) []string { s[len(s)-1] = "forged"; return s }},
	} {
		t.Run(tc.name+" trusted registry", func(t *testing.T) {
			changed := evidence
			changed.ExpectedSPIRVModules = tc.edit(CurrentVulkanShaderRegistry())
			if err := verify(changed); err == nil {
				t.Fatal("caller-defined registry accepted")
			}
		})
	}
	t.Run("missing module", func(t *testing.T) {
		path := filepath.Join(evidence.SPIRVRoot, "v41_tail_rope_qk.spv")
		if err := os.Rename(path, path+".parked"); err != nil {
			t.Fatal(err)
		}
		defer os.Rename(path+".parked", path)
		if _, _, err := hashCurrentSPIRVBundle(evidence.SPIRVRoot); err == nil {
			t.Fatal("missing current module accepted")
		}
	})
	t.Run("additional module", func(t *testing.T) {
		path := filepath.Join(evidence.SPIRVRoot, "forged.spv")
		writeFixtureFile(t, path, "inert extra fixture\n")
		defer os.Remove(path)
		if _, _, err := hashCurrentSPIRVBundle(evidence.SPIRVRoot); err == nil {
			t.Fatal("additional module accepted")
		}
	})
	t.Run("symlink module", func(t *testing.T) {
		path := filepath.Join(evidence.SPIRVRoot, "v41_tail_rope_qk.spv")
		if err := os.Rename(path, path+".parked"); err != nil {
			t.Fatal(err)
		}
		defer os.Rename(path+".parked", path)
		if err := os.Symlink(path+".parked", path); err != nil {
			t.Skipf("symlink creation unavailable: %v", err)
		}
		defer os.Remove(path)
		if _, _, err := hashCurrentSPIRVBundle(evidence.SPIRVRoot); err == nil {
			t.Fatal("symlink current module accepted")
		}
	})
	t.Run("unknown or trailing JSON", func(t *testing.T) {
		unknown := append([]byte(nil), bytes.TrimSpace(currentRaw)...)
		unknown = append(unknown[:len(unknown)-1], []byte(`,"modules":["forged"]}`)...)
		for _, raw := range [][]byte{unknown, append(append([]byte(nil), currentRaw...), []byte("{}\n")...)} {
			if _, err := decodeStrictVulkanReceiptV3(raw); err == nil {
				t.Fatal("unknown registry list or trailing JSON accepted")
			}
		}
	})
	t.Run("cross-schema comparison", func(t *testing.T) {
		path := filepath.Join(legacy.root, "historical.json")
		if err := os.WriteFile(path, legacyRaw, 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := compareVulkanBuildReceiptV3(path, receipt.Artifact, receipt.Vulkan, receipt.VulkanRegistry); err == nil {
			t.Fatal("current build compared equal to historical V2")
		}
		forged := receipt
		forged.Schema = VulkanBuildReceiptSchema
		if err := CompareReceiptProvenance(&receipt, &forged); err == nil {
			t.Fatal("cross-schema stable identity accepted")
		}
	})
	t.Run("V3 comparison checks declared fields", func(t *testing.T) {
		if err := CompareReceiptProvenance(&receipt, &receipt); err != nil {
			t.Fatalf("same current provenance rejected: %v", err)
		}
		missing := receipt
		missing.Vulkan = nil
		if err := CompareReceiptProvenance(&missing, &missing); err == nil {
			t.Fatal("V3 without provenance fell through historical comparison")
		}
		for _, edit := range []func(*ComputeBuildReceipt){
			func(r *ComputeBuildReceipt) { r.Outcome = "failed" },
			func(r *ComputeBuildReceipt) { r.Backend = "cuda" },
			func(r *ComputeBuildReceipt) { r.GitCommit = evidence.ExpectedCommit },
		} {
			invalid := receipt
			edit(&invalid)
			if err := CompareReceiptProvenance(&receipt, &invalid); err == nil {
				t.Fatal("copied identity concealed an invalid V3 envelope")
			}
		}
		forged, err := decodeStrictVulkanReceiptV3(currentRaw)
		if err != nil {
			t.Fatal(err)
		}
		forged.Vulkan.Source.GitTree = strings.Repeat("0", 40)
		if err := CompareReceiptProvenance(&receipt, &forged); err == nil {
			t.Fatal("same hashes hid a changed source identity")
		}
	})
	t.Run("historical wire remains unchanged", func(t *testing.T) {
		old, err := decodeStrictVulkanReceipt(legacyRaw)
		if err != nil {
			t.Fatal(err)
		}
		base, err := json.Marshal(old)
		if err != nil {
			t.Fatal(err)
		}
		old.VulkanRegistry = &registry
		withInMemoryRegistry, err := json.Marshal(buildReceiptWireValue(&old))
		if err != nil {
			t.Fatal(err)
		}
		if string(base) != string(withInMemoryRegistry) {
			t.Fatal("in-memory V3 metadata changed the V2 wire")
		}
		// Match the old standard-library wire encoding, including HTML-sensitive
		// content, for both default Marshal and the production non-escaping encoder.
		old.Error = "<historical> & preserved"
		for _, escape := range []bool{false, true} {
			var before, after bytes.Buffer
			legacyEncoder := json.NewEncoder(&before)
			legacyEncoder.SetEscapeHTML(escape)
			legacyEncoder.SetIndent("", "  ")
			if err := legacyEncoder.Encode(computeBuildReceiptV2Wire(old)); err != nil {
				t.Fatal(err)
			}
			currentEncoder := json.NewEncoder(&after)
			currentEncoder.SetEscapeHTML(escape)
			currentEncoder.SetIndent("", "  ")
			if err := currentEncoder.Encode(old); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before.Bytes(), after.Bytes()) {
				t.Fatalf("V2 wire bytes changed with escapeHTML=%t", escape)
			}
		}
	})
}
