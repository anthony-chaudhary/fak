package computebuild

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func qualifiedIndexScaleFixture(t *testing.T) (string, []byte, []byte) {
	t.Helper()
	root := t.TempDir()
	for _, rel := range []string{"shaders/v41_index_scale_bf16.comp", "v41_index_scale_bf16_spirv.h"} {
		data, err := os.ReadFile(filepath.Join("..", "compute", filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		dst := filepath.Join(root, "internal", "compute", filepath.FromSlash(rel))
		if err = os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(dst, data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	source, err := os.ReadFile(filepath.Join(root, "internal/compute/shaders/v41_index_scale_bf16.comp"))
	if err != nil {
		t.Fatal(err)
	}
	header, err := os.ReadFile(filepath.Join(root, vulkanQualifiedIndexScaleHeader))
	if err != nil {
		t.Fatal(err)
	}
	return root, source, header
}

// fak-test:runtime fast est=10ms
func TestVulkanQualifiedArtifact(t *testing.T) {
	root, source, header := qualifiedIndexScaleFixture(t)
	const arrayStart = "static const unsigned char qualified[] = {"
	if bytes.Count(header, []byte(arrayStart)) != 1 {
		t.Fatal("canonical header array framing differs")
	}
	_, encoded, _ := bytes.Cut(header, []byte(arrayStart))
	encoded, _, terminated := bytes.Cut(encoded, []byte("};"))
	if !terminated {
		t.Fatal("canonical header array terminator missing")
	}
	// Materialization must preserve every byte entry in the canonical input.
	// The independent hash below continues to pin the exact module contents.
	wantBytes := bytes.Count(encoded, []byte("0x"))
	if wantBytes == 0 {
		t.Fatal("canonical header array is empty")
	}
	module, err := decodeVulkanQualifiedIndexScale(source, header)
	if err != nil {
		t.Fatal(err)
	}
	if len(module) != wantBytes || !vulkanQualifiedHash(module, "b67ed132187bde41131d355d9d493f56e5063ab87aa0a6a53ea66c8743136500") {
		t.Fatal("canonical module differs")
	}
	for _, pair := range [][2][]byte{{bytes.ReplaceAll(source, []byte("\n"), []byte("\r\n")), header}, {source, bytes.ReplaceAll(header, []byte("\n"), []byte("\r\n"))}, {append(append([]byte(nil), source...), '\n'), header}, {source, append(append([]byte(nil), header...), '\n')}, {source, header[:len(header)-1]}, {source, bytes.Replace(header, []byte("0x03"), []byte("0x02"), 1)}} {
		if _, err := decodeVulkanQualifiedIndexScale(pair[0], pair[1]); err == nil {
			t.Fatal("source/header mutation accepted")
		}
	}
	spv := filepath.Join(root, "internal/compute/spirv")
	if err := os.MkdirAll(spv, 0755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(spv, vulkanQualifiedIndexScaleStem+".spv")
	if err := writeVulkanQualifiedModule(dst, module); err != nil {
		t.Fatal(err)
	}
	if err := verifyVulkanQualifiedIndexScale(root, spv); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{module[:len(module)-1], append(append([]byte(nil), module...), 0), bytes.Repeat([]byte{0}, len(module))} {
		if err := writeVulkanQualifiedModule(dst, bad); err == nil {
			t.Fatal("bad length/hash accepted")
		}
		got, err := os.ReadFile(dst)
		if err != nil || !bytes.Equal(got, module) {
			t.Fatal("rejected output changed prior module")
		}
	}
	if err := os.WriteFile(dst, make([]byte, len(module)), 0644); err != nil {
		t.Fatal(err)
	}
	if err := verifyVulkanQualifiedIndexScale(root, spv); err == nil {
		t.Fatal("corrupted packaged module accepted")
	}
}

// fak-test:runtime fast est=20ms
func TestVulkanQualifiedBuilder(t *testing.T) {
	root, _, _ := qualifiedIndexScaleFixture(t)
	installQualifiedChunkFixture(t, root)
	installQualifiedDownChunkFixture(t, root)
	installQualifiedQ2TokenPairFixture(t, root)
	for _, stem := range vulkanCurrentRegistry.stems() {
		if stem != vulkanQualifiedIndexScaleStem && stem != vulkanQualifiedActivationChunkStem && stem != vulkanQualifiedDownChunkStem && stem != vulkanQualifiedQ2TokenPairStem {
			if err := os.WriteFile(filepath.Join(root, "internal/compute/shaders", stem+".comp"), []byte("fixture"), 0644); err != nil {
				t.Fatal(err)
			}
		}
	}
	calls := 0
	compile := func(src, dst string) error {
		calls++
		if strings.Contains(src, vulkanQualifiedIndexScaleStem) || strings.Contains(src, vulkanQualifiedActivationChunkStem) || strings.Contains(src, vulkanQualifiedDownChunkStem) || strings.Contains(src, vulkanQualifiedQ2TokenPairStem) {
			t.Fatal("canonical module reached local compiler")
		}
		return os.WriteFile(dst, []byte("ordinary compiler fixture"), 0644)
	}
	if err := buildVulkanShaders(root, io.Discard, compile); err != nil {
		t.Fatal(err)
	}
	if calls != len(vulkanCurrentRegistry.stems())-4 {
		t.Fatalf("ordinary compiler calls=%d", calls)
	}
	if err := verifyVulkanQualifiedIndexScale(root, filepath.Join(root, "internal/compute/spirv")); err != nil {
		t.Fatal(err)
	}
	calls = 0
	if err := os.WriteFile(filepath.Join(root, "internal/compute/shaders/v41_index_scale_bf16.comp"), []byte("changed"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := buildVulkanShaders(root, io.Discard, compile); err == nil || calls != 0 {
		t.Fatal("source drift did not fail before compiler effects")
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanReceiptV23Materialization(t *testing.T) {
	if vulkanRegistryV23.count != 86 || !reflect.DeepEqual(vulkanRegistryV23.stems(), vulkanRegistryV22.stems()) {
		t.Fatal("materialization changed membership")
	}
	for _, windows := range []bool{false, true} {
		root := t.TempDir()
		cfg := &VulkanConfig{RepoRoot: root, PkgDir: filepath.Join(root, "internal/compute"), OutPkg: "./cmd/fak", Toolchain: &Toolchain{IsWindows: windows}}
		old, oldHash, err := normalizedVulkanBuildCommand(cfg)
		if err != nil {
			t.Fatal(err)
		}
		legacy, legacyHash, err := vulkanRegistryV22.normalizedBuildCommand(cfg)
		if err != nil || oldHash != legacyHash || !reflect.DeepEqual(old, legacy) {
			t.Fatal("historical V22 recipe changed")
		}
		current, currentHash, err := vulkanRegistryV23.normalizedBuildCommand(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if currentHash == oldHash || len(current) != len(old)+1 || !strings.Contains(current[0], "registered-except-v41_index_scale_bf16") || !strings.HasPrefix(current[1], "qualified-header|") || !strings.Contains(current[1], vulkanQualifiedIndexScaleSourceSHA) || !strings.Contains(current[1], vulkanQualifiedIndexScaleModuleSHA) {
			t.Fatal("canonical materialization provenance missing")
		}
		for _, r := range vulkanSuccessorRegistries {
			if r != vulkanRegistryV23 && r != vulkanRegistryV24 && r != vulkanRegistryV25 && r != vulkanRegistryV26 && r != vulkanRegistryV27 && r != vulkanRegistryV28 && r.usesQualifiedIndexScale() {
				t.Fatal("historical schema acquired materialization claim")
			}
		}
	}
}

// fak-test:runtime medium est=6s lane=default
func TestVulkanReceiptV23CanonicalEvidence(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	evidence := fixture.evidence
	_, source, header := qualifiedIndexScaleFixture(t)
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/shaders/v41_index_scale_bf16.comp"), string(source))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, vulkanQualifiedIndexScaleHeader), string(header))
	for _, stem := range vulkanRegistryV23.stems()[59:] {
		writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stem+".spv"), "inert fixture")
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv"), string(vulkanV5StructuralFixture()))
	module, err := decodeVulkanQualifiedIndexScale(source, header)
	if err != nil {
		t.Fatal(err)
	}
	modulePath := filepath.Join(evidence.SPIRVRoot, vulkanQualifiedIndexScaleStem+".spv")
	writeFixtureFile(t, modulePath, string(module))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/libfakvulkan.a"), "inert archive")
	commit := func() {
		runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
		runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "canonical fixture")
		evidence.ExpectedCommit = runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD")
	}
	commit()
	evidence.ExpectedSPIRVModules = vulkanRegistryV23.stems()
	observed, err := vulkanRegistryV23.observeEvidence(context.Background(), evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := vulkanRegistryV23.identity()
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{Schema: VulkanBuildReceiptSchemaV23, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact, VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"}, Vulkan: &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256}}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV23.verifyIdentityWithRunners(context.Background(), evidence, fixture.gitRun, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := vulkanRegistryV22.decodeStrict(raw); err == nil {
		t.Fatal("V22 accepted V23 materialization receipt")
	}
	cfg := &VulkanConfig{RepoRoot: evidence.SourceRoot, PkgDir: evidence.BuildPlan.PackageDir, OutPkg: "./cmd/fak", Toolchain: evidence.Tools.toolchain()}
	if _, _, _, _, err := vulkanRegistryV23.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, modulePath, "corrupted")
	commit() // A clean source tree does not authorize a different module.
	if _, err := vulkanRegistryV23.observeEvidence(context.Background(), evidence, fixture.gitRun); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("observation accepted corruption: %v", err)
	}
	if _, _, _, _, err := vulkanRegistryV23.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("finalization accepted corruption: %v", err)
	}
}
