package computebuild

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func installQualifiedQ2TokenPairFixture(t *testing.T, root string) {
	t.Helper()
	for _, rel := range []string{"shaders/v41_q2k_token_pair.comp", "v41_q2k_token_pair_spirv.h"} {
		data, err := os.ReadFile(filepath.Join("..", "compute", rel))
		if err != nil {
			t.Fatal(err)
		}
		dst := filepath.Join(root, "internal", "compute", rel)
		if err = os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(dst, data, 0644); err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime fast est=20ms
func TestVulkanQualifiedQ2TokenPair(t *testing.T) {
	root := t.TempDir()
	installQualifiedQ2TokenPairFixture(t, root)
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal/compute/shaders", vulkanQualifiedQ2TokenPairStem+".comp"))
	if err != nil {
		t.Fatal(err)
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, vulkanQualifiedQ2TokenPairHeader))
	if err != nil {
		t.Fatal(err)
	}
	module, err := decodeVulkanQualifiedQ2TokenPair(source, header)
	if err != nil {
		t.Fatal(err)
	}
	if len(module) != 11796 || !vulkanQualifiedHash(module, "106f42d56d87f0d2ce44704a1e944efefc7a67320cf1a44cb492c4bacf07acbf") {
		t.Fatal("qualified module differs")
	}
	for _, pair := range [][2][]byte{{bytes.ReplaceAll(source, []byte("\n"), []byte("\r\n")), header}, {source, bytes.ReplaceAll(header, []byte("\n"), []byte("\r\n"))}, {append(append([]byte(nil), source...), 0), header}, {source, header[:len(header)-1]}, {source, bytes.Replace(header, []byte("0x03"), []byte("0x02"), 1)}} {
		if _, err := decodeVulkanQualifiedQ2TokenPair(pair[0], pair[1]); err == nil {
			t.Fatal("drift accepted")
		}
	}
	out := filepath.Join(root, "internal/compute/spirv")
	if err = os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(out, vulkanQualifiedQ2TokenPairStem+".spv")
	if err = writeVulkanQualifiedQ2TokenPair(dst, module); err != nil {
		t.Fatal(err)
	}
	if err = verifyVulkanQualifiedQ2TokenPair(root, out); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{module[:len(module)-1], append(append([]byte(nil), module...), 0), bytes.Repeat([]byte{0}, len(module))} {
		if err = writeVulkanQualifiedQ2TokenPair(dst, bad); err == nil {
			t.Fatal("bad output accepted")
		}
		got, e := os.ReadFile(dst)
		if e != nil || !bytes.Equal(got, module) {
			t.Fatal("rejection replaced installed module")
		}
	}
	if err = os.WriteFile(dst, bytes.Repeat([]byte{0}, len(module)), 0644); err != nil {
		t.Fatal(err)
	}
	if verifyVulkanQualifiedQ2TokenPair(root, out) == nil {
		t.Fatal("corrupt bundle accepted")
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanReceiptV26Q2TokenPair(t *testing.T) {
	want := append(vulkanRegistryV25.stems(), "v41_q2k_token_pair")
	if len(want) != 89 || !reflect.DeepEqual(want, vulkanRegistryV26.stems()) {
		t.Fatal("V26 membership differs")
	}
	schema := vulkanRegistryV26.schema
	id, err := vulkanRegistryV26.identity()
	if err != nil || schema != VulkanBuildReceiptSchemaV26 || id.ID != VulkanShaderRegistryV26ID || id.ModuleCount != 89 {
		t.Fatal("V26 identity differs")
	}
	if vulkanRegistryV25.validateStems(want) == nil || vulkanRegistryV26.validateStems(vulkanRegistryV25.stems()) == nil {
		t.Fatal("historical membership interchangeable")
	}
	for _, win := range []bool{false, true} {
		root := t.TempDir()
		cfg := &VulkanConfig{RepoRoot: root, PkgDir: filepath.Join(root, "internal/compute"), OutPkg: "./cmd/fak", Toolchain: &Toolchain{IsWindows: win}}
		old, oldHash, e := vulkanRegistryV25.normalizedBuildCommand(cfg)
		if e != nil {
			t.Fatal(e)
		}
		now, hash, e := vulkanRegistryV26.normalizedBuildCommand(cfg)
		if e != nil {
			t.Fatal(e)
		}
		if hash == oldHash || len(now) != len(old)+1 || !reflect.DeepEqual(now[1:4], old[1:4]) || !reflect.DeepEqual(now[5:], old[4:]) || !strings.Contains(now[0], "v41_q2k_token_pair") || !strings.Contains(now[4], vulkanQualifiedQ2TokenPairModuleSHA) || !strings.Contains(now[4], vulkanQualifiedQ2TokenPairHeaderSHA) {
			t.Fatal("V26 provenance differs")
		}
	}
}

// fak-test:runtime fast est=100ms
func TestVulkanQ2TokenPairCanonicalCheckoutAttributes(t *testing.T) {
	root := t.TempDir()
	installQualifiedQ2TokenPairFixture(t, root)
	attrs, err := os.ReadFile(filepath.Join("..", "..", ".gitattributes"))
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(root, ".gitattributes"), attrs, 0644); err != nil {
		t.Fatal(err)
	}
	runFixtureGit(t, root, "init", "-q")
	runFixtureGit(t, root, "config", "core.autocrlf", "true")
	runFixtureGit(t, root, "config", "user.name", "Fixture")
	runFixtureGit(t, root, "config", "user.email", "fixture@example.invalid")
	runFixtureGit(t, root, "add", ".")
	runFixtureGit(t, root, "commit", "-q", "-m", "qualified fixture")
	for _, p := range []string{"internal/compute/shaders/v41_q2k_token_pair.comp", vulkanQualifiedQ2TokenPairHeader} {
		if !strings.HasSuffix(strings.TrimSpace(runFixtureGit(t, root, "check-attr", "eol", "--", p)), ": lf") {
			t.Fatal("canonical LF attribute missing")
		}
		if err = os.Remove(filepath.Join(root, p)); err != nil {
			t.Fatal(err)
		}
		runFixtureGit(t, root, "checkout", "HEAD", "--", p)
	}
	if _, err := readVulkanQualifiedQ2TokenPair(root); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime medium est=6s lane=default
func TestVulkanReceiptV26CanonicalEvidence(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	evidence := fixture.evidence
	installQualifiedQ2TokenPairFixture(t, evidence.SourceRoot)
	installQualifiedChunkFixture(t, evidence.SourceRoot)
	installQualifiedDownChunkFixture(t, evidence.SourceRoot)
	_, source, header := qualifiedIndexScaleFixture(t)
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/shaders/v41_index_scale_bf16.comp"), string(source))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, vulkanQualifiedIndexScaleHeader), string(header))
	for _, stem := range vulkanRegistryV26.stems()[59:] {
		writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stem+".spv"), "inert fixture")
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv"), string(vulkanV5StructuralFixture()))
	module, err := decodeVulkanQualifiedIndexScale(source, header)
	if err != nil {
		t.Fatal(err)
	}
	modulePath := filepath.Join(evidence.SPIRVRoot, vulkanQualifiedIndexScaleStem+".spv")
	writeFixtureFile(t, modulePath, string(module))
	activation, err := readVulkanQualifiedActivationChunk(evidence.SourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, vulkanQualifiedActivationChunkStem+".spv"), string(activation))
	down, err := readVulkanQualifiedDownChunk(evidence.SourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, vulkanQualifiedDownChunkStem+".spv"), string(down))
	chunk, err := readVulkanQualifiedQ2TokenPair(evidence.SourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	modulePath = filepath.Join(evidence.SPIRVRoot, vulkanQualifiedQ2TokenPairStem+".spv")
	writeFixtureFile(t, modulePath, string(chunk))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/libfakvulkan.a"), "inert archive")
	commit := func() {
		runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
		runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "canonical fixture")
		evidence.ExpectedCommit = runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD")
	}
	commit()
	evidence.ExpectedSPIRVModules = vulkanRegistryV26.stems()
	observed, err := vulkanRegistryV26.observeEvidence(context.Background(), evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := vulkanRegistryV26.identity()
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{Schema: VulkanBuildReceiptSchemaV26, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact, VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"}, Vulkan: &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256}}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV26.verifyIdentityWithRunners(context.Background(), evidence, fixture.gitRun, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := vulkanRegistryV25.decodeStrict(raw); err == nil {
		t.Fatal("V25 accepted V26 materialization receipt")
	}
	cfg := &VulkanConfig{RepoRoot: evidence.SourceRoot, PkgDir: evidence.BuildPlan.PackageDir, OutPkg: "./cmd/fak", Toolchain: evidence.Tools.toolchain()}
	if _, _, _, _, err := vulkanRegistryV26.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, modulePath, "corrupted")
	commit() // A clean source tree does not authorize a different module.
	if _, err := vulkanRegistryV26.observeEvidence(context.Background(), evidence, fixture.gitRun); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("observation accepted corruption: %v", err)
	}
	if _, _, _, _, err := vulkanRegistryV26.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("finalization accepted corruption: %v", err)
	}
}
