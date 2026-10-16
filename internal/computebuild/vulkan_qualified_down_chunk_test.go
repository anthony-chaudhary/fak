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

func installQualifiedDownChunkFixture(t *testing.T, root string) {
	t.Helper()
	for _, rel := range []string{"shaders/v41_routed_down_panel_chunk.comp", "v41_routed_down_chunk_spirv.h"} {
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
func TestVulkanQualifiedDownChunk(t *testing.T) {
	root := t.TempDir()
	installQualifiedDownChunkFixture(t, root)
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal/compute/shaders", vulkanQualifiedDownChunkStem+".comp"))
	if err != nil {
		t.Fatal(err)
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, vulkanQualifiedDownChunkHeader))
	if err != nil {
		t.Fatal(err)
	}
	module, err := decodeVulkanQualifiedDownChunk(source, header)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(module, qualifiedHeaderBytesForRelation(t, header)) || !vulkanQualifiedHash(module, "a73eb3056a058e058556825254ce9de65e92eba7f3aec946718e272d978ec674") {
		t.Fatal("qualified module differs")
	}
	for _, pair := range [][2][]byte{{bytes.ReplaceAll(source, []byte("\n"), []byte("\r\n")), header}, {source, bytes.ReplaceAll(header, []byte("\n"), []byte("\r\n"))}, {append(append([]byte(nil), source...), 0), header}, {source, header[:len(header)-1]}, {source, bytes.Replace(header, []byte("0x03"), []byte("0x02"), 1)}} {
		if _, err := decodeVulkanQualifiedDownChunk(pair[0], pair[1]); err == nil {
			t.Fatal("drift accepted")
		}
	}
	out := filepath.Join(root, "internal/compute/spirv")
	if err = os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(out, vulkanQualifiedDownChunkStem+".spv")
	if err = writeVulkanQualifiedDownChunk(dst, module); err != nil {
		t.Fatal(err)
	}
	if err = verifyVulkanQualifiedDownChunk(root, out); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{module[:len(module)-1], append(append([]byte(nil), module...), 0), bytes.Repeat([]byte{0}, len(module))} {
		if err = writeVulkanQualifiedDownChunk(dst, bad); err == nil {
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
	if verifyVulkanQualifiedDownChunk(root, out) == nil {
		t.Fatal("corrupt bundle accepted")
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanReceiptV25DownChunk(t *testing.T) {
	previous := vulkanRegistryV24.stems()
	for _, existing := range previous {
		if existing == "v41_routed_down_panel_chunk" {
			t.Fatal("new qualified module already in predecessor registry")
		}
	}
	want := append(previous, "v41_routed_down_panel_chunk")
	if !reflect.DeepEqual(want, vulkanRegistryV25.stems()) {
		t.Fatal("V25 membership differs")
	}
	id, err := vulkanRegistryV25.identity()
	schema := VulkanBuildReceiptSchemaV25
	if err != nil || schema != VulkanBuildReceiptSchemaV25 || id.ID != VulkanShaderRegistryV25ID || id.ModuleCount != len(want) {
		t.Fatal("V25 identity differs")
	}
	if vulkanRegistryV24.validateStems(want) == nil || vulkanRegistryV25.validateStems(vulkanRegistryV24.stems()) == nil {
		t.Fatal("historical membership interchangeable")
	}
	for _, win := range []bool{false, true} {
		root := t.TempDir()
		cfg := &VulkanConfig{RepoRoot: root, PkgDir: filepath.Join(root, "internal/compute"), OutPkg: "./cmd/fak", Toolchain: &Toolchain{IsWindows: win}}
		old, oldHash, e := vulkanRegistryV24.normalizedBuildCommand(cfg)
		if e != nil {
			t.Fatal(e)
		}
		now, hash, e := vulkanRegistryV25.normalizedBuildCommand(cfg)
		if e != nil {
			t.Fatal(e)
		}
		if hash == oldHash || len(now) != len(old)+1 || !reflect.DeepEqual(now[1:3], old[1:3]) || !reflect.DeepEqual(now[4:], old[3:]) || !strings.Contains(now[0], "v41_routed_down_panel_chunk") || !strings.Contains(now[3], vulkanQualifiedDownChunkModuleSHA) || !strings.Contains(now[3], vulkanQualifiedDownChunkHeaderSHA) {
			t.Fatal("V25 provenance differs")
		}
	}
}

// fak-test:runtime fast est=100ms
func TestVulkanDownChunkCanonicalCheckoutAttributes(t *testing.T) {
	root := t.TempDir()
	installQualifiedDownChunkFixture(t, root)
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
	for _, p := range []string{"internal/compute/shaders/v41_routed_down_panel_chunk.comp", vulkanQualifiedDownChunkHeader} {
		if !strings.HasSuffix(strings.TrimSpace(runFixtureGit(t, root, "check-attr", "eol", "--", p)), ": lf") {
			t.Fatal("canonical LF attribute missing")
		}
		if err = os.Remove(filepath.Join(root, p)); err != nil {
			t.Fatal(err)
		}
		runFixtureGit(t, root, "checkout", "HEAD", "--", p)
	}
	if _, err := readVulkanQualifiedDownChunk(root); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime medium est=6s lane=default
func TestVulkanReceiptV25CanonicalEvidence(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	evidence := fixture.evidence
	installQualifiedDownChunkFixture(t, evidence.SourceRoot)
	installQualifiedChunkFixture(t, evidence.SourceRoot)
	_, source, header := qualifiedIndexScaleFixture(t)
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/shaders/v41_index_scale_bf16.comp"), string(source))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, vulkanQualifiedIndexScaleHeader), string(header))
	for _, stem := range vulkanRegistryV25.stems()[59:] {
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
	chunk, err := readVulkanQualifiedDownChunk(evidence.SourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	modulePath = filepath.Join(evidence.SPIRVRoot, vulkanQualifiedDownChunkStem+".spv")
	writeFixtureFile(t, modulePath, string(chunk))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/libfakvulkan.a"), "inert archive")
	commit := func() {
		runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
		runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "canonical fixture")
		evidence.ExpectedCommit = runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD")
	}
	commit()
	evidence.ExpectedSPIRVModules = vulkanRegistryV25.stems()
	observed, err := vulkanRegistryV25.observeEvidence(context.Background(), evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := vulkanRegistryV25.identity()
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{Schema: VulkanBuildReceiptSchemaV25, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact, VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"}, Vulkan: &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256}}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV25.verifyIdentityWithRunners(context.Background(), evidence, fixture.gitRun, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := vulkanRegistryV24.decodeStrict(raw); err == nil {
		t.Fatal("V24 accepted V25 materialization receipt")
	}
	cfg := &VulkanConfig{RepoRoot: evidence.SourceRoot, PkgDir: evidence.BuildPlan.PackageDir, OutPkg: "./cmd/fak", Toolchain: evidence.Tools.toolchain()}
	if _, _, _, _, err := vulkanRegistryV25.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, modulePath, "corrupted")
	commit() // A clean source tree does not authorize a different module.
	if _, err := vulkanRegistryV25.observeEvidence(context.Background(), evidence, fixture.gitRun); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("observation accepted corruption: %v", err)
	}
	if _, _, _, _, err := vulkanRegistryV25.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("finalization accepted corruption: %v", err)
	}
}
