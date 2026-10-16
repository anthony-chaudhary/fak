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

func installQualifiedChunkFixture(t *testing.T, root string) {
	t.Helper()
	for _, rel := range []string{"shaders/v41_routed_swiglu_panel_chunk.comp", "v41_routed_swiglu_chunk_spirv.h"} {
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
func TestVulkanQualifiedActivationChunk(t *testing.T) {
	root := t.TempDir()
	installQualifiedChunkFixture(t, root)
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal/compute/shaders", vulkanQualifiedActivationChunkStem+".comp"))
	if err != nil {
		t.Fatal(err)
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, vulkanQualifiedActivationChunkHeader))
	if err != nil {
		t.Fatal(err)
	}
	module, err := decodeVulkanQualifiedActivationChunk(source, header)
	if err != nil {
		t.Fatal(err)
	}
	if len(module) != 6052 || !vulkanQualifiedHash(module, "6bb4c0d03199686cafb9efa5faa37e4ae14b567f6f2308f687ec5e3b926ed72c") {
		t.Fatal("qualified module differs")
	}
	for _, pair := range [][2][]byte{{bytes.ReplaceAll(source, []byte("\n"), []byte("\r\n")), header}, {source, bytes.ReplaceAll(header, []byte("\n"), []byte("\r\n"))}, {append(append([]byte(nil), source...), 0), header}, {source, header[:len(header)-1]}, {source, bytes.Replace(header, []byte("0x03"), []byte("0x02"), 1)}} {
		if _, err := decodeVulkanQualifiedActivationChunk(pair[0], pair[1]); err == nil {
			t.Fatal("drift accepted")
		}
	}
	out := filepath.Join(root, "internal/compute/spirv")
	if err = os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(out, vulkanQualifiedActivationChunkStem+".spv")
	if err = writeVulkanQualifiedActivationChunk(dst, module); err != nil {
		t.Fatal(err)
	}
	if err = verifyVulkanQualifiedActivationChunk(root, out); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{module[:len(module)-1], append(append([]byte(nil), module...), 0), bytes.Repeat([]byte{0}, len(module))} {
		if err = writeVulkanQualifiedActivationChunk(dst, bad); err == nil {
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
	if verifyVulkanQualifiedActivationChunk(root, out) == nil {
		t.Fatal("corrupt bundle accepted")
	}
}

// fak-test:runtime fast est=5ms
func TestVulkanReceiptV24ActivationChunk(t *testing.T) {
	want := append(vulkanRegistryV23.stems(), "v41_routed_swiglu_panel_chunk")
	if len(want) != 87 || !reflect.DeepEqual(want, VulkanShaders) || !reflect.DeepEqual(want, CurrentVulkanShaderRegistryLatest()) {
		t.Fatal("V24 membership differs")
	}
	schema, id, err := CurrentVulkanBuildIdentity()
	if err != nil || schema != VulkanBuildReceiptSchemaV24 || id.ID != VulkanShaderRegistryV24ID || id.ModuleCount != 87 {
		t.Fatal("V24 identity differs")
	}
	if vulkanRegistryV23.validateStems(want) == nil || vulkanRegistryV24.validateStems(vulkanRegistryV23.stems()) == nil {
		t.Fatal("historical membership interchangeable")
	}
	for _, win := range []bool{false, true} {
		root := t.TempDir()
		cfg := &VulkanConfig{RepoRoot: root, PkgDir: filepath.Join(root, "internal/compute"), OutPkg: "./cmd/fak", Toolchain: &Toolchain{IsWindows: win}}
		old, oldHash, e := vulkanRegistryV23.normalizedBuildCommand(cfg)
		if e != nil {
			t.Fatal(e)
		}
		now, hash, e := vulkanRegistryV24.normalizedBuildCommand(cfg)
		if e != nil {
			t.Fatal(e)
		}
		if hash == oldHash || len(now) != len(old)+1 || now[1] != old[1] || !reflect.DeepEqual(now[3:], old[2:]) || !strings.Contains(now[0], "v41_routed_swiglu_panel_chunk") || !strings.Contains(now[2], vulkanQualifiedActivationChunkModuleSHA) || !strings.Contains(now[2], vulkanQualifiedActivationChunkHeaderSHA) {
			t.Fatal("V24 provenance differs")
		}
	}
}

// fak-test:runtime fast est=100ms
func TestVulkanChunkCanonicalCheckoutAttributes(t *testing.T) {
	root := t.TempDir()
	installQualifiedChunkFixture(t, root)
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
	for _, p := range []string{"internal/compute/shaders/v41_routed_swiglu_panel_chunk.comp", vulkanQualifiedActivationChunkHeader} {
		if !strings.HasSuffix(strings.TrimSpace(runFixtureGit(t, root, "check-attr", "eol", "--", p)), ": lf") {
			t.Fatal("canonical LF attribute missing")
		}
		if err = os.Remove(filepath.Join(root, p)); err != nil {
			t.Fatal(err)
		}
		runFixtureGit(t, root, "checkout", "HEAD", "--", p)
	}
	if _, err := readVulkanQualifiedActivationChunk(root); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime medium est=6s lane=default
func TestVulkanReceiptV24CanonicalEvidence(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	evidence := fixture.evidence
	installQualifiedChunkFixture(t, evidence.SourceRoot)
	_, source, header := qualifiedIndexScaleFixture(t)
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/shaders/v41_index_scale_bf16.comp"), string(source))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, vulkanQualifiedIndexScaleHeader), string(header))
	for _, stem := range vulkanRegistryV24.stems()[59:] {
		writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stem+".spv"), "inert fixture")
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv"), string(vulkanV5StructuralFixture()))
	module, err := decodeVulkanQualifiedIndexScale(source, header)
	if err != nil {
		t.Fatal(err)
	}
	modulePath := filepath.Join(evidence.SPIRVRoot, vulkanQualifiedIndexScaleStem+".spv")
	writeFixtureFile(t, modulePath, string(module))
	chunk, err := readVulkanQualifiedActivationChunk(evidence.SourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	modulePath = filepath.Join(evidence.SPIRVRoot, vulkanQualifiedActivationChunkStem+".spv")
	writeFixtureFile(t, modulePath, string(chunk))
	writeFixtureFile(t, filepath.Join(evidence.SourceRoot, "internal/compute/libfakvulkan.a"), "inert archive")
	commit := func() {
		runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
		runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "canonical fixture")
		evidence.ExpectedCommit = runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD")
	}
	commit()
	evidence.ExpectedSPIRVModules = vulkanRegistryV24.stems()
	observed, err := vulkanRegistryV24.observeEvidence(context.Background(), evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := vulkanRegistryV24.identity()
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{Schema: VulkanBuildReceiptSchemaV24, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact, VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"}, Vulkan: &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256}}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	if _, err := vulkanRegistryV24.verifyIdentityWithRunners(context.Background(), evidence, fixture.gitRun, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := vulkanRegistryV23.decodeStrict(raw); err == nil {
		t.Fatal("V23 accepted V24 materialization receipt")
	}
	cfg := &VulkanConfig{RepoRoot: evidence.SourceRoot, PkgDir: evidence.BuildPlan.PackageDir, OutPkg: "./cmd/fak", Toolchain: evidence.Tools.toolchain()}
	if _, _, _, _, err := vulkanRegistryV24.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err != nil {
		t.Fatal(err)
	}
	writeFixtureFile(t, modulePath, "corrupted")
	commit() // A clean source tree does not authorize a different module.
	if _, err := vulkanRegistryV24.observeEvidence(context.Background(), evidence, fixture.gitRun); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("observation accepted corruption: %v", err)
	}
	if _, _, _, _, err := vulkanRegistryV24.finalizeBinary(cfg, observed.Source, evidence.BinaryPath, observed.Toolchain, observed.ToolchainSHA256); err == nil || !strings.Contains(err.Error(), "differs from qualified artifact") {
		t.Fatalf("finalization accepted corruption: %v", err)
	}
}
