package devcmd

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

const (
	vulkanVerifyRegistrySHA = "c032225a0fa215d5cea54ffd8d2c5bdc1b14c66685959ae4913e6521ca5cde2a"
	vulkanVerifyPolicyID    = "fak.v41-indexer-score-f32-structure.v1"
)

type vulkanVerifyFixture struct {
	repo              string
	commit            string
	tree              string
	archiveSHA        string
	spirvRoot         string
	spirvSHA          string
	spirvModules      []string
	matchedReceipt    string
	baselineReceipt   string
	matchedBinary     string
	baselineBinary    string
	binarySHA         string
	binarySize        int64
	gitPath           string
	gitSHA            string
	toolPath          string
	toolSHA           string
	toolchain         []computebuild.BuildToolIdentity
	toolchainSHA      string
	registry          computebuild.VulkanShaderRegistryIdentity
	nativeArchive     computebuild.BuildArtifact
	indexer           computebuild.VulkanIndexerScoreContract
	reproducibility   computebuild.BuildReproducibility
	normalizedCommand []string
	commandSHA        string
	stableIdentity    string
	baselineSHA       string
	marker            string
	argv              []string
}

// fak-test:runtime medium est=8s lane=default
// Ref #13823. This exercises static identity admission only: the candidate is
// an execution sentinel, and no compiler, model, device, service, or network is used.
func TestBuildVulkanVerifyV11ProjectionAndRefusals(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the canonical verifier intentionally requires Linux fd-bound Git execution")
	}
	fixture := newVulkanVerifyFixture(t)

	run := func(t *testing.T, argv []string) (int, map[string]any) {
		t.Helper()
		var stdout, stderr bytes.Buffer
		code := RunBuildVulkan(&stdout, &stderr, argv)
		var result map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
			t.Fatalf("decode JSON result: %v\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
		}
		if _, err := os.Stat(fixture.marker); !os.IsNotExist(err) {
			t.Fatalf("candidate executed or marker observation failed: %v", err)
		}
		return code, result
	}
	assertRefusal := func(t *testing.T, got map[string]any, wantCode string) {
		t.Helper()
		if got["verdict"] != "refused" {
			t.Errorf("verdict = %#v; want refused", got["verdict"])
		}
		codeValue, _ := got["refusal_code"].(string)
		if wantCode != "" {
			if codeValue != wantCode {
				t.Errorf("refusal_code = %q; want %q", codeValue, wantCode)
			}
		} else if codeValue != "VULKAN_VERIFY_INVALID_ARGUMENT" && codeValue != "VULKAN_VERIFY_OBSERVATION_REFUSED" && codeValue != "VULKAN_VERIFY_IDENTITY_REFUSED" {
			t.Errorf("refusal_code = %q", codeValue)
		}
		if got["error"] == nil || got["error"] == "" {
			t.Error("typed refusal omitted its error")
		}
		for _, field := range []string{
			"receipt_schema", "receipt_sha256", "public_revision", "public_tree", "source_archive_sha256",
			"binary_sha256", "binary_size_bytes", "spirv_bundle_sha256", "spirv_module_count", "spirv_modules",
			"shader_registry", "native_archive", "indexer_score_contract", "toolchain", "toolchain_sha256",
			"normalized_build_command", "build_command_sha256", "stable_identity_sha256", "reproducibility",
			"baseline_receipt_sha256", "historical_build_causality", "unavailable_claims",
		} {
			if value, exists := got[field]; exists && value != nil && value != "" {
				t.Errorf("refusal leaked partial verified field %s=%#v", field, value)
			}
		}
	}

	t.Run("verified projection", func(t *testing.T) {
		code, got := run(t, fixture.argv)
		if code != 0 {
			t.Fatalf("verify exit = %d; result = %#v", code, got)
		}
		wantScalars := map[string]any{
			"schema":                     "fak.vulkan-build-verification.v1",
			"command":                    "verify",
			"verdict":                    "verified",
			"receipt_schema":             computebuild.VulkanBuildReceiptSchemaV11,
			"receipt_sha256":             vulkanVerifyTestSHA256File(t, fixture.matchedReceipt),
			"public_revision":            fixture.commit,
			"public_tree":                fixture.tree,
			"source_archive_sha256":      fixture.archiveSHA,
			"binary_sha256":              fixture.binarySHA,
			"binary_size_bytes":          float64(fixture.binarySize),
			"spirv_bundle_sha256":        fixture.spirvSHA,
			"spirv_module_count":         float64(71),
			"toolchain_sha256":           fixture.toolchainSHA,
			"build_command_sha256":       fixture.commandSHA,
			"stable_identity_sha256":     fixture.stableIdentity,
			"baseline_receipt_sha256":    fixture.baselineSHA,
			"historical_build_causality": "unavailable",
		}
		for field, want := range wantScalars {
			if got[field] != want {
				t.Errorf("%s = %#v; want %#v", field, got[field], want)
			}
		}
		if got["refusal_code"] != nil && got["refusal_code"] != "" {
			t.Errorf("verified result has refusal_code = %#v", got["refusal_code"])
		}
		if got["error"] != nil && got["error"] != "" {
			t.Errorf("verified result has error = %#v", got["error"])
		}
		assertVulkanVerifyJSONKeys(t, got, []string{
			"schema", "command", "verdict", "receipt_schema", "receipt_sha256", "public_revision", "public_tree",
			"source_archive_sha256", "binary_sha256", "binary_size_bytes", "spirv_bundle_sha256", "spirv_module_count",
			"spirv_modules", "shader_registry", "native_archive", "indexer_score_contract", "toolchain", "toolchain_sha256",
			"normalized_build_command", "build_command_sha256", "stable_identity_sha256", "reproducibility",
			"baseline_receipt_sha256", "historical_build_causality", "unavailable_claims",
		})
		assertVulkanVerifyJSONList(t, got, "spirv_modules", fixture.spirvModules)
		assertVulkanVerifyJSONList(t, got, "normalized_build_command", fixture.normalizedCommand)
		assertVulkanVerifyJSONList(t, got, "unavailable_claims", []string{
			"historical_source_cleanliness",
			"actual_build_argv",
			"actual_build_environment",
			"actual_build_tool_causality",
			"sdk_header_library_vsdevenv_inputs",
			"smoke_execution",
			"build_timing",
			"indexer_score_source_to_module_correspondence",
			"indexer_score_ordered_arithmetic_semantics",
			"device_f32_qualification",
			"native_archive_to_binary_linkage",
			"production_build_profile_causality",
		})
		assertVulkanVerifyJSONValue(t, got, "shader_registry", fixture.registry)
		assertVulkanVerifyJSONValue(t, got, "native_archive", fixture.nativeArchive)
		assertVulkanVerifyJSONValue(t, got, "indexer_score_contract", fixture.indexer)
		assertVulkanVerifyJSONValue(t, got, "toolchain", fixture.toolchain)
		assertVulkanVerifyJSONValue(t, got, "reproducibility", fixture.reproducibility)
	})

	matchedRaw := readVulkanVerifyTestFile(t, fixture.matchedReceipt)
	binaryRaw := readVulkanVerifyTestFile(t, fixture.matchedBinary)
	baselineRaw := readVulkanVerifyTestFile(t, fixture.baselineReceipt)
	baselineBinaryRaw := readVulkanVerifyTestFile(t, fixture.baselineBinary)
	spirvPath := filepath.Join(fixture.spirvRoot, fixture.spirvModules[0]+".spv")
	spirvRaw := readVulkanVerifyTestFile(t, spirvPath)
	toolRaw := readVulkanVerifyTestFile(t, fixture.toolPath)
	zeroCommit := strings.Repeat("0", 40)

	refusals := []struct {
		name   string
		mutate func(t *testing.T) func()
	}{
		{
			name: "altered binary",
			mutate: func(t *testing.T) func() {
				writeVulkanVerifyTestFile(t, fixture.matchedBinary, append(append([]byte(nil), binaryRaw...), 'x'), 0755)
				return func() { writeVulkanVerifyTestFile(t, fixture.matchedBinary, binaryRaw, 0755) }
			},
		},
		{
			name: "altered baseline binary",
			mutate: func(t *testing.T) func() {
				writeVulkanVerifyTestFile(t, fixture.baselineBinary, append(append([]byte(nil), baselineBinaryRaw...), 'x'), 0755)
				return func() { writeVulkanVerifyTestFile(t, fixture.baselineBinary, baselineBinaryRaw, 0755) }
			},
		},
		{
			name: "altered SPIR-V",
			mutate: func(t *testing.T) func() {
				writeVulkanVerifyTestFile(t, spirvPath, append(append([]byte(nil), spirvRaw...), 'x'), 0644)
				return func() { writeVulkanVerifyTestFile(t, spirvPath, spirvRaw, 0644) }
			},
		},
		{
			name: "altered registry",
			mutate: func(t *testing.T) func() {
				mutateVulkanVerifyReceiptJSON(t, fixture.matchedReceipt, func(receipt map[string]any) {
					receipt["shader_registry"].(map[string]any)["sha256"] = strings.Repeat("0", 64)
				})
				return func() { writeVulkanVerifyTestFile(t, fixture.matchedReceipt, matchedRaw, 0644) }
			},
		},
		{
			name: "altered baseline receipt",
			mutate: func(t *testing.T) func() {
				writeVulkanVerifyTestFile(t, fixture.baselineReceipt, append(append([]byte(nil), baselineRaw...), ' '), 0644)
				return func() { writeVulkanVerifyTestFile(t, fixture.baselineReceipt, baselineRaw, 0644) }
			},
		},
		{
			name: "altered source CAS",
			mutate: func(t *testing.T) func() {
				old := fixture.argv
				fixture.argv = replaceVulkanVerifyFlagValue(t, old, "--commit", zeroCommit)
				return func() { fixture.argv = old }
			},
		},
		{
			name: "altered tool",
			mutate: func(t *testing.T) func() {
				writeVulkanVerifyTestFile(t, fixture.toolPath, append(append([]byte(nil), toolRaw...), 'x'), 0755)
				return func() { writeVulkanVerifyTestFile(t, fixture.toolPath, toolRaw, 0755) }
			},
		},
		{
			name: "altered reproducibility",
			mutate: func(t *testing.T) func() {
				mutateVulkanVerifyReceiptJSON(t, fixture.matchedReceipt, func(receipt map[string]any) {
					receipt["reproducibility"].(map[string]any)["compared_receipt_sha256"] = strings.Repeat("f", 64)
				})
				return func() { writeVulkanVerifyTestFile(t, fixture.matchedReceipt, matchedRaw, 0644) }
			},
		},
	}
	for _, tc := range refusals {
		t.Run(tc.name, func(t *testing.T) {
			restore := tc.mutate(t)
			t.Cleanup(restore)
			code, got := run(t, fixture.argv)
			if code == 0 {
				t.Fatalf("refusal exit/result = %d/%#v", code, got)
			}
			assertRefusal(t, got, "")
		})
	}

	for _, tc := range []struct {
		name string
		edit func(*computebuild.ComputeBuildReceipt)
	}{
		{"historical canonical schema", func(r *computebuild.ComputeBuildReceipt) { r.Schema = computebuild.VulkanBuildReceiptSchemaV10 }},
		{"historical canonical registry", func(r *computebuild.ComputeBuildReceipt) {
			r.VulkanRegistry.ID = computebuild.VulkanShaderRegistryV10ID
		}},
		{"historical canonical count", func(r *computebuild.ComputeBuildReceipt) { r.Vulkan.SPIRVModuleCount = 69 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := verifyVulkanBinaryReceiptIdentityForSchema
			verifyVulkanBinaryReceiptIdentityForSchema = func(ctx context.Context, schema string, evidence computebuild.VulkanBinaryReceiptEvidence) (*computebuild.VulkanBinaryReceiptIdentityVerification, error) {
				if schema != computebuild.VulkanBuildReceiptSchemaV11 || !reflect.DeepEqual(evidence.ExpectedSPIRVModules, computebuild.CurrentVulkanShaderRegistryV11()) {
					t.Fatal("CLI did not request the exact current V11 identity")
				}
				verified, err := original(ctx, schema, evidence)
				if err != nil {
					t.Fatalf("unmodified canonical verification failed: %v", err)
				}
				tc.edit(&verified.Receipt)
				return verified, nil
			}
			t.Cleanup(func() { verifyVulkanBinaryReceiptIdentityForSchema = original })
			code, got := run(t, fixture.argv)
			if code == 0 {
				t.Fatalf("historical identity projected as current success: %#v", got)
			}
			assertRefusal(t, got, "VULKAN_VERIFY_INCOMPLETE_RESULT")
		})
	}

	t.Run("incomplete canonical success refuses", func(t *testing.T) {
		original := verifyVulkanBinaryReceiptIdentityForSchema
		verifyVulkanBinaryReceiptIdentityForSchema = func(context.Context, string, computebuild.VulkanBinaryReceiptEvidence) (*computebuild.VulkanBinaryReceiptIdentityVerification, error) {
			return &computebuild.VulkanBinaryReceiptIdentityVerification{}, nil
		}
		t.Cleanup(func() { verifyVulkanBinaryReceiptIdentityForSchema = original })
		code, got := run(t, fixture.argv)
		if code == 0 {
			t.Fatalf("incomplete verifier success exit/result = %d/%#v", code, got)
		}
		assertRefusal(t, got, "VULKAN_VERIFY_INCOMPLETE_RESULT")
	})

	t.Run("cmd verify requires explicit matched receipt", func(t *testing.T) {
		cwd, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		defaultRoot := t.TempDir()
		writeVulkanVerifyTestFile(t, filepath.Join(defaultRoot, ".fak", "vulkan-build-receipt.json"), matchedRaw, 0644)
		if err := os.Chdir(defaultRoot); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := os.Chdir(cwd); err != nil {
				t.Errorf("restore working directory: %v", err)
			}
		})
		argv := removeVulkanVerifyFlag(fixture.argv[1:], "--matched-receipt")
		argv = append([]string{"--cmd", "verify"}, argv...)
		code, got := run(t, argv)
		if code == 0 {
			t.Fatalf("implicit matched receipt exit/result = %d/%#v", code, got)
		}
		assertRefusal(t, got, "VULKAN_VERIFY_INVALID_ARGUMENT")
	})

	t.Run("verify-only flag rejected by build mode", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunBuildVulkan(&stdout, &stderr, []string{
			"lib", "--matched-receipt", fixture.matchedReceipt, "--repo-root", fixture.repo,
			"--cxx", fixture.matchedBinary,
		})
		if code == 0 || !strings.Contains(stderr.String(), "verify") {
			t.Fatalf("non-verify flag admission = %d; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
		}
		if _, err := os.Stat(fixture.marker); !os.IsNotExist(err) {
			t.Fatalf("non-verify mode invoked compiler candidate or marker observation failed: %v", err)
		}
	})
}

func newVulkanVerifyFixture(t *testing.T) *vulkanVerifyFixture {
	t.Helper()
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	writeVulkanVerifyTestFile(t, filepath.Join(repo, "go.mod"), []byte("module example.test/vulkan-verify\n\ngo 1.26\n"), 0644)
	writeVulkanVerifyTestFile(t, filepath.Join(repo, "cmd", "fak", "main.go"), []byte("package main\nfunc main() {}\n"), 0644)
	archivePath := filepath.Join(repo, "internal", "compute", "libfakvulkan.a")
	archiveRaw := []byte("inert native archive identity\n")
	writeVulkanVerifyTestFile(t, archivePath, archiveRaw, 0644)

	gitPath, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	gitPath, err = filepath.Abs(gitPath)
	if err != nil {
		t.Fatal(err)
	}
	gitPath, err = filepath.EvalSymlinks(gitPath)
	if err != nil {
		t.Fatal(err)
	}
	runVulkanVerifyTestGit(t, gitPath, root, "init", "-q", repo)
	runVulkanVerifyTestGit(t, gitPath, repo, "config", "user.email", "fixture@example.invalid")
	runVulkanVerifyTestGit(t, gitPath, repo, "config", "user.name", "Fixture")
	runVulkanVerifyTestGit(t, gitPath, repo, "add", "-f", ".")
	runVulkanVerifyTestGit(t, gitPath, repo, "commit", "-q", "-m", "fixture")
	commit := strings.TrimSpace(runVulkanVerifyTestGit(t, gitPath, repo, "rev-parse", "HEAD"))
	tree := strings.TrimSpace(runVulkanVerifyTestGit(t, gitPath, repo, "rev-parse", commit+"^{tree}"))
	archive := runVulkanVerifyTestGitBytes(t, gitPath, repo, "archive", "--format=tar", commit)
	archiveSum := sha256.Sum256(archive)

	marker := filepath.Join(root, "candidate-executed")
	candidate := []byte("#!/bin/sh\nprintf executed > '" + strings.ReplaceAll(marker, "'", "'\\''") + "'\nexit 97\n")
	baselineBinary := filepath.Join(root, "baseline-fak")
	matchedBinary := filepath.Join(root, "matched-fak")
	writeVulkanVerifyTestFile(t, baselineBinary, candidate, 0755)
	writeVulkanVerifyTestFile(t, matchedBinary, candidate, 0755)
	binarySum := sha256.Sum256(candidate)

	toolPath := filepath.Join(root, "inert-tool")
	toolRaw := []byte("not an executable; the verifier must hash but never run these bytes\n")
	writeVulkanVerifyTestFile(t, toolPath, toolRaw, 0755)
	toolSum := sha256.Sum256(toolRaw)
	gitSum := sha256.Sum256(readVulkanVerifyTestFile(t, gitPath))

	spirvRoot := filepath.Join(root, "spirv")
	modules := computebuild.CurrentVulkanShaderRegistryV11()
	for _, stem := range modules {
		raw := []byte("spirv:" + stem + "\n")
		if stem == "v41_indexer_score" {
			raw = vulkanVerifyIndexerFixture()
		}
		writeVulkanVerifyTestFile(t, filepath.Join(spirvRoot, stem+".spv"), raw, 0644)
	}
	spirvSHA := hashVulkanVerifySPIRV(t, spirvRoot, modules)
	indexerRaw := readVulkanVerifyTestFile(t, filepath.Join(spirvRoot, "v41_indexer_score.spv"))
	indexerSum := sha256.Sum256(indexerRaw)

	toolSHA := hex.EncodeToString(toolSum[:])
	toolchain := make([]computebuild.BuildToolIdentity, 0, 5)
	for _, role := range []string{"go", "cc", "cxx", "ar", "glslc"} {
		toolchain = append(toolchain, computebuild.BuildToolIdentity{Role: role, Executable: filepath.Base(toolPath), SHA256: toolSHA})
	}
	toolchainSHA := hashVulkanVerifyTestJSON(t, struct {
		Tools      []computebuild.BuildToolIdentity `json:"tools"`
		CXXRuntime string                           `json:"cxx_runtime"`
		IsWindows  bool                             `json:"is_windows"`
	}{toolchain, "-lstdc++", false})
	normalized := []string{
		"glslc|-O|--target-env=vulkan1.2|-fshader-stage=comp|$REPO/internal/compute/shaders/{registered}.comp|-o|$REPO/internal/compute/spirv/{registered}.spv",
		"cxx|-O3,-std=c++17,-fPIC|-c|$REPO/internal/compute/vulkan_shim.cpp|-o|$REPO/internal/compute/vulkan_shim.o",
		"ar|rcs|$REPO/internal/compute/libfakvulkan.a|$REPO/internal/compute/vulkan_shim.o",
		"go|build|-tags|vulkan|-o|$OUT|cmd/fak",
		"pkg_dir=$REPO/internal/compute",
		"smoke=false",
	}
	commandSHA := hashVulkanVerifyTestJSON(t, normalized)

	source := computebuild.BuildSourceProvenance{
		GitCommit: commit, GitTree: tree, Clean: true, SourceArchiveSHA256: hex.EncodeToString(archiveSum[:]),
	}
	registry := computebuild.VulkanShaderRegistryIdentity{
		ID: computebuild.VulkanShaderRegistryV11ID, SHA256: vulkanVerifyRegistrySHA, ModuleCount: 71,
	}
	nativeArchive := computebuild.BuildArtifact{Path: "internal/compute/libfakvulkan.a", SizeBytes: int64(len(archiveRaw)), SHA256: vulkanVerifyTestSHA256Hex(archiveRaw)}
	indexer := computebuild.VulkanIndexerScoreContract{
		PolicyID: vulkanVerifyPolicyID, ModuleSHA256: hex.EncodeToString(indexerSum[:]),
		EntryPoint: "main", LocalSize: [3]uint32{1, 1, 1}, FloatWidth: 32,
		DenormPreserve: true, SignedZeroInfNanPreserve: true, RoundingModeRTE: true,
		NoContractionFAddCount: 2, NoContractionFMulCount: 2,
	}
	binarySHA := hex.EncodeToString(binarySum[:])
	stableSHA := hashVulkanVerifyTestJSON(t, struct {
		Schema             string                                    `json:"schema"`
		Registry           computebuild.VulkanShaderRegistryIdentity `json:"shader_registry"`
		Source             computebuild.BuildSourceProvenance        `json:"source"`
		SPIRVBundleSHA256  string                                    `json:"spirv_bundle_sha256"`
		SPIRVModuleCount   int                                       `json:"spirv_module_count"`
		ToolchainSHA256    string                                    `json:"toolchain_sha256"`
		BuildCommandSHA256 string                                    `json:"build_command_sha256"`
		BinarySHA256       string                                    `json:"binary_sha256"`
		NativeArchive      computebuild.BuildArtifact                `json:"native_archive"`
		IndexerScore       computebuild.VulkanIndexerScoreContract   `json:"indexer_score_contract"`
	}{computebuild.VulkanBuildReceiptSchemaV11, registry, source, spirvSHA, 71, toolchainSHA, commandSHA, binarySHA, nativeArchive, indexer})

	baselinePath := filepath.Join(root, "baseline-receipt.json")
	matchedPath := filepath.Join(root, "matched-receipt.json")
	makeReceipt := func(path, binary string, reproducibility *computebuild.BuildReproducibility) computebuild.ComputeBuildReceipt {
		return computebuild.ComputeBuildReceipt{
			Schema: computebuild.VulkanBuildReceiptSchemaV11, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: path,
			Phases: []computebuild.ComputeBuildPhase{}, Artifact: &computebuild.BuildArtifact{Path: binary, SizeBytes: int64(len(candidate)), SHA256: binarySHA},
			Vulkan: &computebuild.VulkanBuildProvenance{
				Source: source, SPIRVBundleSHA256: spirvSHA, SPIRVModuleCount: 71,
				Toolchain: toolchain, ToolchainSHA256: toolchainSHA, NormalizedBuildCommand: normalized,
				BuildCommandSHA256: commandSHA, StableIdentitySHA256: stableSHA,
			},
			VulkanRegistry: &registry, VulkanNativeArchive: &nativeArchive, VulkanIndexerScore: &indexer,
			Reproducibility: reproducibility,
		}
	}
	baseline := makeReceipt(baselinePath, baselineBinary, &computebuild.BuildReproducibility{Status: "baseline"})
	writeVulkanVerifyTestReceipt(t, baselinePath, baseline)
	baselineRaw := readVulkanVerifyTestFile(t, baselinePath)
	baselineSum := sha256.Sum256(baselineRaw)
	baselineSHA := hex.EncodeToString(baselineSum[:])
	reproducibility := computebuild.BuildReproducibility{Status: "match", ComparedReceiptSHA256: baselineSHA}
	matched := makeReceipt(matchedPath, matchedBinary, &reproducibility)
	writeVulkanVerifyTestReceipt(t, matchedPath, matched)

	argv := []string{
		"verify", "--repo-root", repo, "--commit", commit,
		"--matched-receipt", matchedPath, "--matched-binary", matchedBinary,
		"--baseline-receipt", baselinePath, "--baseline-binary", baselineBinary,
		"--spirv-dir", spirvRoot, "--git", gitPath, "--git-sha256", hex.EncodeToString(gitSum[:]),
		"--go", toolPath, "--cc", toolPath, "--cxx", toolPath, "--cxx-runtime", "-lstdc++",
		"--ar", toolPath, "--glslc", toolPath, "--json",
	}
	return &vulkanVerifyFixture{
		repo: repo, commit: commit, tree: tree, archiveSHA: source.SourceArchiveSHA256,
		spirvRoot: spirvRoot, spirvSHA: spirvSHA, spirvModules: modules,
		matchedReceipt: matchedPath, baselineReceipt: baselinePath,
		matchedBinary: matchedBinary, baselineBinary: baselineBinary,
		binarySHA: binarySHA, binarySize: int64(len(candidate)),
		gitPath: gitPath, gitSHA: hex.EncodeToString(gitSum[:]), toolPath: toolPath, toolSHA: toolSHA,
		toolchain: toolchain, toolchainSHA: toolchainSHA, normalizedCommand: normalized,
		registry: registry, nativeArchive: nativeArchive, indexer: indexer, reproducibility: reproducibility,
		commandSHA: commandSHA, stableIdentity: stableSHA, baselineSHA: baselineSHA,
		marker: marker, argv: argv,
	}
}

func vulkanVerifyIndexerFixture() []byte {
	words := []uint32{
		0x07230203, 0x00010500, 0, 64, 0,
		2<<16 | 17, 1,
		2<<16 | 17, 4464,
		2<<16 | 17, 4466,
		2<<16 | 17, 4467,
		7<<16 | 10, 0x5f565053, 0x5f52484b, 0x616f6c66, 0x6f635f74, 0x6f72746e, 0x0000736c,
		5<<16 | 15, 5, 1, 0x6e69616d, 0,
		6<<16 | 16, 1, 17, 1, 1, 1,
		4<<16 | 16, 1, 4459, 32,
		4<<16 | 16, 1, 4461, 32,
		4<<16 | 16, 1, 4462, 32,
		3<<16 | 22, 2, 32,
		3<<16 | 71, 10, 42,
		3<<16 | 71, 11, 42,
		3<<16 | 71, 12, 42,
		3<<16 | 71, 13, 42,
		5<<16 | 133, 2, 10, 3, 4,
		5<<16 | 129, 2, 11, 5, 10,
		5<<16 | 133, 2, 12, 11, 6,
		5<<16 | 129, 2, 13, 7, 12,
	}
	raw := make([]byte, 4*len(words))
	for i, word := range words {
		binary.LittleEndian.PutUint32(raw[4*i:], word)
	}
	return raw
}

func hashVulkanVerifySPIRV(t *testing.T, root string, modules []string) string {
	t.Helper()
	names := make([]string, len(modules))
	for i, stem := range modules {
		names[i] = stem + ".spv"
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		raw := readVulkanVerifyTestFile(t, filepath.Join(root, name))
		fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(filepath.Join("internal", "compute", "spirv", name)), len(raw))
		_, _ = h.Write(raw)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func hashVulkanVerifyTestJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return vulkanVerifyTestSHA256Hex(raw)
}

func vulkanVerifyTestSHA256Hex(raw []byte) string {
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:])
}

func vulkanVerifyTestSHA256File(t *testing.T, path string) string {
	t.Helper()
	return vulkanVerifyTestSHA256Hex(readVulkanVerifyTestFile(t, path))
}

func writeVulkanVerifyTestReceipt(t *testing.T, path string, receipt computebuild.ComputeBuildReceipt) {
	t.Helper()
	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	writeVulkanVerifyTestFile(t, path, append(raw, '\n'), 0644)
}

func mutateVulkanVerifyReceiptJSON(t *testing.T, path string, mutate func(map[string]any)) {
	t.Helper()
	var receipt map[string]any
	if err := json.Unmarshal(readVulkanVerifyTestFile(t, path), &receipt); err != nil {
		t.Fatal(err)
	}
	mutate(receipt)
	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	writeVulkanVerifyTestFile(t, path, append(raw, '\n'), 0644)
}

func writeVulkanVerifyTestFile(t *testing.T, path string, raw []byte, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, mode); err != nil {
		t.Fatal(err)
	}
}

func readVulkanVerifyTestFile(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func runVulkanVerifyTestGit(t *testing.T, gitPath, root string, args ...string) string {
	t.Helper()
	return string(runVulkanVerifyTestGitBytes(t, gitPath, root, args...))
}

func runVulkanVerifyTestGitBytes(t *testing.T, gitPath, root string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command(gitPath, append([]string{"-C", root}, args...)...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, output)
	}
	return output
}

func replaceVulkanVerifyFlagValue(t *testing.T, argv []string, flag, value string) []string {
	t.Helper()
	copyArgv := append([]string(nil), argv...)
	for i := 0; i+1 < len(copyArgv); i++ {
		if copyArgv[i] == flag {
			copyArgv[i+1] = value
			return copyArgv
		}
	}
	t.Fatalf("flag %s missing from argv", flag)
	return nil
}

func removeVulkanVerifyFlag(argv []string, flag string) []string {
	result := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		if argv[i] == flag && i+1 < len(argv) {
			i++
			continue
		}
		result = append(result, argv[i])
	}
	return result
}

func assertVulkanVerifyJSONList(t *testing.T, got map[string]any, field string, want []string) {
	t.Helper()
	values, ok := got[field].([]any)
	if !ok || len(values) != len(want) {
		t.Fatalf("%s = %#v; want %v", field, got[field], want)
	}
	for i := range want {
		if values[i] != want[i] {
			t.Errorf("%s[%d] = %#v; want %q", field, i, values[i], want[i])
		}
	}
}

func assertVulkanVerifyJSONKeys(t *testing.T, got map[string]any, want []string) {
	t.Helper()
	keys := make([]string, 0, len(got))
	for key := range got {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	want = append([]string(nil), want...)
	sort.Strings(want)
	if !reflect.DeepEqual(keys, want) {
		t.Errorf("JSON keys = %v; want %v", keys, want)
	}
}

func assertVulkanVerifyJSONValue(t *testing.T, got map[string]any, field string, want any) {
	t.Helper()
	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var normalized any
	if err := json.Unmarshal(raw, &normalized); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got[field], normalized) {
		t.Errorf("%s = %#v; want %#v", field, got[field], normalized)
	}
}
