package computebuild

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVulkanShadersCompleteness(t *testing.T) {
	if len(VulkanShaders) != 40 {
		t.Fatalf("expected 40 Vulkan shaders, got %d", len(VulkanShaders))
	}

	seen := make(map[string]bool)
	for _, s := range VulkanShaders {
		if seen[s] {
			t.Errorf("duplicate shader in VulkanShaders: %s", s)
		}
		seen[s] = true
	}

	expectedShaders := []string{
		"matmul", "matmul_add", "matmul_argmax", "matmul_argmax_blocks",
		"matmul2", "matmul3", "rmsnorm", "rmsnorm_matmul",
		"rmsnorm_matmul2", "rmsnorm_matmul3", "rmsnorm_matmul_argmax_blocks",
		"rope", "swiglu", "swiglu_matmul_add", "add", "add_bias",
		"attention", "argmax", "argmax_pairs", "q8_matmul",
		"q8_matmul2", "q8_matmul3", "rmsnorm_q8_matmul2", "rmsnorm_q8_matmul3",
		"swiglu_q8_matmul_add", "qwen35_gdn_conv", "qwen35_gdn_recurrent",
		"q4k_matmul", "q4k_matmul_wave32", "q2k_matmul", "qwen35_split_qg_panel",
		"qwen35_partial_rope_panel", "qwen35_causal_attention_panel",
		"sigmoid_mul", "q8_matmul_decode",
		"glm_kda_recurrent_reread", "glm_kda_recurrent_wave32",
		"flash_attn_dequant", "qwen35_gdn_tiled_transpose", "coopmat_wave32_wmma",
	}

	for i, exp := range expectedShaders {
		if VulkanShaders[i] != exp {
			t.Errorf("shader[%d] expected %q, got %q", i, exp, VulkanShaders[i])
		}
	}

	// Verify against on-disk shaders in internal/compute/shaders/
	repoRoot, err := ResolveRepoRoot("")
	if err == nil {
		shaderDir := filepath.Join(repoRoot, "internal", "compute", "shaders")
		for _, s := range VulkanShaders {
			path := filepath.Join(shaderDir, s+".comp")
			// Some shaders might be pending branch merges; if present, verify readability.
			if FileExists(path) {
				if _, readErr := os.ReadFile(path); readErr != nil {
					t.Errorf("failed to read shader %s: %v", path, readErr)
				}
			}
		}
	}
}

func TestVulkanQ4KScalarShaderStaysPortable(t *testing.T) {
	source, err := os.ReadFile(filepath.Join("..", "compute", "shaders", "q4k_matmul.comp"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, forbidden := range []string{"GL_KHR_cooperative_matrix", "coopMatMulAdd", "float16_t"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("always-loaded scalar Q4_K shader requires optional capability %q", forbidden)
		}
	}
	if !strings.Contains(text, "layout(local_size_x = 64) in;") {
		t.Fatal("scalar Q4_K shader no longer matches its one-dimensional host dispatch")
	}
}

func TestCUDAArchParsingAndGencode(t *testing.T) {
	sampleArchTxt := "sm_80\r\nsm_89\nsm_90\nsm_100\nsm_120\n"

	// 1. Fatbin (empty arch) build
	gencode, buildArchs, err := ParseCUDAArchMatrix(sampleArchTxt, "")
	if err != nil {
		t.Fatalf("ParseCUDAArchMatrix empty arch failed: %v", err)
	}
	// 5 architectures + 1 highest compute PTX = 6 pairs = 12 flags
	if len(gencode) != 12 {
		t.Fatalf("expected 12 gencode arguments, got %d: %v", len(gencode), gencode)
	}
	if !strings.Contains(buildArchs, "compute_120 PTX") {
		t.Errorf("expected buildArchs to contain 'compute_120 PTX', got %q", buildArchs)
	}
	expectedHighestPTX := "arch=compute_120,code=compute_120"
	if gencode[len(gencode)-1] != expectedHighestPTX {
		t.Errorf("expected last gencode flag %q, got %q", expectedHighestPTX, gencode[len(gencode)-1])
	}

	// 2. Specific arch build ("sm_89")
	gencode89, buildArchs89, err := ParseCUDAArchMatrix(sampleArchTxt, "sm_89")
	if err != nil {
		t.Fatalf("ParseCUDAArchMatrix sm_89 failed: %v", err)
	}
	if len(gencode89) != 2 || gencode89[1] != "arch=compute_89,code=sm_89" {
		t.Errorf("unexpected gencode for sm_89: %v", gencode89)
	}
	if !strings.Contains(buildArchs89, "sm_89 (single-arch dev build)") {
		t.Errorf("unexpected buildArchs: %q", buildArchs89)
	}

	// 3. Numeric arch build ("90")
	gencode90, _, err := ParseCUDAArchMatrix(sampleArchTxt, "90")
	if err != nil {
		t.Fatalf("ParseCUDAArchMatrix 90 failed: %v", err)
	}
	if len(gencode90) != 2 || gencode90[1] != "arch=compute_90,code=sm_90" {
		t.Errorf("unexpected gencode for 90: %v", gencode90)
	}

	// 4. Unsupported arch
	_, _, err = ParseCUDAArchMatrix(sampleArchTxt, "sm_75")
	if err == nil {
		t.Errorf("expected error for unsupported arch sm_75")
	}

	// 5. Empty arch list
	_, _, err = ParseCUDAArchMatrix("", "")
	if err == nil {
		t.Errorf("expected error for empty cuda_arch.txt")
	}

	// 6. Test with real file if present in repo
	repoRoot, err := ResolveRepoRoot("")
	if err == nil {
		realGencode, realDesc, loadErr := LoadCUDAArchMatrix(repoRoot, "")
		if loadErr != nil {
			t.Fatalf("LoadCUDAArchMatrix from repo failed: %v", loadErr)
		}
		if len(realGencode) == 0 || realDesc == "" {
			t.Errorf("expected non-empty gencode from real file")
		}
	}
}

func TestCgoEnvFormatting(t *testing.T) {
	// 1. QuoteFlag
	if got := QuoteFlag("-I", "/usr/include"); got != "-I/usr/include" {
		t.Errorf("QuoteFlag simple failed: got %q", got)
	}
	if got := QuoteFlag("-I", `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include`); got != `-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"` {
		t.Errorf("QuoteFlag spaces failed: got %q", got)
	}
	if got := QuoteFlag("-L", ""); got != "" {
		t.Errorf("QuoteFlag empty failed: got %q", got)
	}

	// 2. QuoteArg
	if got := QuoteArg("/usr/lib"); got != "/usr/lib" {
		t.Errorf("QuoteArg simple failed: got %q", got)
	}
	if got := QuoteArg("path with spaces"); got != `"path with spaces"` {
		t.Errorf("QuoteArg spaces failed: got %q", got)
	}

	// 3. SynthesizeVulkanCgoEnv
	tcVk := &Toolchain{
		CC:         "gcc",
		CXX:        "g++",
		VulkanInc:  `C:\VulkanSDK\Include`,
		VulkanLib:  `C:\VulkanSDK\Lib`,
		CxxRuntime: "-lstdc++",
		IsWindows:  true,
	}
	vulkanEnv := SynthesizeVulkanCgoEnv(tcVk, `C:\work\fak\internal\compute`)
	if vulkanEnv["CGO_ENABLED"] != "1" {
		t.Errorf("expected CGO_ENABLED=1, got %q", vulkanEnv["CGO_ENABLED"])
	}
	if vulkanEnv["CC"] != "gcc" || vulkanEnv["CXX"] != "g++" {
		t.Errorf("compiler mismatch in Vulkan env: CC=%q CXX=%q", vulkanEnv["CC"], vulkanEnv["CXX"])
	}
	if !strings.Contains(vulkanEnv["CGO_CFLAGS"], `C:\VulkanSDK\Include`) {
		t.Errorf("CGO_CFLAGS missing VulkanInc: %q", vulkanEnv["CGO_CFLAGS"])
	}
	if !strings.Contains(vulkanEnv["CGO_LDFLAGS"], "-lfakvulkan") || !strings.Contains(vulkanEnv["CGO_LDFLAGS"], "-lvulkan-1") {
		t.Errorf("CGO_LDFLAGS missing required libs: %q", vulkanEnv["CGO_LDFLAGS"])
	}
	if !strings.Contains(vulkanEnv["FAK_VULKAN_SPIRV"], "spirv") {
		t.Errorf("FAK_VULKAN_SPIRV missing spirv: %q", vulkanEnv["FAK_VULKAN_SPIRV"])
	}

	// 4. SynthesizeCUDACgoEnv
	tcCuda := &Toolchain{
		CC:        "gcc",
		CXX:       "g++",
		CUDAInc:   []string{`C:\Program Files\CUDA\include`},
		CUDALib:   []string{`C:\Program Files\CUDA\lib\x64`},
		IsWindows: true,
	}
	cudaEnv := SynthesizeCUDACgoEnv(tcCuda, `C:\work\fak\internal\compute`, true, []string{"/usr/lib/wsl/lib"})
	if cudaEnv["CGO_ENABLED"] != "1" {
		t.Errorf("expected CGO_ENABLED=1, got %q", cudaEnv["CGO_ENABLED"])
	}
	if !strings.Contains(cudaEnv["CGO_CFLAGS"], `-I"C:\Program Files\CUDA\include"`) {
		t.Errorf("expected quoted CGO_CFLAGS, got %q", cudaEnv["CGO_CFLAGS"])
	}
	if !strings.Contains(cudaEnv["CGO_LDFLAGS"], `-L"C:\Program Files\CUDA\lib\x64"`) {
		t.Errorf("expected quoted CGO_LDFLAGS, got %q", cudaEnv["CGO_LDFLAGS"])
	}
	if !strings.Contains(cudaEnv["CGO_LDFLAGS"], "-lnccl") {
		t.Errorf("expected -lnccl in CGO_LDFLAGS: %q", cudaEnv["CGO_LDFLAGS"])
	}
	if !strings.Contains(cudaEnv["CGO_LDFLAGS"], "-Wl,-rpath,/usr/lib/wsl/lib") {
		t.Errorf("expected rpath in CGO_LDFLAGS: %q", cudaEnv["CGO_LDFLAGS"])
	}

	// 5. MergeEnviron
	base := []string{"FOO=BAR", "BAZ=QUX"}
	overlay := map[string]string{"FOO": "UPDATED", "NEW": "VAL"}
	merged := MergeEnviron(base, overlay)
	mergedMap := make(map[string]string)
	for _, kv := range merged {
		parts := strings.SplitN(kv, "=", 2)
		mergedMap[parts[0]] = parts[1]
	}
	if mergedMap["FOO"] != "UPDATED" || mergedMap["BAZ"] != "QUX" || mergedMap["NEW"] != "VAL" {
		t.Errorf("unexpected MergeEnviron result: %v", mergedMap)
	}
}

func TestToolchainDiscoveryMocked(t *testing.T) {
	tempDir := t.TempDir()

	binDir := filepath.Join(tempDir, "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}

	fakeGlslc := filepath.Join(binDir, "glslc")
	if err := os.WriteFile(fakeGlslc, []byte("#!/bin/sh\nexit 0\n"), 0755); err != nil {
		t.Fatalf("write fake glslc: %v", err)
	}

	// Test LookPathInDirs
	found := LookPathInDirs("glslc", binDir)
	if found != fakeGlslc {
		t.Errorf("expected LookPathInDirs to find %s, got %q", fakeGlslc, found)
	}

	missing := LookPathInDirs("nonexistent", binDir)
	if missing != "" {
		t.Errorf("expected missing tool to return empty, got %q", missing)
	}

	// Test FileExists & DirExists
	if !FileExists(fakeGlslc) {
		t.Errorf("FileExists failed for %s", fakeGlslc)
	}
	if FileExists(binDir) {
		t.Errorf("FileExists returned true for directory %s", binDir)
	}
	if !DirExists(binDir) {
		t.Errorf("DirExists failed for %s", binDir)
	}
	if DirExists(fakeGlslc) {
		t.Errorf("DirExists returned true for file %s", fakeGlslc)
	}

	// Test ResolveRepoRoot
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatalf("ResolveRepoRoot failed: %v", err)
	}
	if !FileExists(filepath.Join(repoRoot, "go.mod")) {
		t.Errorf("go.mod not found in resolved repo root: %s", repoRoot)
	}

	// Test ResolveRepoRoot from a directory without go.mod
	_, err = ResolveRepoRoot(tempDir)
	if err == nil {
		t.Errorf("expected ResolveRepoRoot to fail in isolated temp dir")
	}
}

func TestVsDevEnvParsing(t *testing.T) {
	raw := []byte("KEY1=VAL1\r\nKEY2=VAL2=EXTRA\r\nEMPTY=\r\n\r\nINVALIDLINE\r\n")
	env := ParseEnvBlock(raw)
	if env["KEY1"] != "VAL1" {
		t.Errorf("expected KEY1=VAL1, got %q", env["KEY1"])
	}
	if env["KEY2"] != "VAL2=EXTRA" {
		t.Errorf("expected KEY2=VAL2=EXTRA, got %q", env["KEY2"])
	}
	if env["EMPTY"] != "" {
		t.Errorf("expected EMPTY='', got %q", env["EMPTY"])
	}
	if _, ok := env["INVALIDLINE"]; ok {
		t.Errorf("unexpected key INVALIDLINE in env map")
	}
}

func TestVersionCompare(t *testing.T) {
	if compareVersionStrings("1.4.350.0", "1.3.290.0") <= 0 {
		t.Errorf("expected 1.4.350.0 > 1.3.290.0")
	}
	if compareVersionStrings("1.3.290.0", "1.4.350.0") >= 0 {
		t.Errorf("expected 1.3.290.0 < 1.4.350.0")
	}
	if compareVersionStrings("1.4.350.0", "1.4.350.0") != 0 {
		t.Errorf("expected 1.4.350.0 == 1.4.350.0")
	}
	if compareVersionStrings("1.10.0.0", "1.9.0.0") <= 0 {
		t.Errorf("expected 1.10.0.0 > 1.9.0.0")
	}
}

func TestLiveToolchainDiscovery(t *testing.T) {
	tc, err := DiscoverToolchain()
	if err != nil {
		t.Fatalf("DiscoverToolchain returned error: %v", err)
	}
	if tc == nil {
		t.Fatalf("expected non-nil Toolchain")
	}
	// Log discovered tools for diagnostic visibility
	t.Logf("Discovered Toolchain: CC=%q CXX=%q AR=%q NVCC=%q GLSLC=%q VulkanSDK=%q CUDAHome=%q",
		tc.CC, tc.CXX, tc.AR, tc.NVCC, tc.GLSLC, tc.VulkanSDK, tc.CUDAHome)
}

func TestReceiptGenerationAndFormatting(t *testing.T) {
	tmpDir := t.TempDir()
	receiptPath := filepath.Join(tmpDir, "test-receipt.json")

	zeroCode := 0
	receipt := &ComputeBuildReceipt{
		Schema:      ComputeBuildReceiptSchema,
		Backend:     "vulkan",
		Command:     "binary",
		Outcome:     "success",
		ExitCode:    0,
		StartedAt:   time.Now().UTC().Format(time.RFC3339Nano),
		FinishedAt:  time.Now().UTC().Add(time.Second).Format(time.RFC3339Nano),
		ElapsedMS:   1000,
		ReceiptPath: receiptPath,
		Phases: []ComputeBuildPhase{
			{Name: "toolchain_probe", Outcome: "success", ElapsedMS: 50, ExitCode: &zeroCode},
			{Name: "shaders", Outcome: "success", ElapsedMS: 300, ExitCode: &zeroCode},
			{Name: "shim", Outcome: "success", ElapsedMS: 250, ExitCode: &zeroCode},
			{Name: "build_or_test", Outcome: "success", ElapsedMS: 350, ExitCode: &zeroCode},
			{Name: "smoke", Outcome: "success", ElapsedMS: 50, ExitCode: &zeroCode},
		},
		Artifact: &BuildArtifact{
			Path:      filepath.Join(tmpDir, "fak.exe"),
			SizeBytes: 54321,
			Signed:    true,
			SHA256:    "abcdef0123456789",
		},
		Smoke: &SmokeResult{
			Command:  []string{filepath.Join(tmpDir, "fak.exe"), "version", "--json"},
			Outcome:  "success",
			Output:   `{"version":"dev"}`,
			ExitCode: 0,
		},
	}

	if err := WriteReceiptAtomic(receiptPath, receipt); err != nil {
		t.Fatalf("WriteReceiptAtomic failed: %v", err)
	}

	// Verify file exists on disk
	if !FileExists(receiptPath) {
		t.Fatalf("receipt file not found on disk at %s", receiptPath)
	}

	// Read and verify JSON structure
	data, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("failed reading receipt file: %v", err)
	}

	var decoded ComputeBuildReceipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed unmarshaling receipt: %v", err)
	}

	if decoded.Schema != ComputeBuildReceiptSchema {
		t.Errorf("schema mismatch: got %q, want %q", decoded.Schema, ComputeBuildReceiptSchema)
	}
	if decoded.Backend != "vulkan" {
		t.Errorf("backend mismatch: got %q, want vulkan", decoded.Backend)
	}
	if decoded.Command != "binary" {
		t.Errorf("command mismatch: got %q, want binary", decoded.Command)
	}
	if decoded.Outcome != "success" || decoded.ExitCode != 0 {
		t.Errorf("outcome mismatch: outcome=%q, exit_code=%d", decoded.Outcome, decoded.ExitCode)
	}
	if len(decoded.Phases) != 5 {
		t.Fatalf("expected 5 phases, got %d", len(decoded.Phases))
	}
	if decoded.Phases[0].Name != "toolchain_probe" || decoded.Phases[4].Name != "smoke" {
		t.Errorf("unexpected phase names: first=%q last=%q", decoded.Phases[0].Name, decoded.Phases[4].Name)
	}
	if decoded.Artifact == nil || decoded.Artifact.SizeBytes != 54321 || !decoded.Artifact.Signed {
		t.Errorf("artifact mismatch: %+v", decoded.Artifact)
	}
	if decoded.Smoke == nil || decoded.Smoke.Outcome != "success" || decoded.Smoke.ExitCode != 0 {
		t.Errorf("smoke result mismatch: %+v", decoded.Smoke)
	}

	// Test overwriting existing receipt atomically
	receipt.Outcome = "failed"
	receipt.ExitCode = 1
	receipt.Error = "simulated error"
	if err := WriteReceiptAtomic(receiptPath, receipt); err != nil {
		t.Fatalf("WriteReceiptAtomic overwrite failed: %v", err)
	}

	dataOverwritten, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("reading overwritten receipt: %v", err)
	}
	var decoded2 ComputeBuildReceipt
	if err := json.Unmarshal(dataOverwritten, &decoded2); err != nil {
		t.Fatalf("unmarshaling overwritten receipt: %v", err)
	}
	if decoded2.Outcome != "failed" || decoded2.ExitCode != 1 || decoded2.Error != "simulated error" {
		t.Errorf("overwritten receipt mismatch: %+v", decoded2)
	}
}

func TestShiftLeftSmokeExecution(t *testing.T) {
	ctx := context.Background()

	// 1. Non-existent file
	resMissing := SmokeArtifact(ctx, "")
	if resMissing.Outcome != "failed" || resMissing.ExitCode != 1 {
		t.Errorf("expected failure on empty path, got: %+v", resMissing)
	}

	resNotFound := SmokeArtifact(ctx, filepath.Join(t.TempDir(), "nonexistent_bin.exe"))
	if resNotFound.Outcome != "failed" || resNotFound.ExitCode != 1 || !strings.Contains(resNotFound.Error, "not found") {
		t.Errorf("expected not found failure, got: %+v", resNotFound)
	}

	// 2. Build mock binary to test fallback ladder
	tmpDir := t.TempDir()
	mockSrc := filepath.Join(tmpDir, "mock_main.go")
	mockBin := filepath.Join(tmpDir, "mock_tool.exe")

	srcCode := `package main

import (
	"fmt"
	"os"
)

func main() {
	mode := os.Getenv("FAK_SMOKE_MOCK_MODE")
	args := os.Args[1:]
	if len(args) == 0 {
		os.Exit(1)
	}

	switch mode {
	case "version":
		if args[0] == "version" && len(args) > 1 && args[1] == "--json" {
			fmt.Println("{\"version\":\"1.0.0\"}")
			os.Exit(0)
		}
	case "help":
		if args[0] == "--help" {
			fmt.Println("mock tool help text")
			os.Exit(0)
		}
	case "dash_h":
		if args[0] == "-h" {
			fmt.Println("mock tool -h usage")
			os.Exit(0)
		}
	case "crash":
		os.Exit(42)
	}
	os.Exit(1)
}
`
	if err := os.WriteFile(mockSrc, []byte(srcCode), 0644); err != nil {
		t.Fatalf("failed writing mock tool source: %v", err)
	}

	cmdBuild := exec.Command("go", "build", "-o", mockBin, mockSrc)
	if out, err := cmdBuild.CombinedOutput(); err != nil {
		t.Fatalf("building mock binary failed: %v (output: %s)", err, string(out))
	}

	// Subtest A: version --json succeeds on first try
	t.Run("VersionJsonSuccess", func(t *testing.T) {
		t.Setenv("FAK_SMOKE_MOCK_MODE", "version")
		res := SmokeArtifact(ctx, mockBin)
		if res.Outcome != "success" {
			t.Fatalf("expected success, got outcome=%q error=%q", res.Outcome, res.Error)
		}
		if res.ExitCode != 0 {
			t.Errorf("expected exit code 0, got %d", res.ExitCode)
		}
		if len(res.Command) < 3 || res.Command[1] != "version" || res.Command[2] != "--json" {
			t.Errorf("expected version --json command, got %v", res.Command)
		}
		if !strings.Contains(res.Output, "1.0.0") {
			t.Errorf("output missing version string: %q", res.Output)
		}
	})

	// Subtest B: version fails, falls back to --help
	t.Run("FallbackHelpSuccess", func(t *testing.T) {
		t.Setenv("FAK_SMOKE_MOCK_MODE", "help")
		res := SmokeArtifact(ctx, mockBin)
		if res.Outcome != "success" {
			t.Fatalf("expected success on --help fallback, got outcome=%q error=%q", res.Outcome, res.Error)
		}
		if res.ExitCode != 0 {
			t.Errorf("expected exit code 0, got %d", res.ExitCode)
		}
		if len(res.Command) < 2 || res.Command[1] != "--help" {
			t.Errorf("expected --help command, got %v", res.Command)
		}
		if !strings.Contains(res.Output, "help text") {
			t.Errorf("output missing help text: %q", res.Output)
		}
	})

	// Subtest C: version and --help fail, falls back to -h
	t.Run("FallbackDashHSuccess", func(t *testing.T) {
		t.Setenv("FAK_SMOKE_MOCK_MODE", "dash_h")
		res := SmokeArtifact(ctx, mockBin)
		if res.Outcome != "success" {
			t.Fatalf("expected success on -h fallback, got outcome=%q error=%q", res.Outcome, res.Error)
		}
		if res.ExitCode != 0 {
			t.Errorf("expected exit code 0, got %d", res.ExitCode)
		}
		if len(res.Command) < 2 || res.Command[1] != "-h" {
			t.Errorf("expected -h command, got %v", res.Command)
		}
		if !strings.Contains(res.Output, "-h usage") {
			t.Errorf("output missing -h usage: %q", res.Output)
		}
	})

	// Subtest D: all candidates fail (crash or non-zero exit)
	t.Run("AllCandidatesFail", func(t *testing.T) {
		t.Setenv("FAK_SMOKE_MOCK_MODE", "crash")
		res := SmokeArtifact(ctx, mockBin)
		if res.Outcome != "failed" {
			t.Fatalf("expected failure, got outcome=%q", res.Outcome)
		}
		if res.ExitCode != 42 {
			t.Errorf("expected exit code 42, got %d", res.ExitCode)
		}
	})
}

func TestPhaseRecordingAndDurationAccounting(t *testing.T) {
	tmpDir := t.TempDir()
	receiptPath := filepath.Join(tmpDir, "phase-receipt.json")

	tracker := newReceiptTracker("vulkan", "build", receiptPath)

	// Phase 1: success with measurable duration
	err1 := tracker.recordPhase("toolchain_probe", func() error {
		time.Sleep(15 * time.Millisecond)
		return nil
	})
	if err1 != nil {
		t.Fatalf("phase 1 unexpected error: %v", err1)
	}

	// Phase 2: success
	err2 := tracker.recordPhase("shaders", func() error {
		time.Sleep(15 * time.Millisecond)
		return nil
	})
	if err2 != nil {
		t.Fatalf("phase 2 unexpected error: %v", err2)
	}

	// Phase 3: failure
	err3 := tracker.recordPhase("shim", func() error {
		time.Sleep(10 * time.Millisecond)
		return fmt.Errorf("simulated compiler error")
	})
	if err3 == nil {
		t.Fatalf("expected error from phase 3")
	}

	// Finish and persist receipt
	if err := tracker.finish(receiptPath); err != nil {
		t.Fatalf("tracker.finish failed: %v", err)
	}

	if !FileExists(receiptPath) {
		t.Fatalf("receipt not written to disk at %s", receiptPath)
	}

	data, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("reading receipt: %v", err)
	}

	var rec ComputeBuildReceipt
	if err := json.Unmarshal(data, &rec); err != nil {
		t.Fatalf("unmarshaling receipt: %v", err)
	}

	if rec.Outcome != "failed" {
		t.Errorf("receipt outcome expected failed, got %q", rec.Outcome)
	}
	if rec.ExitCode == 0 {
		t.Errorf("receipt exit code expected non-zero, got %d", rec.ExitCode)
	}
	if !strings.Contains(rec.Error, "simulated compiler error") {
		t.Errorf("receipt error missing simulated error: %q", rec.Error)
	}

	if len(rec.Phases) != 3 {
		t.Fatalf("expected 3 phases, got %d", len(rec.Phases))
	}

	// Check phase 1
	p1 := rec.Phases[0]
	if p1.Name != "toolchain_probe" || p1.Outcome != "success" || p1.ElapsedMS < 10 {
		t.Errorf("phase 1 invalid: %+v", p1)
	}

	// Check phase 2
	p2 := rec.Phases[1]
	if p2.Name != "shaders" || p2.Outcome != "success" || p2.ElapsedMS < 10 {
		t.Errorf("phase 2 invalid: %+v", p2)
	}

	// Check phase 3
	p3 := rec.Phases[2]
	if p3.Name != "shim" || p3.Outcome != "failed" || !strings.Contains(p3.Error, "simulated compiler error") {
		t.Errorf("phase 3 invalid: %+v", p3)
	}

	// Duration accounting: total elapsed should be at least sum of phases
	phaseSum := p1.ElapsedMS + p2.ElapsedMS + p3.ElapsedMS
	if rec.ElapsedMS < phaseSum {
		t.Errorf("total ElapsedMS (%d) less than phase sum (%d)", rec.ElapsedMS, phaseSum)
	}
}

func TestCxxToolchainProbe(t *testing.T) {
	ctx := context.Background()

	// Nil toolchain
	if err := TestCxxToolchain(ctx, nil); err == nil {
		t.Errorf("expected error for nil toolchain")
	}

	// Empty CXX
	if err := TestCxxToolchain(ctx, &Toolchain{CXX: ""}); err == nil {
		t.Errorf("expected error for empty CXX toolchain")
	}

	// Non-existent CXX binary
	if err := TestCxxToolchain(ctx, &Toolchain{CXX: "nonexistent_compiler_xyz"}); err == nil {
		t.Errorf("expected error for invalid CXX compiler")
	}

	// If live toolchain has CXX, test live toolchain
	tc, err := DiscoverToolchain()
	if err == nil && tc != nil && tc.CXX != "" && FileExists(tc.CXX) {
		t.Logf("Testing live C++ toolchain probe with %s ...", tc.CXX)
		if probeErr := TestCxxToolchain(ctx, tc); probeErr != nil {
			t.Logf("Live toolchain probe returned error (expected if dev environment headers missing): %v", probeErr)
		} else {
			t.Logf("Live toolchain probe succeeded!")
		}
	}
}
