package computebuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestVulkanShadersCompleteness(t *testing.T) {
	if len(VulkanShaders) != 46 {
		t.Fatalf("expected 46 Vulkan shaders, got %d", len(VulkanShaders))
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
		"swiglu_q8_matmul_add", "qwen35_gdn_q8_in_proj", "qwen35_gdn_conv", "qwen35_gdn_recurrent",
		"q4k_matmul", "q4k_matmul_wave32", "q6k_matmul", "q2k_matmul", "qwen35_split_qg_panel",
		"qwen35_partial_rope_panel", "qwen35_causal_attention_panel",
		"sigmoid_mul", "q8_matmul_decode",
		"glm_kda_recurrent_reread", "glm_kda_recurrent_wave32",
		"flash_attn_dequant", "qwen35_gdn_tiled_transpose", "coopmat_wave32_wmma",
		"rmsnorm_q4k_matmul2", "swiglu_q4k_matmul_add",
		"qwen35_gdn_prefill_tiled", "qwen35_gdn_prefill_norm",
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

func TestVulkanQwen35GDNQ8InputProjectionLoaderIsRegistered(t *testing.T) {
	const stem = "qwen35_gdn_q8_in_proj"
	registered := 0
	for _, shader := range VulkanShaders {
		if shader == stem {
			registered++
		}
	}
	if registered != 1 {
		t.Fatalf("%s registry entries=%d, want exactly 1", stem, registered)
	}
	if _, err := os.Stat(filepath.Join("..", "compute", "shaders", stem+".comp")); err != nil {
		t.Fatalf("registered optional Q8 GDN input projection source: %v", err)
	}
	shim, err := os.ReadFile(filepath.Join("..", "compute", "vulkan_shim.cpp"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(shim), `P("`+stem+`.spv")`) {
		t.Fatalf("native optional loader does not request registered shader %s.spv", stem)
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
	linuxVulkanEnv := SynthesizeVulkanCgoEnv(&Toolchain{
		CC: "gcc", CXX: "g++", VulkanInc: "/usr/include", VulkanLib: "/usr/lib",
		CxxRuntime: "-lstdc++", IsWindows: false,
	}, "/work/fak/internal/compute")
	if flags := linuxVulkanEnv["CGO_LDFLAGS"]; !strings.Contains(flags, "-lvulkan") || strings.Contains(flags, "-lvulkan-1") {
		t.Errorf("Linux Vulkan CGO_LDFLAGS must select libvulkan, got %q", flags)
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
	isolatedDir, err := os.MkdirTemp(os.TempDir(), "fak-isolated-test-*")
	if err == nil {
		defer os.RemoveAll(isolatedDir)
		if _, err := ResolveRepoRoot(isolatedDir); err == nil {
			t.Errorf("expected ResolveRepoRoot to fail in isolated temp dir")
		}
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

func runFixtureGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func writeFixtureFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func buildFakeVulkanTool(t *testing.T, root string) string {
	t.Helper()
	source := filepath.Join(root, "fake-tool.go")
	binary := filepath.Join(root, "fake-tool")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	writeFixtureFile(t, source, `package main

import (
	"os"
	"path/filepath"
)

func main() {
	if marker := os.Getenv("FAK_VULKAN_TEST_TOOL_MARKER"); marker != "" {
		_ = os.WriteFile(marker, []byte("invoked"), 0644)
	}
	if os.Getenv("FAK_VULKAN_TEST_TOOL_MODE") == "crash" {
		os.Exit(139)
	}
	if target := os.Getenv("FAK_VULKAN_TEST_MUTATE_SOURCE"); target != "" {
		_ = os.WriteFile(target, []byte("mutated during build\n"), 0644)
	}
	args := os.Args[1:]
	var out string
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "-o" {
			out = args[i+1]
		}
	}
	if len(args) >= 2 && args[0] == "rcs" {
		out = args[1]
	}
	if out == "" {
		os.Exit(2)
	}
	if err := os.MkdirAll(filepath.Dir(out), 0755); err != nil {
		os.Exit(3)
	}
	if err := os.WriteFile(out, []byte("fak-fixture-output\n"), 0644); err != nil {
		os.Exit(4)
	}
}
`)
	cmd := exec.Command("go", "build", "-o", binary, source)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fake Vulkan tool: %v\n%s", err, out)
	}
	return binary
}

func newVulkanFixture(t *testing.T) (string, string, *Toolchain) {
	t.Helper()
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	writeFixtureFile(t, filepath.Join(repo, "go.mod"), "module example.test/vulkanfixture\n\ngo 1.26\n")
	writeFixtureFile(t, filepath.Join(repo, "README.md"), "fixture\n")
	writeFixtureFile(t, filepath.Join(repo, "cmd", "app", "main.go"), "package main\nfunc main() {}\n")
	writeFixtureFile(t, filepath.Join(repo, "internal", "compute", "vulkan_shim.cpp"), "int fixture;\n")
	for _, shader := range VulkanShaders {
		writeFixtureFile(t, filepath.Join(repo, "internal", "compute", "shaders", shader+".comp"), "#version 450\n")
		writeFixtureFile(t, filepath.Join(repo, "internal", "compute", "spirv", shader+".spv"), "fak-fixture-output\n")
	}
	writeFixtureFile(t, filepath.Join(repo, "internal", "compute", "vulkan_shim.o"), "fak-fixture-output\n")
	writeFixtureFile(t, filepath.Join(repo, "internal", "compute", "libfakvulkan.a"), "fak-fixture-output\n")
	runFixtureGit(t, root, "init", "-q", repo)
	runFixtureGit(t, repo, "config", "user.email", "fixture@example.invalid")
	runFixtureGit(t, repo, "config", "user.name", "Fixture")
	runFixtureGit(t, repo, "add", "-f", ".")
	runFixtureGit(t, repo, "commit", "-q", "-m", "fixture")
	tool := buildFakeVulkanTool(t, root)
	tc := &Toolchain{
		Go:         tool,
		CC:         tool,
		CXX:        tool,
		AR:         tool,
		GLSLC:      tool,
		CxxRuntime: "-lstdc++",
		IsWindows:  runtime.GOOS == "windows",
	}
	return root, repo, tc
}

func readBuildReceipt(t *testing.T, path string) ComputeBuildReceipt {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var receipt ComputeBuildReceipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		t.Fatal(err)
	}
	return receipt
}

func TestVulkanBinaryDirtySourceRefusesBeforeBuildTool(t *testing.T) {
	root, repo, tc := newVulkanFixture(t)
	marker := filepath.Join(root, "tool-invoked")
	t.Setenv("FAK_VULKAN_TEST_TOOL_MARKER", marker)
	writeFixtureFile(t, filepath.Join(repo, "README.md"), "fixture\ndirty\n")
	receiptPath := filepath.Join(root, "dirty-receipt.json")
	cfg := &VulkanConfig{
		Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
		OutBin: filepath.Join(root, "app"), ReceiptPath: receiptPath,
		SkipSmoke: true, Toolchain: tc,
	}
	err := RunVulkan(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "dirty") {
		t.Fatalf("RunVulkan error = %v, want dirty-source refusal", err)
	}
	if FileExists(marker) {
		t.Fatal("build tool executed before dirty-source refusal")
	}
	receipt := readBuildReceipt(t, receiptPath)
	if receipt.Schema != VulkanBuildReceiptSchema || receipt.Outcome != "failed" {
		t.Fatalf("receipt = %+v", receipt)
	}
	if len(receipt.Phases) != 1 || receipt.Phases[0].Name != "source_preflight" {
		t.Fatalf("phases = %+v, want only failed source_preflight", receipt.Phases)
	}
	if receipt.Artifact != nil || receipt.Vulkan != nil || receipt.Reproducibility != nil {
		t.Fatalf("failed dirty receipt retained success evidence: %+v", receipt)
	}
}

func TestVulkanBinaryToolFailureOmitsSuccessEvidence(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mode      string
		breakTool func(*Toolchain)
	}{
		{name: "missing tool", breakTool: func(tc *Toolchain) { tc.CXX = "" }},
		{name: "crash exit 139", mode: "crash", breakTool: func(*Toolchain) {}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root, repo, tools := newVulkanFixture(t)
			t.Setenv("FAK_VULKAN_TEST_TOOL_MODE", tc.mode)
			tc.breakTool(tools)
			receiptPath := filepath.Join(root, "failed-receipt.json")
			cfg := &VulkanConfig{
				Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
				OutBin: filepath.Join(root, "app"), ReceiptPath: receiptPath,
				SkipSmoke: true, Toolchain: tools,
			}
			if err := RunVulkan(context.Background(), cfg); err == nil {
				t.Fatal("RunVulkan unexpectedly succeeded")
			}
			receipt := readBuildReceipt(t, receiptPath)
			if receipt.Outcome != "failed" || receipt.Artifact != nil || receipt.Vulkan != nil || receipt.Reproducibility != nil {
				t.Fatalf("failed receipt retained success evidence: %+v", receipt)
			}
			if len(receipt.Phases) < 2 || receipt.Phases[0].Name != "source_preflight" || receipt.Phases[len(receipt.Phases)-1].Outcome != "failed" {
				t.Fatalf("unexpected failure phases: %+v", receipt.Phases)
			}
		})
	}
}

func TestVulkanBinaryRevalidatesSourceBeforeSuccess(t *testing.T) {
	root, repo, tc := newVulkanFixture(t)
	t.Setenv("FAK_VULKAN_TEST_MUTATE_SOURCE", filepath.Join(repo, "README.md"))
	receiptPath := filepath.Join(root, "mutated-receipt.json")
	cfg := &VulkanConfig{
		Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
		OutBin: filepath.Join(root, "app"), ReceiptPath: receiptPath,
		SkipSmoke: true, Toolchain: tc,
	}
	if err := RunVulkan(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "dirty") {
		t.Fatalf("source mutation error = %v, want dirty refusal", err)
	}
	receipt := readBuildReceipt(t, receiptPath)
	if receipt.Outcome != "failed" || receipt.Artifact != nil || receipt.Vulkan != nil || receipt.Reproducibility != nil {
		t.Fatalf("source mutation retained success evidence: %+v", receipt)
	}
	if got := receipt.Phases[len(receipt.Phases)-1].Name; got != "source_revalidation" {
		t.Fatalf("last phase = %q, want source_revalidation", got)
	}
}

func TestVulkanBinaryPropagatesReceiptWriteFailure(t *testing.T) {
	root, repo, tc := newVulkanFixture(t)
	receiptPath := filepath.Join(root, "receipt-destination")
	writeFixtureFile(t, filepath.Join(receiptPath, "keep"), "occupied\n")
	cfg := &VulkanConfig{
		Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
		OutBin: filepath.Join(root, "app"), ReceiptPath: receiptPath,
		SkipSmoke: true, Toolchain: tc,
	}
	if err := RunVulkan(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "write Vulkan build receipt") {
		t.Fatalf("receipt write error = %v", err)
	}
}

func TestMergeToolchainOverridesPreservesDiscoveredDefaults(t *testing.T) {
	base := &Toolchain{Go: "go-default", CC: "cc-default", CXX: "cxx-default", AR: "ar-default", GLSLC: "glslc-default"}
	got := mergeToolchainOverrides(base, &Toolchain{CXX: "cxx-explicit"})
	if got.CXX != "cxx-explicit" || got.Go != base.Go || got.CC != base.CC || got.AR != base.AR || got.GLSLC != base.GLSLC {
		t.Fatalf("merged toolchain = %+v", got)
	}
}

func TestVulkanBinaryReceiptBindsReproducibleSourceToolsShadersAndBinary(t *testing.T) {
	root, repo, tc := newVulkanFixture(t)
	firstSource, err := prepareVulkanSource(context.Background(), repo, "")
	if err != nil {
		t.Fatal(err)
	}
	secondSource, err := prepareVulkanSource(context.Background(), repo, "")
	if err != nil {
		t.Fatal(err)
	}
	if firstSource != secondSource || !firstSource.Clean || len(firstSource.GitCommit) != 40 || len(firstSource.GitTree) != 40 || len(firstSource.SourceArchiveSHA256) != 64 {
		t.Fatalf("source provenance is not deterministic and complete: first=%+v second=%+v", firstSource, secondSource)
	}
	if _, err := prepareVulkanSource(context.Background(), repo, strings.Repeat("0", 40)); err == nil {
		t.Fatal("absent requested commit unexpectedly accepted")
	}
	firstReceiptPath := filepath.Join(root, "first-receipt.json")
	first := &VulkanConfig{
		Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
		OutBin: filepath.Join(root, "app-first"), ReceiptPath: firstReceiptPath,
		SkipSmoke: true, Toolchain: tc,
	}
	if err := RunVulkan(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	firstReceipt := readBuildReceipt(t, firstReceiptPath)
	if firstReceipt.Schema != VulkanBuildReceiptSchema || firstReceipt.Artifact == nil || len(firstReceipt.Artifact.SHA256) != 64 || firstReceipt.Vulkan == nil {
		t.Fatalf("incomplete first receipt: %+v", firstReceipt)
	}
	if firstReceipt.Vulkan.Source != firstSource || firstReceipt.Vulkan.SPIRVModuleCount != len(VulkanShaders) || len(firstReceipt.Vulkan.SPIRVBundleSHA256) != 64 || len(firstReceipt.Vulkan.Toolchain) != 5 || len(firstReceipt.Vulkan.ToolchainSHA256) != 64 || len(firstReceipt.Vulkan.BuildCommandSHA256) != 64 || len(firstReceipt.Vulkan.StableIdentitySHA256) != 64 {
		t.Fatalf("incomplete first provenance: %+v", firstReceipt.Vulkan)
	}
	if firstReceipt.Reproducibility == nil || firstReceipt.Reproducibility.Status != "baseline" {
		t.Fatalf("first reproducibility = %+v", firstReceipt.Reproducibility)
	}

	secondReceiptPath := filepath.Join(root, "second-receipt.json")
	second := &VulkanConfig{
		Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
		OutBin: filepath.Join(root, "app-second"), ReceiptPath: secondReceiptPath,
		CompareReceiptPath: firstReceiptPath, SkipSmoke: true, Toolchain: tc,
	}
	if err := RunVulkan(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	secondReceipt := readBuildReceipt(t, secondReceiptPath)
	if secondReceipt.Reproducibility == nil || secondReceipt.Reproducibility.Status != "match" || len(secondReceipt.Reproducibility.ComparedReceiptSHA256) != 64 {
		t.Fatalf("second reproducibility = %+v", secondReceipt.Reproducibility)
	}
	if secondReceipt.Vulkan == nil || secondReceipt.Vulkan.StableIdentitySHA256 != firstReceipt.Vulkan.StableIdentitySHA256 {
		t.Fatalf("stable identity mismatch: first=%+v second=%+v", firstReceipt.Vulkan, secondReceipt.Vulkan)
	}
	if secondReceipt.StartedAt == firstReceipt.StartedAt || secondReceipt.ReceiptPath == firstReceipt.ReceiptPath {
		t.Fatal("volatile receipt metadata unexpectedly reused")
	}

	mismatchBaseline := firstReceipt
	mismatchArtifact := *firstReceipt.Artifact
	mismatchBaseline.Artifact = &mismatchArtifact
	mismatchBaseline.Artifact.SHA256 = strings.Repeat("0", 64)
	mismatchPath := filepath.Join(root, "mismatch-baseline.json")
	if err := WriteReceiptAtomic(mismatchPath, &mismatchBaseline); err != nil {
		t.Fatal(err)
	}
	mismatchReceiptPath := filepath.Join(root, "mismatch-result.json")
	mismatch := &VulkanConfig{
		Command: "binary", RepoRoot: repo, OutPkg: "./cmd/app",
		OutBin: filepath.Join(root, "app-mismatch"), ReceiptPath: mismatchReceiptPath,
		CompareReceiptPath: mismatchPath, SkipSmoke: true, Toolchain: tc,
	}
	if err := RunVulkan(context.Background(), mismatch); err == nil || !strings.Contains(err.Error(), "reproducibility mismatch") {
		t.Fatalf("mismatch error = %v", err)
	}
	mismatchReceipt := readBuildReceipt(t, mismatchReceiptPath)
	if mismatchReceipt.Artifact != nil || mismatchReceipt.Vulkan != nil || mismatchReceipt.Reproducibility == nil || mismatchReceipt.Reproducibility.Status != "mismatch" {
		t.Fatalf("mismatch receipt is not fail-closed: %+v", mismatchReceipt)
	}

	if err := os.Remove(filepath.Join(repo, "internal", "compute", "spirv", VulkanShaders[0]+".spv")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := hashSPIRVBundle(repo); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("missing SPIR-V module error = %v", err)
	}
}

type vulkanVerifierFixture struct {
	evidence VulkanBinaryReceiptEvidence
	receipt  ComputeBuildReceipt
	root     string
	gitRun   vulkanGitRunner
}

func pathVulkanGitRunner(gitPath string) vulkanGitRunner {
	return func(ctx context.Context, root string, args ...string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, gitPath, controlledVulkanGitArgs(root, args...)...)
		cmd.Env = controlledVulkanGitEnv()
		out, err := cmd.CombinedOutput()
		if err != nil {
			return nil, fmt.Errorf("fixture Git failed: %w: %s", err, out)
		}
		return out, nil
	}
}

func verifyVulkanFixture(ctx context.Context, fixture vulkanVerifierFixture, evidence VulkanBinaryReceiptEvidence) (*VulkanBinaryReceiptIdentityVerification, error) {
	if runtime.GOOS == "linux" {
		return VerifyVulkanBinaryReceiptIdentity(ctx, evidence)
	}
	return verifyVulkanBinaryReceiptIdentityWithRunners(ctx, evidence, fixture.gitRun, fixture.gitRun)
}

func newVulkanVerifierFixture(t *testing.T) vulkanVerifierFixture {
	t.Helper()
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	writeFixtureFile(t, filepath.Join(repo, "go.mod"), "module example.test/verifier\n\ngo 1.26\n")
	writeFixtureFile(t, filepath.Join(repo, "cmd", "fak", "main.go"), "package main\nfunc main() {}\n")
	spirvRoot := filepath.Join(repo, "internal", "compute", "spirv")
	for _, shader := range VulkanShaders {
		writeFixtureFile(t, filepath.Join(spirvRoot, shader+".spv"), "spirv:"+shader+"\n")
	}
	toolPath := filepath.Join(repo, "tooling", "inert-tool")
	writeFixtureFile(t, toolPath, "not an executable; verifier must only hash these bytes\n")
	if err := os.Chmod(toolPath, 0755); err != nil {
		t.Fatal(err)
	}
	binaryPath := filepath.Join(repo, "out", "fak")
	writeFixtureFile(t, binaryPath, "sealed mapped executable bytes\n")
	if err := os.Chmod(binaryPath, 0755); err != nil {
		t.Fatal(err)
	}
	runFixtureGit(t, root, "init", "-q", repo)
	runFixtureGit(t, repo, "config", "user.email", "fixture@example.invalid")
	runFixtureGit(t, repo, "config", "user.name", "Fixture")
	runFixtureGit(t, repo, "add", "-f", ".")
	runFixtureGit(t, repo, "commit", "-q", "-m", "fixture")
	commit := strings.TrimSpace(runFixtureGit(t, repo, "rev-parse", "HEAD"))
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
	gitSHA, _, err := strictFileSHA256(gitPath)
	if err != nil {
		t.Fatal(err)
	}
	gitRun := pathVulkanGitRunner(gitPath)
	binarySHA, binarySize, err := strictFileSHA256(binaryPath)
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(root, "receipt.json")
	evidence := VulkanBinaryReceiptEvidence{
		ReceiptPath: receiptPath, SourceRoot: repo, GitExecutable: gitPath, ExpectedGitSHA256: gitSHA, ExpectedCommit: commit,
		BinaryPath: binaryPath, ExpectedBinarySHA256: binarySHA, ExpectedBinarySize: binarySize,
		SPIRVRoot: spirvRoot, ExpectedSPIRVModules: append([]string(nil), VulkanShaders...),
		Tools:     VulkanReceiptToolInputs{Go: toolPath, CC: toolPath, CXX: toolPath, AR: toolPath, GLSLC: toolPath, CxxRuntime: "-lstdc++", IsWindows: runtime.GOOS == "windows"},
		BuildPlan: VulkanReceiptBuildPlan{PackageDir: filepath.Join(repo, "internal", "compute"), OutPackage: "./cmd/fak", Smoke: false},
	}
	observed, err := observeVulkanReceiptEvidence(context.Background(), evidence, gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchema, Backend: "vulkan", Command: "binary", Outcome: "success", ExitCode: 0,
		ReceiptPath: receiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan: &VulkanBuildProvenance{
			Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount,
			Toolchain: append([]BuildToolIdentity(nil), observed.Toolchain...), ToolchainSHA256: observed.ToolchainSHA256,
			NormalizedBuildCommand: append([]string(nil), observed.NormalizedBuildCommand...), BuildCommandSHA256: observed.BuildCommandSHA256,
			StableIdentitySHA256: observed.StableIdentitySHA256,
		},
		Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	writeVerifierReceipt(t, receiptPath, receipt)
	return vulkanVerifierFixture{evidence: evidence, receipt: receipt, root: root, gitRun: gitRun}
}

func writeVerifierReceipt(t *testing.T, path string, receipt ComputeBuildReceipt) []byte {
	t.Helper()
	b, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0644); err != nil {
		t.Fatal(err)
	}
	return b
}

func cloneVerifierReceipt(t *testing.T, receipt ComputeBuildReceipt) ComputeBuildReceipt {
	t.Helper()
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var clone ComputeBuildReceipt
	if err := json.Unmarshal(b, &clone); err != nil {
		t.Fatal(err)
	}
	return clone
}

func forgeVerifierReceiptDownstream(t *testing.T, receipt *ComputeBuildReceipt, evidence VulkanBinaryReceiptEvidence) {
	t.Helper()
	if receipt.Vulkan == nil || receipt.Artifact == nil {
		return
	}
	toolchainSHA, err := hashJSON(struct {
		Tools      []BuildToolIdentity `json:"tools"`
		CXXRuntime string              `json:"cxx_runtime"`
		IsWindows  bool                `json:"is_windows"`
	}{receipt.Vulkan.Toolchain, evidence.Tools.CxxRuntime, evidence.Tools.IsWindows})
	if err != nil {
		t.Fatal(err)
	}
	receipt.Vulkan.ToolchainSHA256 = toolchainSHA
	commandSHA, err := hashJSON(receipt.Vulkan.NormalizedBuildCommand)
	if err != nil {
		t.Fatal(err)
	}
	receipt.Vulkan.BuildCommandSHA256 = commandSHA
	stableSHA, err := vulkanStableIdentitySHA(receipt.Vulkan.Source, receipt.Vulkan.SPIRVBundleSHA256, receipt.Vulkan.SPIRVModuleCount, toolchainSHA, commandSHA, receipt.Artifact.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	receipt.Vulkan.StableIdentitySHA256 = stableSHA
}

func TestVerifyVulkanBinaryReceiptIdentityAcceptsIndependentFixture(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	verified, err := verifyVulkanFixture(context.Background(), fixture, fixture.evidence)
	if err != nil {
		t.Fatal(err)
	}
	if !validLowerSHA256(verified.ReceiptSHA256) || verified.HistoricalBuildCausality != "unavailable" || len(verified.UnavailableClaims) != len(unavailableVulkanV2Causality) {
		t.Fatalf("verification availability = %+v", verified)
	}
	if verified.Receipt.Vulkan == nil || verified.Receipt.Vulkan.StableIdentitySHA256 != fixture.receipt.Vulkan.StableIdentitySHA256 {
		t.Fatalf("verified receipt = %+v", verified.Receipt)
	}
}

func TestVerifyVulkanBinaryReceiptIdentityFailsClosedWithoutLinuxFDBinding(t *testing.T) {
	if runtime.GOOS == "linux" {
		t.Skip("Linux exercises the fd-bound production runner")
	}
	_, err := VerifyVulkanBinaryReceiptIdentity(context.Background(), VulkanBinaryReceiptEvidence{})
	if err == nil || !strings.Contains(err.Error(), "fd-bound Git execution is unsupported") {
		t.Fatalf("unsupported-platform error = %v", err)
	}
}

func TestPinnedLinuxGitRunnerExecutesVerifiedDescriptorAfterPathSwap(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux /proc/self/fd contract")
	}
	fixture := newVulkanVerifierFixture(t)
	pinnedPath := filepath.Join(fixture.root, "policy-git")
	gitBytes, err := os.ReadFile(fixture.evidence.GitExecutable)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pinnedPath, gitBytes, 0755); err != nil {
		t.Fatal(err)
	}
	pinnedSHA, _, err := strictFileSHA256(pinnedPath)
	if err != nil {
		t.Fatal(err)
	}
	run, closeGit, err := pinnedLinuxGitRunner(pinnedPath, pinnedSHA)
	if err != nil {
		t.Fatal(err)
	}
	defer closeGit()
	parked := pinnedPath + ".verified"
	if err := os.Rename(pinnedPath, parked); err != nil {
		t.Fatal(err)
	}
	defer os.Rename(parked, pinnedPath)
	if err := os.WriteFile(pinnedPath, []byte("forged pathname bytes\n"), 0755); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(pinnedPath)
	out, err := run(context.Background(), fixture.evidence.SourceRoot, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		t.Fatalf("fd-bound Git did not survive pathname swap: %v", err)
	}
	if got := strings.TrimSpace(string(out)); got != fixture.evidence.ExpectedCommit {
		t.Fatalf("fd-bound Git commit = %q, want %q", got, fixture.evidence.ExpectedCommit)
	}
}

func TestVerifyVulkanBinaryReceiptIdentityNeutralizesHostileRepoConfigBeforeObservation(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	marker := filepath.Join(fixture.root, "fsmonitor-was-executed")
	command := fmt.Sprintf("echo invoked > %q", filepath.ToSlash(marker))
	runFixtureGit(t, fixture.evidence.SourceRoot, "config", "core.fsmonitor", command)
	if _, err := verifyVulkanFixture(context.Background(), fixture, fixture.evidence); err != nil {
		t.Fatalf("normalized fsmonitor configuration rejected valid evidence: %v", err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("hostile fsmonitor executed before verification: %v", err)
	}

	runFixtureGit(t, fixture.evidence.SourceRoot, "config", "extensions.partialClone", "origin")
	runFixtureGit(t, fixture.evidence.SourceRoot, "config", "remote.origin.promisor", "true")
	if _, err := verifyVulkanFixture(context.Background(), fixture, fixture.evidence); err == nil || !strings.Contains(err.Error(), "unsupported repo-admin configuration") {
		t.Fatalf("promisor configuration error = %v", err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("hostile configuration caused side effects: %v", err)
	}
}

func TestVerifyVulkanBinaryReceiptIdentityRejectsSelfConsistentForgeries(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	observed, err := observeVulkanReceiptEvidence(context.Background(), fixture.evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	mutations := []struct {
		name       string
		mutate     func(*ComputeBuildReceipt)
		afterForge func(*ComputeBuildReceipt)
	}{
		{"schema", func(r *ComputeBuildReceipt) { r.Schema = ComputeBuildReceiptSchema }, nil},
		{"outcome", func(r *ComputeBuildReceipt) { r.Outcome = "failed" }, nil},
		{"receipt path", func(r *ComputeBuildReceipt) { r.ReceiptPath += ".forged" }, nil},
		{"source commit", func(r *ComputeBuildReceipt) { r.Vulkan.Source.GitCommit = strings.Repeat("0", 40) }, nil},
		{"source tree", func(r *ComputeBuildReceipt) { r.Vulkan.Source.GitTree = strings.Repeat("0", 40) }, nil},
		{"source clean", func(r *ComputeBuildReceipt) { r.Vulkan.Source.Clean = false }, nil},
		{"source archive", func(r *ComputeBuildReceipt) { r.Vulkan.Source.SourceArchiveSHA256 = strings.Repeat("0", 64) }, nil},
		{"spirv bundle", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64) }, nil},
		{"spirv count", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVModuleCount-- }, nil},
		{"tool role", func(r *ComputeBuildReceipt) { r.Vulkan.Toolchain[0].Role = "forged" }, nil},
		{"tool executable", func(r *ComputeBuildReceipt) { r.Vulkan.Toolchain[0].Executable = "forged" }, nil},
		{"tool digest", func(r *ComputeBuildReceipt) { r.Vulkan.Toolchain[0].SHA256 = strings.Repeat("0", 64) }, nil},
		{"toolchain digest", func(*ComputeBuildReceipt) {}, func(r *ComputeBuildReceipt) { r.Vulkan.ToolchainSHA256 = strings.Repeat("0", 64) }},
		{"normalized command", func(r *ComputeBuildReceipt) { r.Vulkan.NormalizedBuildCommand[0] += "|forged" }, nil},
		{"command digest", func(*ComputeBuildReceipt) {}, func(r *ComputeBuildReceipt) { r.Vulkan.BuildCommandSHA256 = strings.Repeat("0", 64) }},
		{"binary path", func(r *ComputeBuildReceipt) { r.Artifact.Path += ".forged" }, nil},
		{"binary size", func(r *ComputeBuildReceipt) { r.Artifact.SizeBytes++ }, nil},
		{"binary digest", func(r *ComputeBuildReceipt) { r.Artifact.SHA256 = strings.Repeat("0", 64) }, nil},
		{"binary signed", func(r *ComputeBuildReceipt) { r.Artifact.Signed = true }, nil},
		{"stable digest", func(*ComputeBuildReceipt) {}, func(r *ComputeBuildReceipt) { r.Vulkan.StableIdentitySHA256 = strings.Repeat("0", 64) }},
		{"reproducibility status", func(r *ComputeBuildReceipt) { r.Reproducibility.Status = "match" }, nil},
		{"reproducibility digest", func(r *ComputeBuildReceipt) { r.Reproducibility.ComparedReceiptSHA256 = strings.Repeat("0", 64) }, nil},
		{"reproducibility mismatches", func(r *ComputeBuildReceipt) { r.Reproducibility.MismatchedFields = []string{"binary"} }, nil},
		{"legacy provenance", func(r *ComputeBuildReceipt) { r.GitCommit = fixture.evidence.ExpectedCommit }, nil},
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			receipt := cloneVerifierReceipt(t, fixture.receipt)
			mutation.mutate(&receipt)
			forgeVerifierReceiptDownstream(t, &receipt, fixture.evidence)
			if mutation.afterForge != nil {
				mutation.afterForge(&receipt)
			}
			err := compareReceiptToObservation(receipt, observed)
			if err == nil {
				err = verifyBaselineReproducibility(receipt)
			}
			if err == nil {
				t.Fatal("self-consistent forged receipt unexpectedly accepted")
			}
		})
	}
}

func TestVerifyVulkanBinaryReceiptIdentityRejectsUnknownTrailingAndIndependentDrift(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	raw := writeVerifierReceipt(t, fixture.evidence.ReceiptPath, fixture.receipt)
	unknown := append([]byte(nil), raw[:len(raw)-2]...)
	unknown = append(unknown, []byte(",\n  \"forged\": true\n}\n")...)
	if err := os.WriteFile(fixture.evidence.ReceiptPath, unknown, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyVulkanFixture(context.Background(), fixture, fixture.evidence); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown-field error = %v", err)
	}
	if err := os.WriteFile(fixture.evidence.ReceiptPath, append(raw, []byte("{}\n")...), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyVulkanFixture(context.Background(), fixture, fixture.evidence); err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("trailing-data error = %v", err)
	}
	writeVerifierReceipt(t, fixture.evidence.ReceiptPath, fixture.receipt)
	evidence := fixture.evidence
	evidence.ExpectedBinarySHA256 = strings.Repeat("0", 64)
	if _, err := verifyVulkanFixture(context.Background(), fixture, evidence); err == nil || !strings.Contains(err.Error(), "sealed mapped executable") {
		t.Fatalf("sealed-binary error = %v", err)
	}
	evidence = fixture.evidence
	evidence.ExpectedGitSHA256 = strings.Repeat("0", 64)
	if _, err := verifyVulkanFixture(context.Background(), fixture, evidence); err == nil || !strings.Contains(err.Error(), "policy-pinned Git") {
		t.Fatalf("pinned-Git error = %v", err)
	}
	writeFixtureFile(t, filepath.Join(fixture.evidence.SourceRoot, "untracked"), "dirty\n")
	if _, err := verifyVulkanFixture(context.Background(), fixture, fixture.evidence); err == nil || !strings.Contains(err.Error(), "not clean") {
		t.Fatalf("dirty-source error = %v", err)
	}
}

func TestVerifyVulkanBinaryReceiptIdentityRejectsInvalidSPIRVCensus(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	t.Run("duplicate registry", func(t *testing.T) {
		registry := append([]string(nil), fixture.evidence.ExpectedSPIRVModules...)
		registry[len(registry)-1] = registry[0]
		if _, _, err := hashObservedSPIRVBundle(fixture.evidence.SPIRVRoot, registry); err == nil || !strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("duplicate registry error = %v", err)
		}
	})
	t.Run("missing module", func(t *testing.T) {
		missing := filepath.Join(fixture.evidence.SPIRVRoot, fixture.evidence.ExpectedSPIRVModules[0]+".spv")
		parked := missing + ".missing"
		if err := os.Rename(missing, parked); err != nil {
			t.Fatal(err)
		}
		defer os.Rename(parked, missing)
		if _, _, err := hashObservedSPIRVBundle(fixture.evidence.SPIRVRoot, fixture.evidence.ExpectedSPIRVModules); err == nil || !strings.Contains(err.Error(), "incomplete") {
			t.Fatalf("missing module error = %v", err)
		}
	})
	t.Run("extra module", func(t *testing.T) {
		extra := filepath.Join(fixture.evidence.SPIRVRoot, "forged-extra.spv")
		writeFixtureFile(t, extra, "extra\n")
		defer os.Remove(extra)
		if _, _, err := hashObservedSPIRVBundle(fixture.evidence.SPIRVRoot, fixture.evidence.ExpectedSPIRVModules); err == nil || !strings.Contains(err.Error(), "unexpected") {
			t.Fatalf("extra module error = %v", err)
		}
	})
	t.Run("symlink module", func(t *testing.T) {
		target := filepath.Join(fixture.evidence.SPIRVRoot, fixture.evidence.ExpectedSPIRVModules[1]+".spv")
		link := filepath.Join(fixture.evidence.SPIRVRoot, fixture.evidence.ExpectedSPIRVModules[0]+".spv")
		parked := link + ".regular"
		if err := os.Rename(link, parked); err != nil {
			t.Fatal(err)
		}
		defer os.Rename(parked, link)
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("symlink creation unavailable: %v", err)
		}
		defer os.Remove(link)
		if _, _, err := hashObservedSPIRVBundle(fixture.evidence.SPIRVRoot, fixture.evidence.ExpectedSPIRVModules); err == nil || !strings.Contains(err.Error(), "non-symlink") {
			t.Fatalf("symlink module error = %v", err)
		}
	})
}

func TestVerifyVulkanBinaryReceiptIdentityRequiresAuthenticatedComparison(t *testing.T) {
	fixture := newVulkanVerifierFixture(t)
	priorPath := filepath.Join(fixture.root, "prior.json")
	prior := cloneVerifierReceipt(t, fixture.receipt)
	prior.ReceiptPath = priorPath
	priorRaw := writeVerifierReceipt(t, priorPath, prior)
	currentPath := filepath.Join(fixture.root, "current.json")
	current := cloneVerifierReceipt(t, fixture.receipt)
	current.ReceiptPath = currentPath
	priorDigest := sha256.Sum256(priorRaw)
	current.Reproducibility = &BuildReproducibility{Status: "match", ComparedReceiptSHA256: hex.EncodeToString(priorDigest[:])}
	writeVerifierReceipt(t, currentPath, current)
	priorEvidence := fixture.evidence
	priorEvidence.ReceiptPath = priorPath
	evidence := fixture.evidence
	evidence.ReceiptPath = currentPath
	evidence.Comparison = &priorEvidence
	if _, err := verifyVulkanFixture(context.Background(), fixture, evidence); err != nil {
		t.Fatal(err)
	}

	forgedPrior := cloneVerifierReceipt(t, prior)
	forgedPrior.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64)
	forgeVerifierReceiptDownstream(t, &forgedPrior, priorEvidence)
	forgedPriorRaw := writeVerifierReceipt(t, priorPath, forgedPrior)
	forgedDigest := sha256.Sum256(forgedPriorRaw)
	current.Reproducibility.ComparedReceiptSHA256 = hex.EncodeToString(forgedDigest[:])
	writeVerifierReceipt(t, currentPath, current)
	if _, err := verifyVulkanFixture(context.Background(), fixture, evidence); err == nil {
		t.Fatal("independently invalid comparison receipt unexpectedly accepted")
	}
}
