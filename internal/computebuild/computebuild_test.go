package computebuild

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVulkanShadersCompleteness(t *testing.T) {
	if len(VulkanShaders) != 34 {
		t.Fatalf("expected 34 Vulkan shaders, got %d", len(VulkanShaders))
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
		"q4k_matmul", "q2k_matmul", "qwen35_split_qg_panel",
		"qwen35_partial_rope_panel", "qwen35_causal_attention_panel",
		"sigmoid_mul", "q8_matmul_decode",
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
