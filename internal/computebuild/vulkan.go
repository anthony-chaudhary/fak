package computebuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

var (
	gitStatusCmd = func(ctx context.Context, repoRoot string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, "git", "-C", repoRoot, "status", "--porcelain")
		return cmd.Output()
	}
	gitRevParseCmd = func(ctx context.Context, repoRoot, arg string) (string, error) {
		cmd := exec.CommandContext(ctx, "git", "-C", repoRoot, "rev-parse", arg)
		out, err := cmd.Output()
		return strings.TrimSpace(string(out)), err
	}
	gitArchiveCmd = func(ctx context.Context, repoRoot, commit string, w io.Writer) error {
		cmd := exec.CommandContext(ctx, "git", "-C", repoRoot, "archive", "--format=tar", commit)
		cmd.Stdout = w
		var errBuf bytes.Buffer
		cmd.Stderr = &errBuf
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("%w: %s", err, errBuf.String())
		}
		return nil
	}
)

// VulkanShaders contains the GLSL compute shaders compiled for the Vulkan backend,
// exactly matching internal/compute/build_vulkan.ps1.
var VulkanShaders = []string{
	"matmul",
	"matmul_add",
	"matmul_argmax",
	"matmul_argmax_blocks",
	"matmul2",
	"matmul3",
	"rmsnorm",
	"rmsnorm_matmul",
	"rmsnorm_matmul2",
	"rmsnorm_matmul3",
	"rmsnorm_matmul_argmax_blocks",
	"rope",
	"swiglu",
	"swiglu_matmul_add",
	"add",
	"add_bias",
	"attention",
	"argmax",
	"argmax_pairs",
	"q8_matmul",
	"q8_matmul2",
	"q8_matmul3",
	"rmsnorm_q8_matmul2",
	"rmsnorm_q8_matmul3",
	"swiglu_q8_matmul_add",
	"qwen35_gdn_conv",
	"qwen35_gdn_recurrent",
	"q4k_matmul",
	"q4k_matmul_wave32",
	"q2k_matmul",
	"qwen35_split_qg_panel",
	"qwen35_partial_rope_panel",
	"qwen35_causal_attention_panel",
	"sigmoid_mul",
	"q8_matmul_decode",
	"glm_kda_recurrent_reread",
	"glm_kda_recurrent_wave32",
	"flash_attn_dequant",
	"qwen35_gdn_tiled_transpose",
	"coopmat_wave32_wmma",
	"rmsnorm_q4k_matmul2",
	"swiglu_q4k_matmul_add",
}

// BuildShaders compiles all registered GLSL shaders to SPIR-V using glslc.
func BuildShaders(ctx context.Context, tc *Toolchain, repoRoot string, stdout io.Writer) error {
	if tc == nil || tc.GLSLC == "" {
		return fmt.Errorf("glslc not found — set VULKAN_SDK or install Vulkan SDK")
	}

	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	shaderSrc := filepath.Join(pkgDir, "shaders")
	spvOut := filepath.Join(pkgDir, "spirv")

	if err := os.MkdirAll(spvOut, 0755); err != nil {
		return fmt.Errorf("creating spirv output directory: %w", err)
	}

	compiledCount := 0
	for _, s := range VulkanShaders {
		src := filepath.Join(shaderSrc, s+".comp")
		dst := filepath.Join(spvOut, s+".spv")

		if !FileExists(src) {
			return fmt.Errorf("required shader source missing: %s", src)
		}

		if stdout != nil {
			fmt.Fprintf(stdout, "[vulkan] glslc %s.comp -> %s.spv\n", s, s)
		}
		args := []string{"-O", "--target-env=vulkan1.2", "-fshader-stage=comp", src, "-o", dst}
		if err := RunCmd(ctx, tc.GLSLC, args, pkgDir, nil, stdout, os.Stderr); err != nil {
			return fmt.Errorf("glslc failed on %s.comp: %w", s, err)
		}
		compiledCount++
	}

	if compiledCount == 0 {
		return fmt.Errorf("no shaders compiled in %s", shaderSrc)
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] compiled %d SPIR-V modules to %s\n", compiledCount, spvOut)
	}
	return nil
}

// BuildVulkanShim compiles vulkan_shim.cpp into vulkan_shim.o and archives it to libfakvulkan.a.
func BuildVulkanShim(ctx context.Context, tc *Toolchain, repoRoot string, stdout io.Writer) (*BuildArtifact, error) {
	if tc == nil || tc.CXX == "" {
		return nil, fmt.Errorf("C++ compiler not found (clang++ or g++)")
	}
	if tc.AR == "" {
		return nil, fmt.Errorf("archiver not found (ar or llvm-ar)")
	}

	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	shimCpp := filepath.Join(pkgDir, "vulkan_shim.cpp")
	shimObj := filepath.Join(pkgDir, "vulkan_shim.o")
	libOut := filepath.Join(pkgDir, "libfakvulkan.a")

	cxxArgs := []string{"-O3", "-std=c++17"}
	if tc.IsWindows {
		cxxArgs = append(cxxArgs, "-D_CRT_SECURE_NO_WARNINGS")
	} else {
		cxxArgs = append(cxxArgs, "-fPIC")
	}
	if tc.VulkanInc != "" {
		cxxArgs = append(cxxArgs, "-I"+tc.VulkanInc)
	}
	cxxArgs = append(cxxArgs, "-c", shimCpp, "-o", shimObj)

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] C++ compile vulkan_shim.cpp -> libfakvulkan.a\n")
	}

	env := os.Environ()
	if len(tc.VsDevEnv) > 0 {
		env = MergeEnviron(env, tc.VsDevEnv)
	}
	if err := RunCmd(ctx, tc.CXX, cxxArgs, pkgDir, env, stdout, os.Stderr); err != nil {
		return nil, fmt.Errorf("compiling vulkan_shim.cpp: %w", err)
	}

	arArgs := []string{"rcs", libOut, shimObj}
	if err := RunCmd(ctx, tc.AR, arArgs, pkgDir, env, stdout, os.Stderr); err != nil {
		return nil, fmt.Errorf("archiving libfakvulkan.a: %w", err)
	}

	fi, err := os.Stat(libOut)
	if err != nil {
		return nil, fmt.Errorf("stat libfakvulkan.a: %w", err)
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] built libfakvulkan.a (%d bytes)\n", fi.Size())
	}
	return &BuildArtifact{
		Path:      libOut,
		SizeBytes: fi.Size(),
	}, nil
}

// TestCxxToolchain performs a shift-left C++ compilation probe, proving the C++
// compiler and standard library are functional before running heavy build tasks.
func TestCxxToolchain(ctx context.Context, tc *Toolchain) error {
	if tc == nil || tc.CXX == "" {
		return fmt.Errorf("C++ compiler not found (clang++ or g++)")
	}

	tmpDir, err := os.MkdirTemp("", "fak-cxx-probe-*")
	if err != nil {
		return fmt.Errorf("creating temp probe dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	probeSrc := filepath.Join(tmpDir, "probe.cpp")
	probeObj := filepath.Join(tmpDir, "probe.o")

	srcContent := []byte("#include <vector>\n#include <string>\nint main() {\n    std::vector<std::string> v;\n    v.push_back(\"fak_probe\");\n    return v.empty() ? 1 : 0;\n}\n")
	if err := os.WriteFile(probeSrc, srcContent, 0644); err != nil {
		return fmt.Errorf("writing probe source: %w", err)
	}

	cxxArgs := []string{"-O3", "-std=c++17"}
	if tc.IsWindows {
		cxxArgs = append(cxxArgs, "-D_CRT_SECURE_NO_WARNINGS")
	} else {
		cxxArgs = append(cxxArgs, "-fPIC")
	}
	cxxArgs = append(cxxArgs, "-c", probeSrc, "-o", probeObj)

	env := os.Environ()
	if len(tc.VsDevEnv) > 0 {
		env = MergeEnviron(env, tc.VsDevEnv)
	}

	if err := RunCmd(ctx, tc.CXX, cxxArgs, tmpDir, env, nil, nil); err != nil {
		return fmt.Errorf("C++ toolchain probe failed: %w", err)
	}
	return nil
}

// ComputeSPIRVBundleSHA256 computes a deterministic, canonical SHA-256 digest over
// all .spv files in spvDir, sorted lexicographically by filename.
func ComputeSPIRVBundleSHA256(spvDir string) (string, error) {
	entries, err := os.ReadDir(spvDir)
	if err != nil {
		return "", fmt.Errorf("reading spirv directory %s: %w", spvDir, err)
	}
	var spvFiles []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".spv") {
			spvFiles = append(spvFiles, e.Name())
		}
	}
	if len(spvFiles) == 0 {
		return "", fmt.Errorf("no .spv files found in %s", spvDir)
	}
	sort.Strings(spvFiles)

	bundleHasher := sha256.New()
	for _, name := range spvFiles {
		filePath := filepath.Join(spvDir, name)
		data, err := os.ReadFile(filePath)
		if err != nil {
			return "", fmt.Errorf("reading spirv file %s: %w", filePath, err)
		}
		fileSum := sha256.Sum256(data)
		fmt.Fprintf(bundleHasher, "%s  %s\n", hex.EncodeToString(fileSum[:]), name)
	}
	return hex.EncodeToString(bundleHasher.Sum(nil)), nil
}

// CompareReceiptProvenance verifies that all stable provenance fields between two receipts match,
// proving reproducibility. Non-reproducible execution metadata (timestamps, elapsed durations,
// receipt file paths, smoke timing) are permitted to vary.
func CompareReceiptProvenance(a, b *ComputeBuildReceipt) error {
	if a == nil || b == nil {
		return fmt.Errorf("receipt cannot be nil (a=%v, b=%v)", a != nil, b != nil)
	}
	if a.Schema != b.Schema {
		return fmt.Errorf("schema mismatch: %q vs %q", a.Schema, b.Schema)
	}
	if a.Backend != b.Backend {
		return fmt.Errorf("backend mismatch: %q vs %q", a.Backend, b.Backend)
	}
	if a.Command != b.Command {
		return fmt.Errorf("command mismatch: %q vs %q", a.Command, b.Command)
	}
	if a.Outcome != b.Outcome {
		return fmt.Errorf("outcome mismatch: %q vs %q", a.Outcome, b.Outcome)
	}
	if a.ExitCode != b.ExitCode {
		return fmt.Errorf("exit code mismatch: %d vs %d", a.ExitCode, b.ExitCode)
	}
	if a.GitCommit != b.GitCommit {
		return fmt.Errorf("git commit mismatch: %q vs %q", a.GitCommit, b.GitCommit)
	}
	if a.GitRef != b.GitRef {
		return fmt.Errorf("git ref mismatch: %q vs %q", a.GitRef, b.GitRef)
	}
	if (a.Clean == nil) != (b.Clean == nil) {
		return fmt.Errorf("clean status mismatch: %v vs %v", a.Clean, b.Clean)
	}
	if a.Clean != nil && *a.Clean != *b.Clean {
		return fmt.Errorf("clean status mismatch: %t vs %t", *a.Clean, *b.Clean)
	}
	if a.SourceArchiveSHA256 != b.SourceArchiveSHA256 {
		return fmt.Errorf("source archive SHA-256 mismatch: %q vs %q", a.SourceArchiveSHA256, b.SourceArchiveSHA256)
	}
	if a.ShaderBundleSHA256 != b.ShaderBundleSHA256 {
		return fmt.Errorf("shader bundle SHA-256 mismatch: %q vs %q", a.ShaderBundleSHA256, b.ShaderBundleSHA256)
	}
	if (a.Artifact == nil) != (b.Artifact == nil) {
		return fmt.Errorf("artifact presence mismatch: %v vs %v", a.Artifact != nil, b.Artifact != nil)
	}
	if a.Artifact != nil {
		if a.Artifact.SHA256 != b.Artifact.SHA256 {
			return fmt.Errorf("binary SHA-256 mismatch: %q vs %q", a.Artifact.SHA256, b.Artifact.SHA256)
		}
		if a.Artifact.SizeBytes != b.Artifact.SizeBytes {
			return fmt.Errorf("binary size mismatch: %d vs %d", a.Artifact.SizeBytes, b.Artifact.SizeBytes)
		}
	}
	if len(a.BuildArgs) != len(b.BuildArgs) {
		return fmt.Errorf("build args count mismatch: %d vs %d", len(a.BuildArgs), len(b.BuildArgs))
	}
	for i := range a.BuildArgs {
		if a.BuildArgs[i] != b.BuildArgs[i] {
			return fmt.Errorf("build arg[%d] mismatch: %q vs %q", i, a.BuildArgs[i], b.BuildArgs[i])
		}
	}
	if (a.Toolchain == nil) != (b.Toolchain == nil) {
		return fmt.Errorf("toolchain identity presence mismatch: %v vs %v", a.Toolchain != nil, b.Toolchain != nil)
	}
	if a.Toolchain != nil {
		if a.Toolchain.GLSLC != b.Toolchain.GLSLC {
			return fmt.Errorf("toolchain GLSLC mismatch: %q vs %q", a.Toolchain.GLSLC, b.Toolchain.GLSLC)
		}
		if a.Toolchain.CXX != b.Toolchain.CXX {
			return fmt.Errorf("toolchain CXX mismatch: %q vs %q", a.Toolchain.CXX, b.Toolchain.CXX)
		}
		if a.Toolchain.AR != b.Toolchain.AR {
			return fmt.Errorf("toolchain AR mismatch: %q vs %q", a.Toolchain.AR, b.Toolchain.AR)
		}
		if a.Toolchain.CC != b.Toolchain.CC {
			return fmt.Errorf("toolchain CC mismatch: %q vs %q", a.Toolchain.CC, b.Toolchain.CC)
		}
		if a.Toolchain.VulkanSDK != b.Toolchain.VulkanSDK {
			return fmt.Errorf("toolchain VulkanSDK mismatch: %q vs %q", a.Toolchain.VulkanSDK, b.Toolchain.VulkanSDK)
		}
		if a.Toolchain.CxxRuntime != b.Toolchain.CxxRuntime {
			return fmt.Errorf("toolchain CxxRuntime mismatch: %q vs %q", a.Toolchain.CxxRuntime, b.Toolchain.CxxRuntime)
		}
	}
	return nil
}

func checkSourceProvenance(ctx context.Context, cfg *VulkanConfig, tracker *receiptTracker) error {
	gitStatusOut, err := gitStatusCmd(ctx, cfg.RepoRoot)
	if err != nil {
		cleanFalse := false
		tracker.receipt.Clean = &cleanFalse
		return fmt.Errorf("verifying git repository status in %s: %w", cfg.RepoRoot, err)
	}

	statusStr := strings.TrimSpace(string(gitStatusOut))
	if statusStr != "" {
		cleanFalse := false
		tracker.receipt.Clean = &cleanFalse
		return fmt.Errorf("repository %s is dirty: uncommitted changes detected:\n%s", cfg.RepoRoot, statusStr)
	}
	cleanTrue := true
	tracker.receipt.Clean = &cleanTrue

	headCommit, err := gitRevParseCmd(ctx, cfg.RepoRoot, "HEAD")
	if err != nil {
		return fmt.Errorf("resolving repository HEAD in %s: %w", cfg.RepoRoot, err)
	}
	headCommit = strings.ToLower(strings.TrimSpace(headCommit))
	if len(headCommit) != 40 {
		return fmt.Errorf("invalid HEAD commit %q in %s (expected 40-hex SHA)", headCommit, cfg.RepoRoot)
	}

	if cfg.GitCommit != "" {
		reqCommit := strings.ToLower(strings.TrimSpace(cfg.GitCommit))
		resolvedReq, err := gitRevParseCmd(ctx, cfg.RepoRoot, reqCommit+"^{commit}")
		if err != nil {
			if !strings.HasPrefix(headCommit, reqCommit) {
				return fmt.Errorf("requested commit %q cannot be resolved: %w", cfg.GitCommit, err)
			}
		} else {
			resolvedReq = strings.ToLower(strings.TrimSpace(resolvedReq))
			if resolvedReq != headCommit {
				return fmt.Errorf("requested commit %s does not match repository HEAD %s", resolvedReq, headCommit)
			}
		}
	}

	tracker.receipt.GitCommit = headCommit
	ref := cfg.GitRef
	if ref == "" {
		ref = "HEAD"
	}
	tracker.receipt.GitRef = ref

	hasher := sha256.New()
	if err := gitArchiveCmd(ctx, cfg.RepoRoot, headCommit, hasher); err != nil {
		return fmt.Errorf("constructing source archive for %s: %w", headCommit, err)
	}
	archiveSHA := hex.EncodeToString(hasher.Sum(nil))
	tracker.receipt.SourceArchiveSHA256 = archiveSHA

	if cfg.Toolchain != nil {
		normalizedArgs := []string{"go", "build", "-tags", "vulkan", "-o", filepath.ToSlash(cfg.OutBin), cfg.OutPkg}
		tracker.receipt.BuildArgs = normalizedArgs
		tracker.receipt.Toolchain = &ToolchainIdentity{
			CC:         filepath.ToSlash(cfg.Toolchain.CC),
			CXX:        filepath.ToSlash(cfg.Toolchain.CXX),
			AR:         filepath.ToSlash(cfg.Toolchain.AR),
			GLSLC:      filepath.ToSlash(cfg.Toolchain.GLSLC),
			VulkanSDK:  filepath.ToSlash(cfg.Toolchain.VulkanSDK),
			CxxRuntime: cfg.Toolchain.CxxRuntime,
			BuildArgs:  normalizedArgs,
		}
	}

	return nil
}

// RunVulkan orchestrates Vulkan build tasks according to cfg.Command.
func RunVulkan(ctx context.Context, cfg *VulkanConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil VulkanConfig")
	}
	if cfg.RepoRoot == "" {
		root, err := ResolveRepoRoot("")
		if err != nil {
			return err
		}
		cfg.RepoRoot = root
	}
	if cfg.PkgDir == "" {
		cfg.PkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	}
	if cfg.Toolchain == nil {
		tc, err := DiscoverToolchain()
		if err != nil {
			return fmt.Errorf("discovering toolchain: %w", err)
		}
		cfg.Toolchain = tc
	}
	if cfg.ReceiptPath == "" {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, DefaultVulkanBuildReceiptPath)
	} else if !filepath.IsAbs(cfg.ReceiptPath) {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, cfg.ReceiptPath)
	}
	if cfg.Command == "binary" && !cfg.SkipSmoke {
		cfg.Smoke = true
	}

	tracker := newReceiptTracker("vulkan", cfg.Command, cfg.ReceiptPath)
	cfg.Receipt = tracker.receipt
	defer func() {
		_ = tracker.finish(cfg.ReceiptPath)
	}()

	switch cfg.Command {
	case "shaders":
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		spvDir := filepath.Join(cfg.PkgDir, "spirv")
		if DirExists(spvDir) {
			if bundleSHA, err := ComputeSPIRVBundleSHA256(spvDir); err == nil {
				tracker.receipt.ShaderBundleSHA256 = bundleSHA
			}
		}
		return nil

	case "lib":
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		var artifact *BuildArtifact
		if err := tracker.recordPhase("shim", func() error {
			art, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			if err != nil {
				return err
			}
			artifact = art
			return nil
		}); err != nil {
			return err
		}
		tracker.receipt.Artifact = artifact
		return nil

	case "build":
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shim", func() error {
			_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", "vulkan", "./internal/compute/"}
		if err := tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		}); err != nil {
			return err
		}
		outBinPath := cfg.OutBin
		if outBinPath != "" && !filepath.IsAbs(outBinPath) {
			outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
		}
		if outBinPath != "" && FileExists(outBinPath) {
			if art, err := InspectArtifact(outBinPath); err == nil {
				tracker.receipt.Artifact = art
			}
			if cfg.Smoke {
				if err := tracker.recordSmoke(ctx, outBinPath); err != nil {
					return err
				}
			}
		}
		return nil

	case "binary":
		if cfg.OutPkg == "" || cfg.OutBin == "" {
			err := fmt.Errorf("usage: binary <pkg> <out>")
			tracker.receipt.Outcome = "failed"
			tracker.receipt.ExitCode = 1
			tracker.receipt.Error = err.Error()
			return err
		}
		if err := tracker.recordPhase("source_provenance", func() error {
			return checkSourceProvenance(ctx, cfg, tracker)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shim", func() error {
			_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", "vulkan", "-o", cfg.OutBin, cfg.OutPkg}
		if err := tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		}); err != nil {
			return err
		}
		outBinPath := cfg.OutBin
		if !filepath.IsAbs(outBinPath) {
			outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
		}
		if !FileExists(outBinPath) {
			err := fmt.Errorf("expected binary output %s was not produced", outBinPath)
			tracker.receipt.Outcome = "failed"
			tracker.receipt.ExitCode = 1
			tracker.receipt.Error = err.Error()
			return err
		}
		art, err := InspectArtifact(outBinPath)
		if err != nil {
			tracker.receipt.Outcome = "failed"
			tracker.receipt.ExitCode = 1
			tracker.receipt.Error = err.Error()
			return fmt.Errorf("inspecting output binary %s: %w", outBinPath, err)
		}
		spvDir := filepath.Join(cfg.PkgDir, "spirv")
		bundleSHA, err := ComputeSPIRVBundleSHA256(spvDir)
		if err != nil {
			tracker.receipt.Outcome = "failed"
			tracker.receipt.ExitCode = 1
			tracker.receipt.Error = err.Error()
			return fmt.Errorf("binding spirv bundle provenance: %w", err)
		}
		tracker.receipt.Artifact = art
		tracker.receipt.ShaderBundleSHA256 = bundleSHA

		if cfg.Smoke {
			if err := tracker.recordSmoke(ctx, outBinPath); err != nil {
				return err
			}
		}
		return nil

	case "test":
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shim", func() error {
			_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"test", "-tags", "vulkan", "-count=1", "-v", "-run", "Vulkan|HALDevice", "./internal/compute/", "./internal/model/"}
		return tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		})

	default:
		err := fmt.Errorf("unknown subcommand: %s (use shaders|lib|build|binary|test)", cfg.Command)
		tracker.receipt.Outcome = "failed"
		tracker.receipt.ExitCode = 1
		tracker.receipt.Error = err.Error()
		return err
	}
}
