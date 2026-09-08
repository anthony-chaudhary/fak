package computebuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
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

func strictFileSHA256(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", 0, err
	}
	if !info.Mode().IsRegular() {
		return "", 0, fmt.Errorf("not a regular file: %s", path)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), info.Size(), nil
}

func gitOutput(ctx context.Context, root string, args ...string) ([]byte, error) {
	argv := append([]string{"-C", root}, args...)
	cmd := exec.CommandContext(ctx, "git", argv...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s failed: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

func cleanGitState(ctx context.Context, root string) error {
	status, err := gitOutput(ctx, root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none")
	if err != nil {
		return err
	}
	if len(status) != 0 {
		return fmt.Errorf("vulkan source repository is dirty; refusing before compiler execution")
	}
	return nil
}

func hashGitArchive(ctx context.Context, root, commit string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", "-C", root, "archive", "--format=tar", commit)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	h := sha256.New()
	cmd.Stdout = h
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git archive failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func prepareVulkanSource(ctx context.Context, root, requestedCommit string) (BuildSourceProvenance, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return BuildSourceProvenance{}, fmt.Errorf("resolve repository root: %w", err)
	}
	commitRaw, err := gitOutput(ctx, root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	commit := strings.TrimSpace(string(commitRaw))
	if requestedCommit != "" {
		requestedRaw, err := gitOutput(ctx, root, "rev-parse", "--verify", requestedCommit+"^{commit}")
		if err != nil {
			return BuildSourceProvenance{}, fmt.Errorf("resolve requested Vulkan source commit: %w", err)
		}
		if resolved := strings.TrimSpace(string(requestedRaw)); resolved != commit {
			return BuildSourceProvenance{}, fmt.Errorf("requested Vulkan source commit %s does not match HEAD %s", resolved, commit)
		}
	}
	treeRaw, err := gitOutput(ctx, root, "rev-parse", "--verify", commit+"^{tree}")
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	tree := strings.TrimSpace(string(treeRaw))
	if err := cleanGitState(ctx, root); err != nil {
		return BuildSourceProvenance{}, err
	}
	archiveSHA, err := hashGitArchive(ctx, root, commit)
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	afterRaw, err := gitOutput(ctx, root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	if strings.TrimSpace(string(afterRaw)) != commit {
		return BuildSourceProvenance{}, fmt.Errorf("vulkan source commit changed during archive construction")
	}
	if err := cleanGitState(ctx, root); err != nil {
		return BuildSourceProvenance{}, err
	}
	return BuildSourceProvenance{
		GitCommit:           commit,
		GitTree:             tree,
		Clean:               true,
		SourceArchiveSHA256: archiveSHA,
	}, nil
}

func hashSPIRVBundle(repoRoot string) (string, int, error) {
	dir := filepath.Join(repoRoot, "internal", "compute", "spirv")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", 0, fmt.Errorf("read SPIR-V bundle: %w", err)
	}
	expected := make(map[string]struct{}, len(VulkanShaders))
	for _, name := range VulkanShaders {
		expected[name+".spv"] = struct{}{}
	}
	observed := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".spv" {
			continue
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return "", 0, fmt.Errorf("SPIR-V module is a symlink: %s", entry.Name())
		}
		if _, ok := expected[entry.Name()]; !ok {
			return "", 0, fmt.Errorf("unexpected SPIR-V module: %s", entry.Name())
		}
		observed = append(observed, entry.Name())
	}
	if len(observed) != len(expected) {
		return "", 0, fmt.Errorf("incomplete SPIR-V bundle: got %d modules, want %d", len(observed), len(expected))
	}
	sort.Strings(observed)
	h := sha256.New()
	for _, name := range observed {
		path := filepath.Join(dir, name)
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			return "", 0, fmt.Errorf("invalid SPIR-V module %s", name)
		}
		fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(filepath.Join("internal", "compute", "spirv", name)), info.Size())
		f, err := os.Open(path)
		if err != nil {
			return "", 0, err
		}
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			return "", 0, copyErr
		}
		if closeErr != nil {
			return "", 0, closeErr
		}
	}
	return hex.EncodeToString(h.Sum(nil)), len(observed), nil
}

// ComputeSPIRVBundleSHA256 retains the v1 exported helper while applying the
// same deterministic filename/size/content framing and fail-closed file rules.
func ComputeSPIRVBundleSHA256(spvDir string) (string, error) {
	entries, err := os.ReadDir(spvDir)
	if err != nil {
		return "", fmt.Errorf("read SPIR-V directory: %w", err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) == ".spv" {
			names = append(names, entry.Name())
		}
	}
	if len(names) == 0 {
		return "", fmt.Errorf("no .spv files found in %s", spvDir)
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		path := filepath.Join(spvDir, name)
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() {
			return "", fmt.Errorf("SPIR-V module must be a regular file: %s", name)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read SPIR-V module %s: %w", name, err)
		}
		fmt.Fprintf(h, "%s\x00%d\x00", name, len(data))
		_, _ = h.Write(data)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// CompareReceiptProvenance retains the v1 API. Vulkan v2 comparisons use the
// stable identity produced from source, tool, command, shader, and binary hashes.
func CompareReceiptProvenance(a, b *ComputeBuildReceipt) error {
	if a == nil || b == nil {
		return fmt.Errorf("receipt cannot be nil")
	}
	if a.Vulkan != nil || b.Vulkan != nil {
		if a.Vulkan == nil || b.Vulkan == nil || a.Artifact == nil || b.Artifact == nil {
			return fmt.Errorf("Vulkan provenance presence mismatch")
		}
		if a.Vulkan.StableIdentitySHA256 != b.Vulkan.StableIdentitySHA256 || a.Artifact.SHA256 != b.Artifact.SHA256 {
			return fmt.Errorf("Vulkan stable provenance mismatch")
		}
		return nil
	}
	left, err := hashJSON([]any{a.Schema, a.Backend, a.Command, a.Outcome, a.ExitCode, a.GitCommit, a.GitRef, a.Clean, a.SourceArchiveSHA256, a.ShaderBundleSHA256, a.BuildArgs, a.Toolchain, a.Artifact})
	if err != nil {
		return err
	}
	right, err := hashJSON([]any{b.Schema, b.Backend, b.Command, b.Outcome, b.ExitCode, b.GitCommit, b.GitRef, b.Clean, b.SourceArchiveSHA256, b.ShaderBundleSHA256, b.BuildArgs, b.Toolchain, b.Artifact})
	if err != nil {
		return err
	}
	if left != right {
		return fmt.Errorf("receipt provenance mismatch")
	}
	return nil
}

func resolveBuildTool(name string) (string, error) {
	if strings.TrimSpace(name) == "" {
		return "", fmt.Errorf("required build tool is not configured")
	}
	resolved := name
	if !filepath.IsAbs(resolved) {
		var err error
		resolved, err = exec.LookPath(resolved)
		if err != nil {
			return "", err
		}
	}
	if real, err := filepath.EvalSymlinks(resolved); err == nil {
		resolved = real
	}
	return resolved, nil
}

func hashJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

func vulkanToolchainIdentity(tc *Toolchain) ([]BuildToolIdentity, string, error) {
	goTool := tc.Go
	if goTool == "" {
		goTool = "go"
	}
	tools := []struct {
		role string
		path string
	}{
		{"go", goTool},
		{"cc", tc.CC},
		{"cxx", tc.CXX},
		{"ar", tc.AR},
		{"glslc", tc.GLSLC},
	}
	identities := make([]BuildToolIdentity, 0, len(tools))
	for _, tool := range tools {
		path, err := resolveBuildTool(tool.path)
		if err != nil {
			return nil, "", fmt.Errorf("resolve %s tool: %w", tool.role, err)
		}
		digest, _, err := strictFileSHA256(path)
		if err != nil {
			return nil, "", fmt.Errorf("hash %s tool: %w", tool.role, err)
		}
		identities = append(identities, BuildToolIdentity{Role: tool.role, Executable: filepath.Base(path), SHA256: digest})
	}
	digest, err := hashJSON(struct {
		Tools      []BuildToolIdentity `json:"tools"`
		CXXRuntime string              `json:"cxx_runtime"`
		IsWindows  bool                `json:"is_windows"`
	}{identities, tc.CxxRuntime, tc.IsWindows})
	return identities, digest, err
}

func normalizedVulkanBuildCommand(cfg *VulkanConfig) ([]string, string, error) {
	relPkg, err := filepath.Rel(cfg.RepoRoot, cfg.PkgDir)
	if err != nil || relPkg == ".." || strings.HasPrefix(relPkg, ".."+string(filepath.Separator)) {
		return nil, "", fmt.Errorf("Vulkan package directory must be inside repository root")
	}
	cxxFlags := "-O3,-std=c++17,-fPIC"
	if cfg.Toolchain.IsWindows {
		cxxFlags = "-O3,-std=c++17,-D_CRT_SECURE_NO_WARNINGS"
	}
	commands := []string{
		"glslc|-O|--target-env=vulkan1.2|-fshader-stage=comp|$REPO/internal/compute/shaders/{registered}.comp|-o|$REPO/internal/compute/spirv/{registered}.spv",
		"cxx|" + cxxFlags + "|-c|$REPO/internal/compute/vulkan_shim.cpp|-o|$REPO/internal/compute/vulkan_shim.o",
		"ar|rcs|$REPO/internal/compute/libfakvulkan.a|$REPO/internal/compute/vulkan_shim.o",
		"go|build|-tags|vulkan|-o|$OUT|" + filepath.ToSlash(filepath.Clean(cfg.OutPkg)),
		"pkg_dir=$REPO/" + filepath.ToSlash(relPkg),
		fmt.Sprintf("smoke=%t", cfg.Smoke),
	}
	digest, err := hashJSON(commands)
	return commands, digest, err
}

func compareVulkanBuildReceipt(path string, artifact *BuildArtifact, provenance *VulkanBuildProvenance) (*BuildReproducibility, error) {
	if path == "" {
		return &BuildReproducibility{Status: "baseline"}, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return &BuildReproducibility{Status: "invalid"}, fmt.Errorf("read comparison receipt: %w", err)
	}
	digest := sha256.Sum256(b)
	rep := &BuildReproducibility{Status: "mismatch", ComparedReceiptSHA256: hex.EncodeToString(digest[:])}
	var prior ComputeBuildReceipt
	if err := json.Unmarshal(b, &prior); err != nil {
		rep.Status = "invalid"
		return rep, fmt.Errorf("decode comparison receipt: %w", err)
	}
	if prior.Schema != VulkanBuildReceiptSchema || prior.Outcome != "success" || prior.Vulkan == nil || prior.Artifact == nil || prior.Artifact.SHA256 == "" {
		rep.Status = "invalid"
		return rep, fmt.Errorf("comparison receipt is not a complete successful %s receipt", VulkanBuildReceiptSchema)
	}
	checks := []struct {
		name string
		a    string
		b    string
	}{
		{"source.git_commit", provenance.Source.GitCommit, prior.Vulkan.Source.GitCommit},
		{"source.git_tree", provenance.Source.GitTree, prior.Vulkan.Source.GitTree},
		{"source.archive", provenance.Source.SourceArchiveSHA256, prior.Vulkan.Source.SourceArchiveSHA256},
		{"spirv_bundle", provenance.SPIRVBundleSHA256, prior.Vulkan.SPIRVBundleSHA256},
		{"toolchain", provenance.ToolchainSHA256, prior.Vulkan.ToolchainSHA256},
		{"build_command", provenance.BuildCommandSHA256, prior.Vulkan.BuildCommandSHA256},
		{"binary", artifact.SHA256, prior.Artifact.SHA256},
		{"stable_identity", provenance.StableIdentitySHA256, prior.Vulkan.StableIdentitySHA256},
	}
	for _, check := range checks {
		if check.a != check.b {
			rep.MismatchedFields = append(rep.MismatchedFields, check.name)
		}
	}
	if len(rep.MismatchedFields) != 0 {
		return rep, fmt.Errorf("Vulkan build reproducibility mismatch: %s", strings.Join(rep.MismatchedFields, ", "))
	}
	rep.Status = "match"
	return rep, nil
}

func mergeToolchainOverrides(base, overrides *Toolchain) *Toolchain {
	if base == nil {
		base = &Toolchain{}
	}
	merged := *base
	if overrides == nil {
		return &merged
	}
	for dst, src := range map[*string]string{
		&merged.Go: overrides.Go, &merged.CC: overrides.CC, &merged.CXX: overrides.CXX,
		&merged.AR: overrides.AR, &merged.GLSLC: overrides.GLSLC,
		&merged.VulkanSDK: overrides.VulkanSDK, &merged.VulkanInc: overrides.VulkanInc,
		&merged.VulkanLib: overrides.VulkanLib,
	} {
		if src != "" {
			*dst = src
		}
	}
	return &merged
}

func finalizeVulkanBinary(cfg *VulkanConfig, source BuildSourceProvenance, outBinPath string, expectedTools []BuildToolIdentity, expectedToolchainSHA string) (*BuildArtifact, *VulkanBuildProvenance, error) {
	binarySHA, binarySize, err := strictFileSHA256(outBinPath)
	if err != nil {
		return nil, nil, fmt.Errorf("hash Vulkan output binary: %w", err)
	}
	spirvSHA, spirvCount, err := hashSPIRVBundle(cfg.RepoRoot)
	if err != nil {
		return nil, nil, err
	}
	_, toolchainSHA, err := vulkanToolchainIdentity(cfg.Toolchain)
	if err != nil {
		return nil, nil, err
	}
	if toolchainSHA != expectedToolchainSHA {
		return nil, nil, fmt.Errorf("Vulkan toolchain changed during build")
	}
	commands, commandSHA, err := normalizedVulkanBuildCommand(cfg)
	if err != nil {
		return nil, nil, err
	}
	artifact := &BuildArtifact{Path: outBinPath, SizeBytes: binarySize, SHA256: binarySHA}
	provenance := &VulkanBuildProvenance{
		Source:                 source,
		SPIRVBundleSHA256:      spirvSHA,
		SPIRVModuleCount:       spirvCount,
		Toolchain:              expectedTools,
		ToolchainSHA256:        toolchainSHA,
		NormalizedBuildCommand: commands,
		BuildCommandSHA256:     commandSHA,
	}
	stableSHA, err := hashJSON(struct {
		Source             BuildSourceProvenance `json:"source"`
		SPIRVBundleSHA256  string                `json:"spirv_bundle_sha256"`
		SPIRVModuleCount   int                   `json:"spirv_module_count"`
		ToolchainSHA256    string                `json:"toolchain_sha256"`
		BuildCommandSHA256 string                `json:"build_command_sha256"`
		BinarySHA256       string                `json:"binary_sha256"`
	}{source, spirvSHA, spirvCount, toolchainSHA, commandSHA, binarySHA})
	if err != nil {
		return nil, nil, err
	}
	provenance.StableIdentitySHA256 = stableSHA
	return artifact, provenance, nil
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

// RunVulkan orchestrates Vulkan build tasks according to cfg.Command.
func RunVulkan(ctx context.Context, cfg *VulkanConfig) (retErr error) {
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
	if cfg.ReceiptPath == "" {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, DefaultVulkanBuildReceiptPath)
	} else if !filepath.IsAbs(cfg.ReceiptPath) {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, cfg.ReceiptPath)
	}
	if cfg.CompareReceiptPath != "" && !filepath.IsAbs(cfg.CompareReceiptPath) {
		cfg.CompareReceiptPath = filepath.Join(cfg.RepoRoot, cfg.CompareReceiptPath)
	}
	if cfg.Command == "binary" && !cfg.SkipSmoke {
		cfg.Smoke = true
	}

	tracker := newReceiptTracker("vulkan", cfg.Command, cfg.ReceiptPath)
	if cfg.Command == "binary" {
		tracker.receipt.Schema = VulkanBuildReceiptSchema
	}
	cfg.Receipt = tracker.receipt
	defer func() {
		if err := tracker.finish(cfg.ReceiptPath); err != nil && retErr == nil {
			retErr = fmt.Errorf("write Vulkan build receipt: %w", err)
		}
	}()
	var source BuildSourceProvenance
	if cfg.Command == "binary" {
		if cfg.OutPkg == "" || cfg.OutBin == "" {
			err := fmt.Errorf("usage: binary <pkg> <out>")
			tracker.fail(err, 1)
			return err
		}
		if err := tracker.recordPhase("source_preflight", func() error {
			var err error
			source, err = prepareVulkanSource(ctx, cfg.RepoRoot, cfg.GitCommit)
			return err
		}); err != nil {
			return err
		}
	}
	if cfg.Toolchain == nil {
		if err := tracker.recordPhase("toolchain_discovery", func() error {
			tc, err := DiscoverToolchain()
			if err != nil {
				return fmt.Errorf("discovering toolchain: %w", err)
			}
			cfg.Toolchain = tc
			return nil
		}); err != nil {
			return err
		}
	}
	if cfg.ToolchainOverrides != nil {
		cfg.Toolchain = mergeToolchainOverrides(cfg.Toolchain, cfg.ToolchainOverrides)
	}
	var toolIdentities []BuildToolIdentity
	var toolchainSHA string
	if cfg.Command == "binary" {
		if err := tracker.recordPhase("toolchain_identity", func() error {
			var err error
			toolIdentities, toolchainSHA, err = vulkanToolchainIdentity(cfg.Toolchain)
			return err
		}); err != nil {
			return err
		}
	}

	switch cfg.Command {
	case "shaders":
		return tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		})

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
		goTool := cfg.Toolchain.Go
		if goTool == "" {
			goTool = "go"
		}
		args := []string{"build", "-tags", "vulkan", "-o", cfg.OutBin, cfg.OutPkg}
		if err := tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, goTool, args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		}); err != nil {
			return err
		}
		outBinPath := cfg.OutBin
		if !filepath.IsAbs(outBinPath) {
			outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
		}
		if err := tracker.recordPhase("source_revalidation", func() error {
			revalidated, err := prepareVulkanSource(ctx, cfg.RepoRoot, cfg.GitCommit)
			if err != nil {
				return err
			}
			if revalidated != source {
				return fmt.Errorf("Vulkan source identity changed during build")
			}
			return nil
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("provenance", func() error {
			artifact, provenance, err := finalizeVulkanBinary(cfg, source, outBinPath, toolIdentities, toolchainSHA)
			if err != nil {
				return err
			}
			tracker.receipt.Artifact = artifact
			tracker.receipt.Vulkan = provenance
			return nil
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("reproducibility", func() error {
			rep, err := compareVulkanBuildReceipt(cfg.CompareReceiptPath, tracker.receipt.Artifact, tracker.receipt.Vulkan)
			tracker.receipt.Reproducibility = rep
			return err
		}); err != nil {
			return err
		}
		if cfg.Smoke && FileExists(outBinPath) {
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
		tracker.fail(err, 1)
		return err
	}
}
