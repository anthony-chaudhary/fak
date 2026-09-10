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
	"runtime"
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
	"tree_attention",
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
	"q6k_matmul",
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

const vulkanV2ModuleCount = 43

var unavailableVulkanV2Causality = []string{
	"historical_source_cleanliness",
	"actual_build_argv",
	"actual_build_environment",
	"actual_build_tool_causality",
	"sdk_header_library_vsdevenv_inputs",
	"smoke_execution",
	"build_timing",
}

// VulkanReceiptToolInputs are policy-pinned executable bytes used to recompute
// a receipt's tool identities. Verification hashes but never executes them.
type VulkanReceiptToolInputs struct {
	Go         string
	CC         string
	CXX        string
	AR         string
	GLSLC      string
	CxxRuntime string
	IsWindows  bool
}

// VulkanReceiptBuildPlan is the independently trusted build plan whose
// normalized identity must match the receipt. V2 verification intentionally
// supports only the cmd/fak binary needed by the physical benchmark boundary.
type VulkanReceiptBuildPlan struct {
	PackageDir string
	OutPackage string
	Smoke      bool
}

// VulkanBinaryReceiptEvidence contains independently trusted observations. No
// field is inferred from the receipt under verification. ExpectedBinarySHA256
// and ExpectedBinarySize must come from a sealed mapped-executable observer.
type VulkanBinaryReceiptEvidence struct {
	ReceiptPath          string
	SourceRoot           string
	GitExecutable        string
	ExpectedGitSHA256    string
	ExpectedCommit       string
	BinaryPath           string
	ExpectedBinarySHA256 string
	ExpectedBinarySize   int64
	SPIRVRoot            string
	ExpectedSPIRVModules []string
	Tools                VulkanReceiptToolInputs
	BuildPlan            VulkanReceiptBuildPlan
	Comparison           *VulkanBinaryReceiptEvidence
}

// VulkanBinaryReceiptIdentityVerification is a point-in-time identity proof.
// It does not upgrade Vulkan v2 into evidence of historical build causality;
// those claims remain explicitly unavailable without a rebuild or attestation.
type VulkanBinaryReceiptIdentityVerification struct {
	Receipt                  ComputeBuildReceipt
	ReceiptSHA256            string
	HistoricalBuildCausality string
	UnavailableClaims        []string
}

type observedVulkanReceiptIdentity struct {
	ReceiptPath            string
	Source                 BuildSourceProvenance
	Artifact               BuildArtifact
	SPIRVBundleSHA256      string
	SPIRVModuleCount       int
	Toolchain              []BuildToolIdentity
	ToolchainSHA256        string
	NormalizedBuildCommand []string
	BuildCommandSHA256     string
	StableIdentitySHA256   string
	GitExecutableSHA256    string
}

// VerifyVulkanBinaryReceiptIdentity verifies every identity claim that Vulkan
// v2 can re-observe from independent local evidence. Receipt JSON is read once,
// unknown or trailing data is rejected, and all mutable evidence is observed
// before and after comparison. The function performs no discovery, compiler,
// model, hardware, or network action.
func VerifyVulkanBinaryReceiptIdentity(ctx context.Context, evidence VulkanBinaryReceiptEvidence) (*VulkanBinaryReceiptIdentityVerification, error) {
	if ctx == nil {
		return nil, fmt.Errorf("verify Vulkan receipt identity: nil context")
	}
	if runtime.GOOS != "linux" {
		return nil, fmt.Errorf("verify Vulkan receipt identity: fd-bound Git execution is unsupported on %s", runtime.GOOS)
	}
	gitRun, closeGit, err := pinnedLinuxGitRunner(evidence.GitExecutable, evidence.ExpectedGitSHA256)
	if err != nil {
		return nil, err
	}
	defer closeGit()
	var comparisonGitRun vulkanGitRunner
	var closeComparisonGit func() error
	if evidence.Comparison != nil {
		comparisonGitRun, closeComparisonGit, err = pinnedLinuxGitRunner(evidence.Comparison.GitExecutable, evidence.Comparison.ExpectedGitSHA256)
		if err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		defer closeComparisonGit()
	}
	return verifyVulkanBinaryReceiptIdentityWithRunners(ctx, evidence, gitRun, comparisonGitRun)
}

type vulkanGitRunner func(context.Context, string, ...string) ([]byte, error)

func verifyVulkanBinaryReceiptIdentityWithRunners(ctx context.Context, evidence VulkanBinaryReceiptEvidence, gitRun, comparisonGitRun vulkanGitRunner) (*VulkanBinaryReceiptIdentityVerification, error) {
	if evidence.Comparison != nil && evidence.Comparison.Comparison != nil {
		return nil, fmt.Errorf("verify Vulkan receipt identity: nested comparison evidence is not supported")
	}
	if gitRun == nil || (evidence.Comparison != nil && comparisonGitRun == nil) {
		return nil, fmt.Errorf("verify Vulkan receipt identity: pinned Git runner is required")
	}
	raw, err := readStableRegularFile(evidence.ReceiptPath, "Vulkan receipt")
	if err != nil {
		return nil, err
	}
	var comparisonRaw []byte
	if evidence.Comparison != nil {
		comparisonRaw, err = readStableRegularFile(evidence.Comparison.ReceiptPath, "comparison Vulkan receipt")
		if err != nil {
			return nil, err
		}
	}

	before, err := observeVulkanReceiptEvidence(ctx, evidence, gitRun)
	if err != nil {
		return nil, err
	}
	receipt, err := decodeStrictVulkanReceipt(raw)
	if err != nil {
		return nil, err
	}

	var comparisonReceipt *ComputeBuildReceipt
	var comparisonObservation *observedVulkanReceiptIdentity
	if evidence.Comparison != nil {
		observed, err := observeVulkanReceiptEvidence(ctx, *evidence.Comparison, comparisonGitRun)
		if err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		decoded, err := decodeStrictVulkanReceipt(comparisonRaw)
		if err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		if err := compareReceiptToObservation(decoded, observed); err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		if err := verifyBaselineReproducibility(decoded); err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		comparisonReceipt = &decoded
		comparisonObservation = &observed
	}

	if err := compareReceiptToObservation(receipt, before); err != nil {
		return nil, err
	}
	if evidence.Comparison == nil {
		if err := verifyBaselineReproducibility(receipt); err != nil {
			return nil, err
		}
	} else if err := verifyMatchedReproducibility(receipt, comparisonRaw, *comparisonReceipt, before, *comparisonObservation); err != nil {
		return nil, err
	}

	after, err := observeVulkanReceiptEvidence(ctx, evidence, gitRun)
	if err != nil {
		return nil, fmt.Errorf("revalidate Vulkan receipt evidence: %w", err)
	}
	if !sameObservedVulkanIdentity(before, after) {
		return nil, fmt.Errorf("Vulkan receipt evidence changed during verification")
	}
	if evidence.Comparison != nil {
		comparisonAfter, err := observeVulkanReceiptEvidence(ctx, *evidence.Comparison, comparisonGitRun)
		if err != nil {
			return nil, fmt.Errorf("revalidate comparison Vulkan receipt evidence: %w", err)
		}
		if !sameObservedVulkanIdentity(*comparisonObservation, comparisonAfter) {
			return nil, fmt.Errorf("comparison Vulkan receipt evidence changed during verification")
		}
	}
	digest := sha256.Sum256(raw)
	return &VulkanBinaryReceiptIdentityVerification{
		Receipt:                  receipt,
		ReceiptSHA256:            hex.EncodeToString(digest[:]),
		HistoricalBuildCausality: "unavailable",
		UnavailableClaims:        append([]string(nil), unavailableVulkanV2Causality...),
	}, nil
}

func decodeStrictVulkanReceipt(raw []byte) (ComputeBuildReceipt, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var receipt ComputeBuildReceipt
	if err := dec.Decode(&receipt); err != nil {
		return ComputeBuildReceipt{}, fmt.Errorf("decode Vulkan receipt: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return ComputeBuildReceipt{}, fmt.Errorf("decode Vulkan receipt: trailing JSON value")
		}
		return ComputeBuildReceipt{}, fmt.Errorf("decode Vulkan receipt trailing data: %w", err)
	}
	return receipt, nil
}

func readStableRegularFile(path, label string) ([]byte, error) {
	clean, before, err := strictAbsoluteRegularFile(path, label)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(clean)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", label, err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return nil, fmt.Errorf("%s changed before read", label)
	}
	first, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("rewind %s: %w", label, err)
	}
	second, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("reread %s: %w", label, err)
	}
	afterHandle, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("restat %s: %w", label, err)
	}
	afterPath, err := os.Lstat(clean)
	if err != nil || afterPath.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, afterPath) || !sameFileMetadata(opened, afterHandle) || !bytes.Equal(first, second) {
		return nil, fmt.Errorf("%s changed during read", label)
	}
	return first, nil
}

func stableRegularFileSHA256(path, label string) (string, int64, error) {
	clean, before, err := strictAbsoluteRegularFile(path, label)
	if err != nil {
		return "", 0, err
	}
	f, err := os.Open(clean)
	if err != nil {
		return "", 0, fmt.Errorf("open %s: %w", label, err)
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return "", 0, fmt.Errorf("%s changed before hashing", label)
	}
	h1 := sha256.New()
	if _, err := io.Copy(h1, f); err != nil {
		return "", 0, fmt.Errorf("hash %s: %w", label, err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", 0, fmt.Errorf("rewind %s: %w", label, err)
	}
	h2 := sha256.New()
	if _, err := io.Copy(h2, f); err != nil {
		return "", 0, fmt.Errorf("rehash %s: %w", label, err)
	}
	afterHandle, err := f.Stat()
	if err != nil {
		return "", 0, fmt.Errorf("restat %s: %w", label, err)
	}
	afterPath, err := os.Lstat(clean)
	if err != nil || afterPath.Mode()&os.ModeSymlink != 0 || !os.SameFile(opened, afterPath) || !sameFileMetadata(opened, afterHandle) || !bytes.Equal(h1.Sum(nil), h2.Sum(nil)) {
		return "", 0, fmt.Errorf("%s changed during hashing", label)
	}
	return hex.EncodeToString(h1.Sum(nil)), opened.Size(), nil
}

func strictAbsoluteRegularFile(path, label string) (string, os.FileInfo, error) {
	if !filepath.IsAbs(path) {
		return "", nil, fmt.Errorf("%s path must be absolute", label)
	}
	clean := filepath.Clean(path)
	info, err := os.Lstat(clean)
	if err != nil {
		return "", nil, fmt.Errorf("stat %s: %w", label, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return "", nil, fmt.Errorf("%s must be a non-symlink regular file", label)
	}
	resolved, err := filepath.EvalSymlinks(clean)
	if err != nil || !samePath(clean, resolved) {
		return "", nil, fmt.Errorf("%s path must not traverse symlinks", label)
	}
	return clean, info, nil
}

func strictAbsoluteDirectory(path, label string) (string, error) {
	if !filepath.IsAbs(path) {
		return "", fmt.Errorf("%s path must be absolute", label)
	}
	clean := filepath.Clean(path)
	info, err := os.Lstat(clean)
	if err != nil {
		return "", fmt.Errorf("stat %s: %w", label, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", fmt.Errorf("%s must be a non-symlink directory", label)
	}
	resolved, err := filepath.EvalSymlinks(clean)
	if err != nil || !samePath(clean, resolved) {
		return "", fmt.Errorf("%s path must not traverse symlinks", label)
	}
	return clean, nil
}

func samePath(a, b string) bool {
	a = filepath.Clean(a)
	b = filepath.Clean(b)
	if os.PathSeparator == '\\' {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func sameFileMetadata(a, b os.FileInfo) bool {
	return os.SameFile(a, b) && a.Size() == b.Size() && a.Mode() == b.Mode() && a.ModTime().Equal(b.ModTime())
}

// pinnedLinuxGitRunner binds every Git invocation to the exact executable
// bytes approved by policy. O_NOFOLLOW closes the final-component symlink
// race, SameFile binds the opened descriptor to the path observation, and
// /proc/self/fd/3 ensures exec never resolves the mutable pathname again.
// There is deliberately no pathname or non-Linux fallback.
func pinnedLinuxGitRunner(path, expectedSHA256 string) (vulkanGitRunner, func() error, error) {
	if runtime.GOOS != "linux" {
		return nil, nil, fmt.Errorf("pin Git executable: fd-bound execution is unsupported on %s", runtime.GOOS)
	}
	if !validLowerSHA256(expectedSHA256) {
		return nil, nil, fmt.Errorf("pin Git executable: policy SHA256 must be a lowercase digest")
	}
	clean, before, err := strictAbsoluteRegularFile(path, "Git executable")
	if err != nil {
		return nil, nil, err
	}
	if before.Mode().Perm()&0111 == 0 {
		return nil, nil, fmt.Errorf("Git executable is not executable")
	}
	// Linux O_NOFOLLOW. This package is cross-platform, so keep the numeric
	// constant local and guard its use with the GOOS check above.
	const linuxONoFollow = 0x20000
	f, err := os.OpenFile(clean, os.O_RDONLY|linuxONoFollow, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("open pinned Git executable: %w", err)
	}
	failure := func(err error) (vulkanGitRunner, func() error, error) {
		_ = f.Close()
		return nil, nil, err
	}
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) || !opened.Mode().IsRegular() {
		return failure(fmt.Errorf("Git executable changed before pinning"))
	}
	digest, err := stableOpenFileSHA256(f, opened, "Git executable")
	if err != nil {
		return failure(err)
	}
	if digest != expectedSHA256 {
		return failure(fmt.Errorf("policy-pinned Git executable identity mismatch"))
	}
	if info, err := os.Stat("/proc/self/fd"); err != nil || !info.IsDir() {
		return failure(fmt.Errorf("pin Git executable: /proc/self/fd is unavailable"))
	}
	run := func(ctx context.Context, root string, args ...string) ([]byte, error) {
		if ctx == nil {
			return nil, fmt.Errorf("run pinned Git executable: nil context")
		}
		current, err := stableOpenFileSHA256(f, opened, "Git executable")
		if err != nil || current != expectedSHA256 {
			return nil, fmt.Errorf("pinned Git executable changed before execution")
		}
		argv := controlledVulkanGitArgs(root, args...)
		cmd := exec.CommandContext(ctx, "/proc/self/fd/3", argv...)
		cmd.ExtraFiles = []*os.File{f}
		cmd.Env = controlledVulkanGitEnv()
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, runErr := cmd.Output()
		current, hashErr := stableOpenFileSHA256(f, opened, "Git executable")
		if hashErr != nil || current != expectedSHA256 {
			return nil, fmt.Errorf("pinned Git executable changed during execution")
		}
		if runErr != nil {
			return nil, fmt.Errorf("pinned Git failed: %w: %s", runErr, strings.TrimSpace(stderr.String()))
		}
		return out, nil
	}
	return run, f.Close, nil
}

func controlledVulkanGitArgs(root string, args ...string) []string {
	argv := []string{
		"-C", root,
		"-c", "core.attributesFile=/dev/null",
		"-c", "core.autocrlf=false",
		"-c", "core.excludesFile=/dev/null",
		"-c", "core.fileMode=true",
		"-c", "core.fsmonitor=false",
		"-c", "core.hooksPath=/dev/null",
		"-c", "core.ignoreCase=false",
		"-c", "core.symlinks=true",
		"-c", "protocol.allow=never",
		"-c", "protocol.ext.allow=never",
		"-c", "protocol.file.allow=never",
	}
	return append(argv, args...)
}

func controlledVulkanGitEnv() []string {
	return []string{
		"GIT_ALLOW_PROTOCOL=",
		"GIT_ASKPASS=/bin/false",
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_NO_LAZY_FETCH=1",
		"GIT_NO_REPLACE_OBJECTS=1",
		"GIT_OPTIONAL_LOCKS=0",
		"GIT_TERMINAL_PROMPT=0",
		"HOME=/nonexistent",
		"LANG=C",
		"LC_ALL=C",
		"PATH=/usr/bin:/bin",
		"SSH_ASKPASS=/bin/false",
	}
}

func stableOpenFileSHA256(f *os.File, pinned os.FileInfo, label string) (string, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", fmt.Errorf("rewind %s: %w", label, err)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hash %s: %w", label, err)
	}
	after, err := f.Stat()
	if err != nil || !sameFileMetadata(pinned, after) {
		return "", fmt.Errorf("%s changed while pinned", label)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func observeVulkanReceiptEvidence(ctx context.Context, evidence VulkanBinaryReceiptEvidence, gitRun vulkanGitRunner) (observedVulkanReceiptIdentity, error) {
	if evidence.Comparison != nil && evidence.Comparison.ReceiptPath == evidence.ReceiptPath {
		return observedVulkanReceiptIdentity{}, fmt.Errorf("comparison receipt must be distinct")
	}
	gitSHA, _, err := stableRegularFileSHA256(evidence.GitExecutable, "Git executable")
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	if !validLowerSHA256(evidence.ExpectedGitSHA256) || gitSHA != evidence.ExpectedGitSHA256 {
		return observedVulkanReceiptIdentity{}, fmt.Errorf("policy-pinned Git executable identity mismatch")
	}
	source, err := prepareVulkanSourceWithGit(ctx, gitRun, evidence.SourceRoot, evidence.ExpectedCommit)
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	artifactSHA, artifactSize, err := stableRegularFileSHA256(evidence.BinaryPath, "sealed mapped executable")
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	if !validLowerSHA256(evidence.ExpectedBinarySHA256) || artifactSHA != evidence.ExpectedBinarySHA256 || artifactSize != evidence.ExpectedBinarySize || artifactSize <= 0 {
		return observedVulkanReceiptIdentity{}, fmt.Errorf("sealed mapped executable identity mismatch")
	}
	spirvSHA, spirvCount, err := hashObservedSPIRVBundle(evidence.SPIRVRoot, evidence.ExpectedSPIRVModules)
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	toolchain, toolchainSHA, err := strictVulkanToolchainIdentity(evidence.Tools)
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	if evidence.BuildPlan.OutPackage != "./cmd/fak" {
		return observedVulkanReceiptIdentity{}, fmt.Errorf("trusted Vulkan build plan must target ./cmd/fak")
	}
	repoRoot, err := strictAbsoluteDirectory(evidence.SourceRoot, "Vulkan source root")
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	expectedPkg := filepath.Join(repoRoot, "internal", "compute")
	if !samePath(evidence.BuildPlan.PackageDir, expectedPkg) {
		return observedVulkanReceiptIdentity{}, fmt.Errorf("trusted Vulkan package directory must be source-root/internal/compute")
	}
	tc := evidence.Tools.toolchain()
	commands, commandSHA, err := normalizedVulkanBuildCommand(&VulkanConfig{
		RepoRoot:  repoRoot,
		PkgDir:    evidence.BuildPlan.PackageDir,
		OutPkg:    evidence.BuildPlan.OutPackage,
		OutBin:    evidence.BinaryPath,
		Smoke:     evidence.BuildPlan.Smoke,
		Toolchain: tc,
	})
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	stableSHA, err := vulkanStableIdentitySHA(source, spirvSHA, spirvCount, toolchainSHA, commandSHA, artifactSHA)
	if err != nil {
		return observedVulkanReceiptIdentity{}, err
	}
	gitAfterSHA, _, err := stableRegularFileSHA256(evidence.GitExecutable, "Git executable")
	if err != nil || gitAfterSHA != gitSHA {
		return observedVulkanReceiptIdentity{}, fmt.Errorf("Git executable changed during verification")
	}
	return observedVulkanReceiptIdentity{
		ReceiptPath:       filepath.Clean(evidence.ReceiptPath),
		Source:            source,
		Artifact:          BuildArtifact{Path: filepath.Clean(evidence.BinaryPath), SizeBytes: artifactSize, SHA256: artifactSHA},
		SPIRVBundleSHA256: spirvSHA, SPIRVModuleCount: spirvCount,
		Toolchain: toolchain, ToolchainSHA256: toolchainSHA,
		NormalizedBuildCommand: commands, BuildCommandSHA256: commandSHA,
		StableIdentitySHA256: stableSHA, GitExecutableSHA256: gitSHA,
	}, nil
}

func (in VulkanReceiptToolInputs) toolchain() *Toolchain {
	return &Toolchain{Go: in.Go, CC: in.CC, CXX: in.CXX, AR: in.AR, GLSLC: in.GLSLC, CxxRuntime: in.CxxRuntime, IsWindows: in.IsWindows}
}

func strictVulkanToolchainIdentity(in VulkanReceiptToolInputs) ([]BuildToolIdentity, string, error) {
	tools := []struct{ role, path string }{{"go", in.Go}, {"cc", in.CC}, {"cxx", in.CXX}, {"ar", in.AR}, {"glslc", in.GLSLC}}
	identities := make([]BuildToolIdentity, 0, len(tools))
	for _, tool := range tools {
		digest, _, err := stableRegularFileSHA256(tool.path, tool.role+" tool")
		if err != nil {
			return nil, "", err
		}
		identities = append(identities, BuildToolIdentity{Role: tool.role, Executable: filepath.Base(tool.path), SHA256: digest})
	}
	digest, err := hashJSON(struct {
		Tools      []BuildToolIdentity `json:"tools"`
		CXXRuntime string              `json:"cxx_runtime"`
		IsWindows  bool                `json:"is_windows"`
	}{identities, in.CxxRuntime, in.IsWindows})
	return identities, digest, err
}

func hashObservedSPIRVBundle(root string, expectedStems []string) (string, int, error) {
	root, err := strictAbsoluteDirectory(root, "runtime SPIR-V root")
	if err != nil {
		return "", 0, err
	}
	if len(expectedStems) != vulkanV2ModuleCount {
		return "", 0, fmt.Errorf("expected SPIR-V registry must contain exactly %d modules", vulkanV2ModuleCount)
	}
	expected := make(map[string]struct{}, len(expectedStems))
	for _, stem := range expectedStems {
		if stem == "" || filepath.Base(stem) != stem || filepath.Ext(stem) != "" {
			return "", 0, fmt.Errorf("invalid expected SPIR-V module stem %q", stem)
		}
		name := stem + ".spv"
		if _, exists := expected[name]; exists {
			return "", 0, fmt.Errorf("duplicate expected SPIR-V module %q", stem)
		}
		expected[name] = struct{}{}
	}
	entriesBefore, err := exactSPIRVEntries(root, expected)
	if err != nil {
		return "", 0, err
	}
	h := sha256.New()
	for _, name := range entriesBefore {
		data, err := readStableRegularFile(filepath.Join(root, name), "SPIR-V module "+name)
		if err != nil {
			return "", 0, err
		}
		fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(filepath.Join("internal", "compute", "spirv", name)), len(data))
		_, _ = h.Write(data)
	}
	entriesAfter, err := exactSPIRVEntries(root, expected)
	if err != nil || strings.Join(entriesBefore, "\x00") != strings.Join(entriesAfter, "\x00") {
		return "", 0, fmt.Errorf("runtime SPIR-V directory changed during hashing")
	}
	return hex.EncodeToString(h.Sum(nil)), len(entriesBefore), nil
}

func exactSPIRVEntries(root string, expected map[string]struct{}) ([]string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read runtime SPIR-V root: %w", err)
	}
	observed := make([]string, 0, len(expected))
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) != ".spv" {
			continue
		}
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("SPIR-V module must be a non-symlink regular file: %s", entry.Name())
		}
		if _, ok := expected[entry.Name()]; !ok {
			return nil, fmt.Errorf("unexpected SPIR-V module: %s", entry.Name())
		}
		observed = append(observed, entry.Name())
	}
	if len(observed) != len(expected) {
		return nil, fmt.Errorf("incomplete SPIR-V bundle: got %d modules, want %d", len(observed), len(expected))
	}
	sort.Strings(observed)
	return observed, nil
}

func prepareVulkanSourceWithGit(ctx context.Context, gitRun vulkanGitRunner, root, requestedCommit string) (BuildSourceProvenance, error) {
	root, err := strictAbsoluteDirectory(root, "Vulkan source root")
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	if !validGitObjectID(requestedCommit) {
		return BuildSourceProvenance{}, fmt.Errorf("expected Vulkan source commit must be a full lowercase Git object ID")
	}
	if err := validateVulkanRepoAdminConfig(ctx, gitRun, root); err != nil {
		return BuildSourceProvenance{}, err
	}
	commitRaw, err := gitRun(ctx, root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	commit := strings.TrimSpace(string(commitRaw))
	if commit != requestedCommit {
		return BuildSourceProvenance{}, fmt.Errorf("expected Vulkan source commit %s does not match HEAD %s", requestedCommit, commit)
	}
	treeRaw, err := gitRun(ctx, root, "rev-parse", "--verify", commit+"^{tree}")
	if err != nil {
		return BuildSourceProvenance{}, err
	}
	tree := strings.TrimSpace(string(treeRaw))
	status, err := gitRun(ctx, root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none")
	if err != nil || len(status) != 0 {
		return BuildSourceProvenance{}, fmt.Errorf("Vulkan source root is not clean")
	}
	archive, err := gitRun(ctx, root, "archive", "--format=tar", commit)
	if err != nil {
		return BuildSourceProvenance{}, fmt.Errorf("git archive failed: %w", err)
	}
	archiveDigest := sha256.Sum256(archive)
	afterCommit, err := gitRun(ctx, root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil || strings.TrimSpace(string(afterCommit)) != commit {
		return BuildSourceProvenance{}, fmt.Errorf("Vulkan source commit changed during verification")
	}
	afterTree, err := gitRun(ctx, root, "rev-parse", "--verify", commit+"^{tree}")
	if err != nil || strings.TrimSpace(string(afterTree)) != tree {
		return BuildSourceProvenance{}, fmt.Errorf("Vulkan source tree changed during verification")
	}
	status, err = gitRun(ctx, root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none")
	if err != nil || len(status) != 0 {
		return BuildSourceProvenance{}, fmt.Errorf("Vulkan source root changed during verification")
	}
	return BuildSourceProvenance{GitCommit: commit, GitTree: tree, Clean: true, SourceArchiveSHA256: hex.EncodeToString(archiveDigest[:])}, nil
}

func validateVulkanRepoAdminConfig(ctx context.Context, gitRun vulkanGitRunner, root string) error {
	raw, err := gitRun(ctx, root, "config", "--no-includes", "--null", "--name-only", "--list")
	if err != nil {
		return fmt.Errorf("inspect Vulkan repository configuration: %w", err)
	}
	for _, rawName := range bytes.Split(raw, []byte{0}) {
		name := strings.ToLower(strings.TrimSpace(string(rawName)))
		if name == "" {
			continue
		}
		unsafe := name == "core.alternaterefscommand" || name == "core.gitproxy" || name == "core.sshcommand" ||
			name == "diff.external" || name == "extensions.partialclone" ||
			strings.HasPrefix(name, "include.") || strings.HasPrefix(name, "includeif.") ||
			strings.HasPrefix(name, "filter.") || strings.HasPrefix(name, "tar.") || strings.HasPrefix(name, "url.") ||
			(strings.HasPrefix(name, "diff.") && (strings.HasSuffix(name, ".command") || strings.HasSuffix(name, ".textconv"))) ||
			(strings.HasPrefix(name, "merge.") && strings.HasSuffix(name, ".driver")) ||
			(strings.HasPrefix(name, "remote.") && (strings.HasSuffix(name, ".promisor") || strings.HasSuffix(name, ".partialclonefilter")))
		if unsafe {
			return fmt.Errorf("Vulkan repository has unsupported repo-admin configuration %q", name)
		}
	}
	return nil
}

func compareReceiptToObservation(receipt ComputeBuildReceipt, observed observedVulkanReceiptIdentity) error {
	if receipt.Schema != VulkanBuildReceiptSchema || receipt.Backend != "vulkan" || receipt.Command != "binary" || receipt.Outcome != "success" || receipt.ExitCode != 0 || receipt.Error != "" || receipt.Vulkan == nil || receipt.Artifact == nil {
		return fmt.Errorf("receipt is not a complete successful %s binary receipt", VulkanBuildReceiptSchema)
	}
	if receipt.GitCommit != "" || receipt.GitRef != "" || receipt.Clean != nil || receipt.SourceArchiveSHA256 != "" || receipt.ShaderBundleSHA256 != "" || len(receipt.BuildArgs) != 0 || receipt.Toolchain != nil {
		return fmt.Errorf("Vulkan v2 receipt carries ambiguous legacy provenance fields")
	}
	if !samePath(receipt.ReceiptPath, observed.ReceiptPath) {
		return fmt.Errorf("Vulkan receipt snapshot path mismatch")
	}
	if receipt.Vulkan.Source != observed.Source {
		return fmt.Errorf("Vulkan receipt source identity mismatch")
	}
	if receipt.Vulkan.SPIRVBundleSHA256 != observed.SPIRVBundleSHA256 || receipt.Vulkan.SPIRVModuleCount != observed.SPIRVModuleCount {
		return fmt.Errorf("Vulkan receipt SPIR-V identity mismatch")
	}
	if !sameToolIdentities(receipt.Vulkan.Toolchain, observed.Toolchain) || receipt.Vulkan.ToolchainSHA256 != observed.ToolchainSHA256 {
		return fmt.Errorf("Vulkan receipt toolchain identity mismatch")
	}
	if strings.Join(receipt.Vulkan.NormalizedBuildCommand, "\x00") != strings.Join(observed.NormalizedBuildCommand, "\x00") || receipt.Vulkan.BuildCommandSHA256 != observed.BuildCommandSHA256 {
		return fmt.Errorf("Vulkan receipt build-plan identity mismatch")
	}
	if !samePath(receipt.Artifact.Path, observed.Artifact.Path) || receipt.Artifact.SizeBytes != observed.Artifact.SizeBytes || receipt.Artifact.SHA256 != observed.Artifact.SHA256 || receipt.Artifact.Signed {
		return fmt.Errorf("Vulkan receipt binary identity mismatch")
	}
	if receipt.Vulkan.StableIdentitySHA256 != observed.StableIdentitySHA256 {
		return fmt.Errorf("Vulkan receipt stable identity mismatch")
	}
	return nil
}

func verifyBaselineReproducibility(receipt ComputeBuildReceipt) error {
	if receipt.Reproducibility == nil || receipt.Reproducibility.Status != "baseline" || receipt.Reproducibility.ComparedReceiptSHA256 != "" || len(receipt.Reproducibility.MismatchedFields) != 0 {
		return fmt.Errorf("Vulkan receipt baseline reproducibility record is invalid")
	}
	return nil
}

func verifyMatchedReproducibility(receipt ComputeBuildReceipt, comparisonRaw []byte, comparison ComputeBuildReceipt, observed, comparisonObserved observedVulkanReceiptIdentity) error {
	if receipt.Reproducibility == nil || receipt.Reproducibility.Status != "match" || len(receipt.Reproducibility.MismatchedFields) != 0 {
		return fmt.Errorf("Vulkan receipt match reproducibility record is invalid")
	}
	digest := sha256.Sum256(comparisonRaw)
	if receipt.Reproducibility.ComparedReceiptSHA256 != hex.EncodeToString(digest[:]) {
		return fmt.Errorf("Vulkan receipt comparison snapshot digest mismatch")
	}
	if observed.StableIdentitySHA256 != comparisonObserved.StableIdentitySHA256 || observed.Artifact.SHA256 != comparisonObserved.Artifact.SHA256 || receipt.Vulkan.StableIdentitySHA256 != comparison.Vulkan.StableIdentitySHA256 || receipt.Artifact.SHA256 != comparison.Artifact.SHA256 {
		return fmt.Errorf("Vulkan receipt comparison identity mismatch")
	}
	return nil
}

func vulkanStableIdentitySHA(source BuildSourceProvenance, spirvSHA string, spirvCount int, toolchainSHA, commandSHA, binarySHA string) (string, error) {
	return hashJSON(struct {
		Source             BuildSourceProvenance `json:"source"`
		SPIRVBundleSHA256  string                `json:"spirv_bundle_sha256"`
		SPIRVModuleCount   int                   `json:"spirv_module_count"`
		ToolchainSHA256    string                `json:"toolchain_sha256"`
		BuildCommandSHA256 string                `json:"build_command_sha256"`
		BinarySHA256       string                `json:"binary_sha256"`
	}{source, spirvSHA, spirvCount, toolchainSHA, commandSHA, binarySHA})
}

func validLowerSHA256(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validGitObjectID(value string) bool {
	if len(value) != 40 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func sameToolIdentities(a, b []BuildToolIdentity) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sameObservedVulkanIdentity(a, b observedVulkanReceiptIdentity) bool {
	if !samePath(a.ReceiptPath, b.ReceiptPath) || a.Source != b.Source || a.Artifact != b.Artifact || a.SPIRVBundleSHA256 != b.SPIRVBundleSHA256 || a.SPIRVModuleCount != b.SPIRVModuleCount || a.ToolchainSHA256 != b.ToolchainSHA256 || a.BuildCommandSHA256 != b.BuildCommandSHA256 || a.StableIdentitySHA256 != b.StableIdentitySHA256 || a.GitExecutableSHA256 != b.GitExecutableSHA256 {
		return false
	}
	return sameToolIdentities(a.Toolchain, b.Toolchain) && strings.Join(a.NormalizedBuildCommand, "\x00") == strings.Join(b.NormalizedBuildCommand, "\x00")
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
	stableSHA, err := vulkanStableIdentitySHA(source, spirvSHA, spirvCount, toolchainSHA, commandSHA, binarySHA)
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
