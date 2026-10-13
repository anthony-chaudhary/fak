package devcmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// VulkanBuildResult represents the structured JSON output for RunBuildVulkan.
type VulkanBuildResult struct {
	Schema                string `json:"schema"`
	Command               string `json:"command"`
	Success               bool   `json:"success"`
	OutBin                string `json:"out_bin,omitempty"`
	ReceiptSchema         string `json:"receipt_schema,omitempty"`
	StableIdentitySHA256  string `json:"stable_identity_sha256,omitempty"`
	ReproducibilityStatus string `json:"reproducibility_status,omitempty"`
	GitCommit             string `json:"git_commit,omitempty"`
	GitRef                string `json:"git_ref,omitempty"`
	Clean                 *bool  `json:"clean,omitempty"`
	SourceArchiveSHA256   string `json:"source_archive_sha256,omitempty"`
	BinarySHA256          string `json:"binary_sha256,omitempty"`
	ShaderBundleSHA256    string `json:"shader_bundle_sha256,omitempty"`
	ReceiptPath           string `json:"receipt_path,omitempty"`
	Error                 string `json:"error,omitempty"`
	DurationMS            int64  `json:"duration_ms"`
}

const (
	vulkanVerificationResultSchema       = "fak.vulkan-build-verification.v1"
	vulkanVerifyInvalidArgument          = "VULKAN_VERIFY_INVALID_ARGUMENT"
	vulkanVerifyObservationRefused       = "VULKAN_VERIFY_OBSERVATION_REFUSED"
	vulkanVerifyIdentityRefused          = "VULKAN_VERIFY_IDENTITY_REFUSED"
	vulkanVerifyIncompleteVerifiedResult = "VULKAN_VERIFY_INCOMPLETE_RESULT"
)

// VulkanVerificationResult is the stable, read-only projection consumed by
// deployment adapters. A refused result deliberately carries no identity fields.
type VulkanVerificationResult struct {
	Schema                   string                               `json:"schema"`
	Command                  string                               `json:"command"`
	Verdict                  string                               `json:"verdict"`
	RefusalCode              string                               `json:"refusal_code,omitempty"`
	Error                    string                               `json:"error,omitempty"`
	ReceiptSchema            string                               `json:"receipt_schema,omitempty"`
	ReceiptSHA256            string                               `json:"receipt_sha256,omitempty"`
	PublicRevision           string                               `json:"public_revision,omitempty"`
	PublicTree               string                               `json:"public_tree,omitempty"`
	SourceArchiveSHA256      string                               `json:"source_archive_sha256,omitempty"`
	BinarySHA256             string                               `json:"binary_sha256,omitempty"`
	BinarySizeBytes          int64                                `json:"binary_size_bytes,omitempty"`
	SPIRVBundleSHA256        string                               `json:"spirv_bundle_sha256,omitempty"`
	SPIRVModuleCount         int                                  `json:"spirv_module_count,omitempty"`
	SPIRVModules             []string                             `json:"spirv_modules,omitempty"`
	ShaderRegistry           *vulkanVerificationRegistryV1        `json:"shader_registry,omitempty"`
	NativeArchive            *vulkanVerificationArtifactV1        `json:"native_archive,omitempty"`
	IndexerScoreContract     *vulkanVerificationIndexerScoreV1    `json:"indexer_score_contract,omitempty"`
	Toolchain                []vulkanVerificationToolV1           `json:"toolchain,omitempty"`
	ToolchainSHA256          string                               `json:"toolchain_sha256,omitempty"`
	NormalizedBuildCommand   []string                             `json:"normalized_build_command,omitempty"`
	BuildCommandSHA256       string                               `json:"build_command_sha256,omitempty"`
	StableIdentitySHA256     string                               `json:"stable_identity_sha256,omitempty"`
	Reproducibility          *vulkanVerificationReproducibilityV1 `json:"reproducibility,omitempty"`
	BaselineReceiptSHA256    string                               `json:"baseline_receipt_sha256,omitempty"`
	HistoricalBuildCausality string                               `json:"historical_build_causality,omitempty"`
	UnavailableClaims        []string                             `json:"unavailable_claims,omitempty"`
}

// These v1 wire DTOs freeze the CLI schema independently of computebuild's
// internal Go structs. Changes to receipt internals therefore cannot silently
// add, remove, or retag fields in fak-dev output.
type vulkanVerificationRegistryV1 struct {
	ID          string `json:"id"`
	SHA256      string `json:"sha256"`
	ModuleCount int    `json:"module_count"`
}

type vulkanVerificationArtifactV1 struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	Signed    bool   `json:"signed,omitempty"`
	SHA256    string `json:"sha256"`
}

type vulkanVerificationIndexerScoreV1 struct {
	PolicyID                 string    `json:"policy_id"`
	ModuleSHA256             string    `json:"module_sha256"`
	EntryPoint               string    `json:"entry_point"`
	LocalSize                [3]uint32 `json:"local_size"`
	FloatWidth               uint32    `json:"float_width"`
	DenormPreserve           bool      `json:"denorm_preserve"`
	SignedZeroInfNanPreserve bool      `json:"signed_zero_inf_nan_preserve"`
	RoundingModeRTE          bool      `json:"rounding_mode_rte"`
	NoContractionFAddCount   uint32    `json:"no_contraction_fadd_count"`
	NoContractionFMulCount   uint32    `json:"no_contraction_fmul_count"`
}

type vulkanVerificationToolV1 struct {
	Role       string `json:"role"`
	Executable string `json:"executable"`
	SHA256     string `json:"sha256"`
}

type vulkanVerificationReproducibilityV1 struct {
	Status                string   `json:"status"`
	ComparedReceiptSHA256 string   `json:"compared_receipt_sha256"`
	MismatchedFields      []string `json:"mismatched_fields,omitempty"`
}

type vulkanVerifyConfig struct {
	RepoRoot        string
	Commit          string
	ReceiptPath     string
	BinaryPath      string
	BaselineReceipt string
	BaselineBinary  string
	SPIRVDir        string
	GitExecutable   string
	GitSHA256       string
	Go              string
	CC              string
	CXX             string
	AR              string
	GLSLC           string
	CXXRuntime      string
}

var (
	verifyVulkanBinaryReceiptIdentityForSchema = computebuild.VerifyVulkanBinaryReceiptIdentityForSchema
	currentVulkanShaderRegistryV15             = computebuild.CurrentVulkanShaderRegistryV15
	observeVulkanVerificationBinary            = hashVulkanVerificationBinary
)

// RunBuildVulkan parses CLI arguments and runs Vulkan build/test tasks via computebuild.RunVulkan.
func RunBuildVulkan(stdout, stderr io.Writer, argv []string) int {
	var sub string
	var flagArgs []string
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		sub = argv[0]
		flagArgs = argv[1:]
	} else {
		flagArgs = argv
	}

	fs := flag.NewFlagSet("build-vulkan", flag.ContinueOnError)
	fs.SetOutput(stderr)

	defaultReceipt := ".fak/vulkan-build-receipt.json"
	vulkanSDK := fs.String("vulkan-sdk", "", "path to Vulkan SDK root directory")
	goTool := fs.String("go", "", "absolute Go executable path for receipt verification")
	cc := fs.String("cc", "", "absolute C compiler path for receipt verification")
	cxx := fs.String("cxx", "", "C++ compiler executable")
	ar := fs.String("ar", "", "archiver executable")
	glslc := fs.String("glslc", "", "GLSL shader compiler executable")
	cxxRuntime := fs.String("cxx-runtime", "", "C++ runtime linker flag recorded by the build")
	gitExecutable := fs.String("git", "", "absolute Git executable path for receipt verification")
	gitSHA256 := fs.String("git-sha256", "", "pinned SHA-256 of the Git executable")
	repoRoot := fs.String("repo-root", "", "repository root directory")
	pkgDir := fs.String("pkg-dir", "", "compute package directory")
	outPkg := fs.String("out-pkg", "", "target Go package for binary build")
	outBin := fs.String("out-bin", "", "output binary path")
	fs.StringVar(outBin, "o", "", "output binary path (shorthand)")
	fs.StringVar(outBin, "out", "", "output binary path (alias)")
	receiptFlag := fs.String("receipt", defaultReceipt, "build receipt path (matched receipt for verify)")
	fs.StringVar(receiptFlag, "matched-receipt", defaultReceipt, "matched V15 receipt path for verify")
	compareReceipt := fs.String("compare-receipt", "", "prior successful Vulkan receipt whose stable build identity must match")
	fs.StringVar(compareReceipt, "baseline-receipt", "", "baseline V15 receipt path for verify")
	verifyBinary := fs.String("binary", "", "matched Vulkan binary path for verify")
	fs.StringVar(verifyBinary, "matched-binary", "", "matched Vulkan binary path for verify")
	baselineBinary := fs.String("baseline-binary", "", "baseline Vulkan binary path for verify")
	fs.StringVar(baselineBinary, "compare-binary", "", "baseline Vulkan binary path for verify")
	spirvDir := fs.String("spirv-dir", "", "exact SPIR-V bundle directory for verify")
	commitFlag := fs.String("commit", "", "pinned Git commit SHA (must resolve exactly to HEAD)")
	fs.StringVar(commitFlag, "git-commit", "", "pinned Git commit SHA (must resolve exactly to HEAD)")
	fs.StringVar(commitFlag, "git-tip", "", "pinned Git commit SHA (must resolve exactly to HEAD)")
	refFlag := fs.String("ref", "HEAD", "display ref retained for compatibility")
	fs.StringVar(refFlag, "git-ref", "HEAD", "display ref retained for compatibility")
	smokeFlag := fs.Bool("smoke", true, "run shift-left smoke verification on compiled binary")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")
	cmdFlag := fs.String("cmd", "", "subcommand mode (shaders, lib, build, binary, test, verify)")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev build-vulkan [subcommand] [flags] [pkg] [out]")
		fmt.Fprintln(stderr, "subcommands:")
		fmt.Fprintln(stderr, "  shaders    compile GLSL compute shaders to SPIR-V")
		fmt.Fprintln(stderr, "  lib        compile vulkan_shim.cpp to libfakvulkan.a")
		fmt.Fprintln(stderr, "  build      compile shaders, shim, and build compute package")
		fmt.Fprintln(stderr, "  binary     build full binary with Vulkan backend enabled")
		fmt.Fprintln(stderr, "  test       run Vulkan HAL device integration tests")
		fmt.Fprintln(stderr, "  verify     verify an existing V15 receipt pair without executing either binary")
		fmt.Fprintln(stderr, "flags:")
		fs.PrintDefaults()
	}

	if err := fs.Parse(flagArgs); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	if sub == "" {
		if *cmdFlag != "" {
			sub = *cmdFlag
		} else if fs.NArg() > 0 {
			sub = fs.Arg(0)
		}
	}

	sub = strings.ToLower(strings.TrimSpace(sub))
	setFlags := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })
	if sub == "verify" && !setFlags["receipt"] && !setFlags["matched-receipt"] {
		// The legacy build receipt default is an output path, never trusted
		// verification evidence. Verification requires an explicit input.
		*receiptFlag = ""
	}
	if sub == "" || sub == "help" {
		fs.Usage()
		if sub == "help" {
			return 0
		}
		return 2
	}

	switch sub {
	case "shaders", "lib", "build", "binary", "test", "verify":
	default:
		fmt.Fprintf(stderr, "build-vulkan: unknown subcommand %q (expected shaders, lib, build, binary, test, verify)\n", sub)
		return 2
	}
	if sub != "verify" {
		for _, name := range []string{
			"go", "cc", "cxx-runtime", "git", "git-sha256", "matched-receipt",
			"binary", "matched-binary", "baseline-receipt", "baseline-binary",
			"compare-binary", "spirv-dir",
		} {
			if setFlags[name] {
				fmt.Fprintf(stderr, "build-vulkan: --%s is only valid with the verify subcommand\n", name)
				return 2
			}
		}
	}

	if sub == "verify" {
		return runBuildVulkanVerify(stdout, stderr, vulkanVerifyConfig{
			RepoRoot:        *repoRoot,
			Commit:          *commitFlag,
			ReceiptPath:     *receiptFlag,
			BinaryPath:      *verifyBinary,
			BaselineReceipt: *compareReceipt,
			BaselineBinary:  *baselineBinary,
			SPIRVDir:        *spirvDir,
			GitExecutable:   *gitExecutable,
			GitSHA256:       *gitSHA256,
			Go:              *goTool,
			CC:              *cc,
			CXX:             *cxx,
			AR:              *ar,
			GLSLC:           *glslc,
			CXXRuntime:      *cxxRuntime,
		})
	}

	cfg := &computebuild.VulkanConfig{
		Command:            sub,
		RepoRoot:           *repoRoot,
		PkgDir:             *pkgDir,
		OutPkg:             *outPkg,
		OutBin:             *outBin,
		ReceiptPath:        *receiptFlag,
		CompareReceiptPath: *compareReceipt,
		GitCommit:          *commitFlag,
		GitRef:             *refFlag,
		Smoke:              *smokeFlag,
		SkipSmoke:          !*smokeFlag,
		Stdout:             stdout,
		Stderr:             stderr,
	}

	// For binary subcommand, allow trailing positional args: [pkg] [out]
	posArgs := fs.Args()
	if len(posArgs) > 0 && posArgs[0] == sub {
		posArgs = posArgs[1:]
	}
	if sub == "binary" {
		if cfg.OutPkg == "" && len(posArgs) > 0 {
			cfg.OutPkg = posArgs[0]
		}
		if cfg.OutBin == "" && len(posArgs) > 1 {
			cfg.OutBin = posArgs[1]
		}
	}

	// Customize toolchain if explicit overrides were passed
	if *vulkanSDK != "" || *cxx != "" || *ar != "" || *glslc != "" {
		// Do not discover or execute any tool here: RunVulkan must complete its
		// clean-source preflight before tool access. RunVulkan merges these partial
		// overrides into discovered defaults only after that preflight succeeds.
		tc := &computebuild.Toolchain{Go: "go", IsWindows: runtime.GOOS == "windows"}
		if *vulkanSDK != "" {
			tc.VulkanSDK = *vulkanSDK
			tc.VulkanInc = filepath.Join(*vulkanSDK, "Include")
			tc.VulkanLib = filepath.Join(*vulkanSDK, "Lib")
		}
		if *cxx != "" {
			tc.CXX = *cxx
		}
		if *ar != "" {
			tc.AR = *ar
		}
		if *glslc != "" {
			tc.GLSLC = *glslc
		}
		cfg.ToolchainOverrides = tc
	}

	if *jsonOut {
		cfg.Stdout = io.Discard
	}

	start := time.Now()
	err := computebuild.RunVulkan(context.Background(), cfg)
	dur := time.Since(start).Milliseconds()

	if *jsonOut {
		res := VulkanBuildResult{
			Schema:      "fak.vulkan-build.v1",
			Command:     sub,
			Success:     err == nil,
			OutBin:      cfg.OutBin,
			GitRef:      cfg.GitRef,
			ReceiptPath: cfg.ReceiptPath,
			DurationMS:  dur,
		}
		if err != nil {
			res.Error = err.Error()
		} else if cfg.Receipt != nil {
			res.ReceiptSchema = cfg.Receipt.Schema
			if cfg.Receipt.Vulkan != nil {
				res.StableIdentitySHA256 = cfg.Receipt.Vulkan.StableIdentitySHA256
				res.GitCommit = cfg.Receipt.Vulkan.Source.GitCommit
				clean := cfg.Receipt.Vulkan.Source.Clean
				res.Clean = &clean
				res.SourceArchiveSHA256 = cfg.Receipt.Vulkan.Source.SourceArchiveSHA256
				res.ShaderBundleSHA256 = cfg.Receipt.Vulkan.SPIRVBundleSHA256
			}
			if cfg.Receipt.Artifact != nil {
				res.BinarySHA256 = cfg.Receipt.Artifact.SHA256
			}
			if cfg.Receipt.Reproducibility != nil {
				res.ReproducibilityStatus = cfg.Receipt.Reproducibility.Status
			}
		}
		_ = json.NewEncoder(stdout).Encode(res)
		if err != nil {
			return 1
		}
		return 0
	}

	if err != nil {
		fmt.Fprintf(stderr, "build-vulkan: %v\n", err)
		return 1
	}

	return 0
}

func runBuildVulkanVerify(stdout, stderr io.Writer, cfg vulkanVerifyConfig) int {
	if err := cfg.validate(); err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyInvalidArgument, Error: err.Error(),
		}, 2)
	}

	matchedSHA, matchedSize, err := observeVulkanVerificationBinary(cfg.BinaryPath)
	if err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyObservationRefused, Error: err.Error(),
		}, 1)
	}
	baselineSHA, baselineSize, err := observeVulkanVerificationBinary(cfg.BaselineBinary)
	if err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyObservationRefused, Error: err.Error(),
		}, 1)
	}
	matchedReceiptSHA, _, err := hashVulkanVerificationFile(cfg.ReceiptPath, "matched Vulkan receipt")
	if err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyObservationRefused, Error: err.Error(),
		}, 1)
	}
	baselineReceiptSHA, _, err := hashVulkanVerificationFile(cfg.BaselineReceipt, "baseline Vulkan receipt")
	if err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyObservationRefused, Error: err.Error(),
		}, 1)
	}

	modules := currentVulkanShaderRegistryV15()
	tools := computebuild.VulkanReceiptToolInputs{
		Go: cfg.Go, CC: cfg.CC, CXX: cfg.CXX, AR: cfg.AR, GLSLC: cfg.GLSLC,
		CxxRuntime: cfg.CXXRuntime, IsWindows: false,
	}
	plan := computebuild.VulkanReceiptBuildPlan{
		PackageDir: filepath.Join(cfg.RepoRoot, "internal", "compute"),
		OutPackage: "./cmd/fak",
		Smoke:      false,
	}
	baseline := &computebuild.VulkanBinaryReceiptEvidence{
		ReceiptPath:          cfg.BaselineReceipt,
		SourceRoot:           cfg.RepoRoot,
		GitExecutable:        cfg.GitExecutable,
		ExpectedGitSHA256:    cfg.GitSHA256,
		ExpectedCommit:       cfg.Commit,
		BinaryPath:           cfg.BaselineBinary,
		ExpectedBinarySHA256: baselineSHA,
		ExpectedBinarySize:   baselineSize,
		SPIRVRoot:            cfg.SPIRVDir,
		ExpectedSPIRVModules: append([]string(nil), modules...),
		Tools:                tools,
		BuildPlan:            plan,
	}
	evidence := computebuild.VulkanBinaryReceiptEvidence{
		ReceiptPath:          cfg.ReceiptPath,
		SourceRoot:           cfg.RepoRoot,
		GitExecutable:        cfg.GitExecutable,
		ExpectedGitSHA256:    cfg.GitSHA256,
		ExpectedCommit:       cfg.Commit,
		BinaryPath:           cfg.BinaryPath,
		ExpectedBinarySHA256: matchedSHA,
		ExpectedBinarySize:   matchedSize,
		SPIRVRoot:            cfg.SPIRVDir,
		ExpectedSPIRVModules: append([]string(nil), modules...),
		Tools:                tools,
		BuildPlan:            plan,
		Comparison:           baseline,
	}

	verified, err := verifyVulkanBinaryReceiptIdentityForSchema(context.Background(), computebuild.VulkanBuildReceiptSchemaV15, evidence)
	if err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyIdentityRefused, Error: err.Error(),
		}, 1)
	}
	if err := validateVulkanVerificationSuccess(verified, cfg, modules, matchedReceiptSHA, baselineReceiptSHA, matchedSHA, matchedSize); err != nil {
		return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
			Schema: vulkanVerificationResultSchema, Command: "verify", Verdict: "refused",
			RefusalCode: vulkanVerifyIncompleteVerifiedResult, Error: err.Error(),
		}, 1)
	}

	receipt := verified.Receipt
	provenance := receipt.Vulkan
	return writeVulkanVerificationResult(stdout, stderr, VulkanVerificationResult{
		Schema:                   vulkanVerificationResultSchema,
		Command:                  "verify",
		Verdict:                  "verified",
		ReceiptSchema:            receipt.Schema,
		ReceiptSHA256:            verified.ReceiptSHA256,
		PublicRevision:           provenance.Source.GitCommit,
		PublicTree:               provenance.Source.GitTree,
		SourceArchiveSHA256:      provenance.Source.SourceArchiveSHA256,
		BinarySHA256:             receipt.Artifact.SHA256,
		BinarySizeBytes:          receipt.Artifact.SizeBytes,
		SPIRVBundleSHA256:        provenance.SPIRVBundleSHA256,
		SPIRVModuleCount:         provenance.SPIRVModuleCount,
		SPIRVModules:             append([]string(nil), modules...),
		ShaderRegistry:           mapVulkanVerificationRegistry(receipt.VulkanRegistry),
		NativeArchive:            mapVulkanVerificationArtifact(receipt.VulkanNativeArchive),
		IndexerScoreContract:     mapVulkanVerificationIndexerScore(receipt.VulkanIndexerScore),
		Toolchain:                mapVulkanVerificationTools(provenance.Toolchain),
		ToolchainSHA256:          provenance.ToolchainSHA256,
		NormalizedBuildCommand:   append([]string(nil), provenance.NormalizedBuildCommand...),
		BuildCommandSHA256:       provenance.BuildCommandSHA256,
		StableIdentitySHA256:     provenance.StableIdentitySHA256,
		Reproducibility:          mapVulkanVerificationReproducibility(receipt.Reproducibility),
		BaselineReceiptSHA256:    receipt.Reproducibility.ComparedReceiptSHA256,
		HistoricalBuildCausality: verified.HistoricalBuildCausality,
		UnavailableClaims:        append([]string(nil), verified.UnavailableClaims...),
	}, 0)
}

func validateVulkanVerificationSuccess(verified *computebuild.VulkanBinaryReceiptIdentityVerification, cfg vulkanVerifyConfig, modules []string, matchedReceiptSHA, baselineReceiptSHA, matchedSHA string, matchedSize int64) error {
	incomplete := func(field string) error {
		return fmt.Errorf("canonical verifier returned an incomplete verified identity: %s", field)
	}
	if verified == nil {
		return incomplete("result")
	}
	if !validVulkanVerificationHex(matchedReceiptSHA, sha256.Size) || verified.ReceiptSHA256 != matchedReceiptSHA {
		return incomplete("receipt digest")
	}
	receipt := &verified.Receipt
	if receipt.Schema != computebuild.VulkanBuildReceiptSchemaV15 || receipt.Backend != "vulkan" || receipt.Command != "binary" || receipt.Outcome != "success" || receipt.ExitCode != 0 || receipt.Error != "" {
		return incomplete("V15 success envelope")
	}
	if receipt.Vulkan == nil || receipt.Artifact == nil || receipt.VulkanRegistry == nil || receipt.VulkanNativeArchive == nil || receipt.VulkanIndexerScore == nil || receipt.Reproducibility == nil {
		return incomplete("required receipt sections")
	}

	provenance := receipt.Vulkan
	if !provenance.Source.Clean || provenance.Source.GitCommit != cfg.Commit || !validVulkanVerificationHex(provenance.Source.GitCommit, 20) || !validVulkanVerificationHex(provenance.Source.GitTree, 20) || !validVulkanVerificationHex(provenance.Source.SourceArchiveSHA256, sha256.Size) {
		return incomplete("public source identity")
	}
	if matchedSize <= 0 || !validVulkanVerificationHex(matchedSHA, sha256.Size) || receipt.Artifact.SHA256 != matchedSHA || receipt.Artifact.SizeBytes != matchedSize || strings.TrimSpace(receipt.Artifact.Path) == "" || receipt.Artifact.Signed {
		return incomplete("matched binary identity")
	}
	if len(modules) != 78 || provenance.SPIRVModuleCount != 78 || !validVulkanVerificationHex(provenance.SPIRVBundleSHA256, sha256.Size) {
		return incomplete("exact78 SPIR-V identity")
	}
	registrySHA, err := vulkanVerificationRegistrySHA(modules)
	if err != nil {
		return incomplete("exact78 registry digest")
	}
	registry := receipt.VulkanRegistry
	if registry.ID != computebuild.VulkanShaderRegistryV15ID || registry.ModuleCount != 78 || registry.SHA256 != registrySHA || !validVulkanVerificationHex(registry.SHA256, sha256.Size) {
		return incomplete("exact78 registry identity")
	}
	archive := receipt.VulkanNativeArchive
	if filepath.ToSlash(archive.Path) != "internal/compute/libfakvulkan.a" || archive.SizeBytes <= 0 || archive.Signed || !validVulkanVerificationHex(archive.SHA256, sha256.Size) {
		return incomplete("native archive identity")
	}
	score := receipt.VulkanIndexerScore
	if score.PolicyID != "fak.v41-indexer-score-f32-structure.v2" || score.EntryPoint != "main" || !validVulkanVerificationHex(score.ModuleSHA256, sha256.Size) || score.LocalSize != [3]uint32{64, 1, 1} || score.FloatWidth != 32 || !score.DenormPreserve || !score.SignedZeroInfNanPreserve || !score.RoundingModeRTE || score.NoContractionFAddCount != 2 || score.NoContractionFMulCount != 2 {
		return incomplete("indexer score contract")
	}
	roles := []string{"go", "cc", "cxx", "ar", "glslc"}
	if len(provenance.Toolchain) != len(roles) || !validVulkanVerificationHex(provenance.ToolchainSHA256, sha256.Size) {
		return incomplete("toolchain identity")
	}
	for i, tool := range provenance.Toolchain {
		if tool.Role != roles[i] || strings.TrimSpace(tool.Executable) == "" || !validVulkanVerificationHex(tool.SHA256, sha256.Size) {
			return incomplete("tool identity")
		}
	}
	if len(provenance.NormalizedBuildCommand) == 0 || !validVulkanVerificationHex(provenance.BuildCommandSHA256, sha256.Size) || !validVulkanVerificationHex(provenance.StableIdentitySHA256, sha256.Size) {
		return incomplete("build-plan identity")
	}
	for _, command := range provenance.NormalizedBuildCommand {
		if strings.TrimSpace(command) == "" {
			return incomplete("normalized build command")
		}
	}
	reproducibility := receipt.Reproducibility
	if reproducibility.Status != "match" || !validVulkanVerificationHex(baselineReceiptSHA, sha256.Size) || reproducibility.ComparedReceiptSHA256 != baselineReceiptSHA || len(reproducibility.MismatchedFields) != 0 {
		return incomplete("matched reproducibility identity")
	}
	if verified.HistoricalBuildCausality != "unavailable" || len(verified.UnavailableClaims) == 0 {
		return incomplete("historical causality limits")
	}
	seenClaims := make(map[string]bool, len(verified.UnavailableClaims))
	for _, claim := range verified.UnavailableClaims {
		claim = strings.TrimSpace(claim)
		if claim == "" || seenClaims[claim] {
			return incomplete("unavailable claims")
		}
		seenClaims[claim] = true
	}
	return nil
}

func vulkanVerificationRegistrySHA(modules []string) (string, error) {
	canonical := append([]string(nil), modules...)
	sort.Strings(canonical)
	for i, module := range canonical {
		if strings.TrimSpace(module) == "" || filepath.Base(module) != module || filepath.Ext(module) != "" || (i > 0 && canonical[i-1] == module) {
			return "", fmt.Errorf("invalid exact78 module registry")
		}
	}
	raw, err := json.Marshal(canonical)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:]), nil
}

func validVulkanVerificationHex(value string, byteLength int) bool {
	if len(value) != byteLength*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func mapVulkanVerificationRegistry(value *computebuild.VulkanShaderRegistryIdentity) *vulkanVerificationRegistryV1 {
	return &vulkanVerificationRegistryV1{ID: value.ID, SHA256: value.SHA256, ModuleCount: value.ModuleCount}
}

func mapVulkanVerificationArtifact(value *computebuild.BuildArtifact) *vulkanVerificationArtifactV1 {
	return &vulkanVerificationArtifactV1{Path: value.Path, SizeBytes: value.SizeBytes, Signed: value.Signed, SHA256: value.SHA256}
}

func mapVulkanVerificationIndexerScore(value *computebuild.VulkanIndexerScoreContract) *vulkanVerificationIndexerScoreV1 {
	return &vulkanVerificationIndexerScoreV1{
		PolicyID: value.PolicyID, ModuleSHA256: value.ModuleSHA256, EntryPoint: value.EntryPoint,
		LocalSize: value.LocalSize, FloatWidth: value.FloatWidth,
		DenormPreserve: value.DenormPreserve, SignedZeroInfNanPreserve: value.SignedZeroInfNanPreserve,
		RoundingModeRTE: value.RoundingModeRTE, NoContractionFAddCount: value.NoContractionFAddCount,
		NoContractionFMulCount: value.NoContractionFMulCount,
	}
}

func mapVulkanVerificationTools(values []computebuild.BuildToolIdentity) []vulkanVerificationToolV1 {
	result := make([]vulkanVerificationToolV1, len(values))
	for i, value := range values {
		result[i] = vulkanVerificationToolV1{Role: value.Role, Executable: value.Executable, SHA256: value.SHA256}
	}
	return result
}

func mapVulkanVerificationReproducibility(value *computebuild.BuildReproducibility) *vulkanVerificationReproducibilityV1 {
	return &vulkanVerificationReproducibilityV1{
		Status: value.Status, ComparedReceiptSHA256: value.ComparedReceiptSHA256,
		MismatchedFields: append([]string(nil), value.MismatchedFields...),
	}
}

func (cfg vulkanVerifyConfig) validate() error {
	paths := []struct {
		name  string
		value string
	}{
		{"repo-root", cfg.RepoRoot},
		{"receipt", cfg.ReceiptPath},
		{"binary", cfg.BinaryPath},
		{"baseline-receipt", cfg.BaselineReceipt},
		{"baseline-binary", cfg.BaselineBinary},
		{"spirv-dir", cfg.SPIRVDir},
		{"git", cfg.GitExecutable},
		{"go", cfg.Go},
		{"cc", cfg.CC},
		{"cxx", cfg.CXX},
		{"ar", cfg.AR},
		{"glslc", cfg.GLSLC},
	}
	for _, field := range paths {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("--%s is required for read-only V15 verification", field.name)
		}
		if !filepath.IsAbs(field.value) {
			return fmt.Errorf("--%s must be an absolute path", field.name)
		}
	}
	if strings.TrimSpace(cfg.Commit) == "" {
		return fmt.Errorf("--commit is required for read-only V15 verification")
	}
	if strings.TrimSpace(cfg.GitSHA256) == "" {
		return fmt.Errorf("--git-sha256 is required for pinned Git execution")
	}
	if strings.TrimSpace(cfg.CXXRuntime) == "" {
		return fmt.Errorf("--cxx-runtime is required to reconstruct the build toolchain identity")
	}
	if sameCleanPath(cfg.ReceiptPath, cfg.BaselineReceipt) {
		return fmt.Errorf("matched and baseline receipts must be distinct files")
	}
	if sameCleanPath(cfg.BinaryPath, cfg.BaselineBinary) {
		return fmt.Errorf("matched and baseline binaries must be distinct files")
	}
	return nil
}

func hashVulkanVerificationBinary(path string) (string, int64, error) {
	return hashVulkanVerificationFile(path, "Vulkan binary")
}

func hashVulkanVerificationFile(path, label string) (string, int64, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return "", 0, fmt.Errorf("stat %s: %w", label, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return "", 0, fmt.Errorf("%s must be a non-symlink regular file", label)
	}
	f, err := os.Open(path)
	if err != nil {
		return "", 0, fmt.Errorf("open %s: %w", label, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", 0, fmt.Errorf("hash %s: %w", label, err)
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(info, after) || info.Size() != after.Size() || !info.ModTime().Equal(after.ModTime()) {
		return "", 0, fmt.Errorf("%s changed during observation", label)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), info.Size(), nil
}

func sameCleanPath(a, b string) bool {
	a = filepath.Clean(a)
	b = filepath.Clean(b)
	if os.PathSeparator == '\\' {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func writeVulkanVerificationResult(stdout, stderr io.Writer, result VulkanVerificationResult, exitCode int) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "build-vulkan verify: encode result: %v\n", err)
		return 1
	}
	return exitCode
}
