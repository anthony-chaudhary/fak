package computebuild

import (
	"io"
)

// Toolchain contains paths to discovered build tools and compiler flags.
type Toolchain struct {
	Go         string            // Go compiler driver (defaults to go on PATH)
	CC         string            // C compiler (gcc, clang, cl)
	CXX        string            // C++ compiler (g++, clang++, cl)
	AR         string            // Archiver (ar, llvm-ar, lib)
	NVCC       string            // NVIDIA CUDA compiler
	GLSLC      string            // Vulkan GLSL compiler
	VulkanSDK  string            // Root of Vulkan SDK
	VulkanInc  string            // Include path for Vulkan SDK
	VulkanLib  string            // Lib path for Vulkan SDK
	CUDAHome   string            // CUDA Toolkit root directory
	CUDAInc    []string          // CUDA include paths
	CUDALib    []string          // CUDA library search paths
	SignTool   string            // Windows Authenticode signtool.exe
	VsDevCmd   string            // Path to VsDevCmd.bat
	VsDevEnv   map[string]string // Environment variables ingested from VsDevCmd.bat
	CxxRuntime string            // C++ runtime linker flag (e.g. -lstdc++, -lmsvcprt, -lc++)
	IsWindows  bool              // Operating system flag
}

// VulkanConfig defines parameters for Vulkan build/test operations.
type VulkanConfig struct {
	Command            string               // "shaders", "lib", "build", "test", "binary"
	RepoRoot           string               // Root of repository containing go.mod
	PkgDir             string               // Directory of compute package (defaults to <RepoRoot>/internal/compute)
	OutPkg             string               // Target Go package for "binary" command
	OutBin             string               // Output binary path for "binary" command
	ReceiptPath        string               // Path to write receipt JSON (defaults to .fak/compute-build-receipt.json or .fak/vulkan-build-receipt.json)
	CompareReceiptPath string               // Optional prior successful Vulkan receipt to compare reproducible fields against
	GitCommit          string               // Optional commit that must resolve exactly to HEAD
	GitRef             string               // Deprecated display ref retained for API compatibility
	Smoke              bool                 // Default true for binary builds: shift-left execution of the output binary
	SkipSmoke          bool                 // Skip shift-left smoke execution even for binary builds
	Receipt            *ComputeBuildReceipt // Generated receipt populated upon completion
	Toolchain          *Toolchain           // Pre-discovered or custom toolchain
	ToolchainOverrides *Toolchain           // CLI overrides merged after source preflight into discovered defaults
	Stdout             io.Writer            // Standard output stream
	Stderr             io.Writer            // Standard error stream
}

// CUDAConfig defines parameters for CUDA build/test/bench operations.
type CUDAConfig struct {
	Command            string               // "check", "build", "test", "bench", "binary"
	RepoRoot           string               // Root of repository containing go.mod
	PkgDir             string               // Directory of compute package (defaults to <RepoRoot>/internal/compute)
	Arch               string               // Target arch (e.g. "sm_89", "89", or empty for all arches in cuda_arch.txt)
	EnableNCCL         bool                 // Compile NCCL collectives and use -tags cuda,nccl
	NCCLHome           string               // Optional user-space NCCL root directory
	OutPkg             string               // Target Go package for "binary" command
	OutBin             string               // Output binary path for "binary" or "build" command
	BenchModelDir      string               // Model directory for "bench" command
	BenchDecodeSteps   int                  // Decode steps for "bench" command
	SkipSign           bool                 // Skip Authenticode code-signing on Windows
	SignCertThumbprint string               // SHA-1 thumbprint for Windows certificate store
	SignPFX            string               // Path to .pfx certificate file
	SignPFXPassword    string               // Password for .pfx certificate
	TimestampURL       string               // RFC-3161 timestamp URL (default: http://timestamp.digicert.com)
	ReceiptPath        string               // Path to write receipt JSON (defaults to .fak/compute-build-receipt.json or .fak/cuda-build-receipt.json)
	Smoke              bool                 // Default true for binary builds: shift-left execution of the output binary
	SkipSmoke          bool                 // Skip shift-left smoke execution even for binary builds
	Receipt            *ComputeBuildReceipt // Generated receipt populated upon completion
	Toolchain          *Toolchain           // Pre-discovered or custom toolchain
	Stdout             io.Writer            // Standard output stream
	Stderr             io.Writer            // Standard error stream
}

// BuildArtifact records metadata about a generated binary or static library.
type BuildArtifact struct {
	Path       string                 `json:"path"`
	SizeBytes  int64                  `json:"size_bytes"`
	Signed     bool                   `json:"signed,omitempty"`
	SHA256     string                 `json:"sha256,omitempty"`
	CUDAImages *CUDAArchiveImageAudit `json:"cuda_images,omitempty"`
}

const (
	// ComputeBuildReceiptSchema is the stable schema for compute build receipts.
	ComputeBuildReceiptSchema = "fak.compute-build-receipt.v1"
	// VulkanBuildReceiptSchema binds successful Vulkan binary receipts to source, tools, shaders, and output.
	VulkanBuildReceiptSchema = "fak.vulkan-build-receipt.v2"
	// VulkanBuildReceiptSchemaV3 binds the current trusted 60-module registry.
	VulkanBuildReceiptSchemaV3 = "fak.vulkan-build-receipt.v3"
	// VulkanBuildReceiptSchemaV4 binds the trusted 61-module shared-attention registry.
	VulkanBuildReceiptSchemaV4 = "fak.vulkan-build-receipt.v4"
	// VulkanBuildReceiptSchemaV5 binds exact62, the native archive, and score float controls.
	VulkanBuildReceiptSchemaV5 = "fak.vulkan-build-receipt.v5"
	// VulkanBuildReceiptSchemaV6 binds exact63 (V5/62 plus the V4.1 mHC module).
	VulkanBuildReceiptSchemaV6 = "fak.vulkan-build-receipt.v6"
	// VulkanBuildReceiptSchemaV7 binds exact66 (V6/63 plus the V4.1 router top-k, index-select and Q2_K embedding-gather modules).
	VulkanBuildReceiptSchemaV7 = "fak.vulkan-build-receipt.v7"
	// VulkanBuildReceiptSchemaV8 binds exact67 (V7/66 plus the V4.1 bit-exact RMSNorm module).
	VulkanBuildReceiptSchemaV8 = "fak.vulkan-build-receipt.v8"
	// VulkanBuildReceiptSchemaV9 binds exact68 (V8/67 plus checked attention BF16 publication).
	VulkanBuildReceiptSchemaV9 = "fak.vulkan-build-receipt.v9"
	// VulkanBuildReceiptSchemaV10 binds exact69 (V9/68 plus the V4.1 routed-expert activation module).
	VulkanBuildReceiptSchemaV10 = "fak.vulkan-build-receipt.v10"
	// VulkanBuildReceiptSchemaV11 binds exact71 (V10/69 plus the V4.1 Engram gate and compressor pool modules).
	VulkanBuildReceiptSchemaV11 = "fak.vulkan-build-receipt.v11"
	// VulkanBuildReceiptSchemaV12 binds exact72 (V11/71 plus the V4.1 strict binary32 tail-RoPE module).
	VulkanBuildReceiptSchemaV12 = "fak.vulkan-build-receipt.v12"
	// VulkanBuildReceiptSchemaV13 binds exact73 (V12/72 plus the V4.1 strict binary32 scalar-multiply module).
	VulkanBuildReceiptSchemaV13 = "fak.vulkan-build-receipt.v13"
	// VulkanBuildReceiptSchemaV14 binds exact74 (V13/73 plus the V4.1 Engram table-row decode module).
	VulkanBuildReceiptSchemaV14 = "fak.vulkan-build-receipt.v14"
	// VulkanBuildReceiptSchemaV15 binds exact78 (V14/74 plus the four V4.1 routed-expert id modules).
	VulkanBuildReceiptSchemaV15 = "fak.vulkan-build-receipt.v15"
	// VulkanBuildReceiptSchemaV16 binds exact79 (V15/78 plus the run-to-run V4.1 shared attention module).
	VulkanBuildReceiptSchemaV16 = "fak.vulkan-build-receipt.v16"
	// VulkanBuildReceiptSchemaV17 binds exact81 (V16/79 plus the two V4.1 fused routed-FFN modules).
	VulkanBuildReceiptSchemaV17 = "fak.vulkan-build-receipt.v17"
	// VulkanBuildReceiptSchemaV18 binds exact82 (V17/81 plus the experimental Q2_K token panel).
	VulkanBuildReceiptSchemaV18 = "fak.vulkan-build-receipt.v18"
	// VulkanBuildReceiptSchemaV19 binds V18/82 plus the Q3_K token panel.
	VulkanBuildReceiptSchemaV19 = "fak.vulkan-build-receipt.v19"

	// VulkanBuildReceiptSchemaV20 binds V19/83 plus optional row-addressed routed SwiGLU.
	VulkanBuildReceiptSchemaV20 = "fak.vulkan-build-receipt.v20"

	// VulkanBuildReceiptSchemaV21 binds V20/84 plus optional row-addressed down publication.
	VulkanBuildReceiptSchemaV21 = "fak.vulkan-build-receipt.v21"

	// VulkanBuildReceiptSchemaV22 binds V21/85 plus optional fused index-scale BF16 publication.
	VulkanBuildReceiptSchemaV22 = "fak.vulkan-build-receipt.v22"

	// VulkanBuildReceiptSchemaV23 binds canonical fused-scale materialization with the unchanged 86-module registry.
	VulkanBuildReceiptSchemaV23 = "fak.vulkan-build-receipt.v23"

	// VulkanBuildReceiptSchemaV24 adds canonical routed activation chunks.
	VulkanBuildReceiptSchemaV24 = "fak.vulkan-build-receipt.v24"
	// VulkanBuildReceiptSchemaV25 adds canonical down publication chunks.
	VulkanBuildReceiptSchemaV25 = "fak.vulkan-build-receipt.v25"

	// VulkanBuildReceiptSchemaV26 adds canonical paired Q2 gate/up contraction.
	VulkanBuildReceiptSchemaV26 = "fak.vulkan-build-receipt.v26"
	// VulkanBuildReceiptSchemaV27 adds the nine K-quant/i-quant prefill coopmat GEMM modules.
	VulkanBuildReceiptSchemaV27 = "fak.vulkan-build-receipt.v27"
	// VulkanBuildReceiptSchemaV28 adds the grouped-query split-KV decode attention module.
	VulkanBuildReceiptSchemaV28 = "fak.vulkan-build-receipt.v28"
	// VulkanBuildReceiptSchemaV29 adds optional raw F32 finite status.
	VulkanBuildReceiptSchemaV29 = "fak.vulkan-build-receipt.v29"
	// VulkanBuildReceiptSchemaV30 adds the Qwen3.5 GDN core value-head tiling module.
	VulkanBuildReceiptSchemaV30 = "fak.vulkan-build-receipt.v30"
	// VulkanBuildReceiptSchemaV31 adds the V4.1 fused router gate projection + top-k module.
	VulkanBuildReceiptSchemaV31 = "fak.vulkan-build-receipt.v31"
	// VulkanBuildReceiptSchemaV32 adds the V4.1 device-resident compressor ring module.
	VulkanBuildReceiptSchemaV32 = "fak.vulkan-build-receipt.v32"
	// VulkanBuildReceiptSchemaV33 adds the V4.1 split-k shared attention module.
	VulkanBuildReceiptSchemaV33 = "fak.vulkan-build-receipt.v33"
	// VulkanBuildReceiptSchemaV34 adds the optional Qwen GDN Q8 sequence panel module.
	VulkanBuildReceiptSchemaV34 = "fak.vulkan-build-receipt.v34"
	// VulkanBuildReceiptSchemaV35 adds optional Qwen attention postprocessing.
	VulkanBuildReceiptSchemaV35 = "fak.vulkan-build-receipt.v35"
	// VulkanBuildReceiptSchemaV36 adds the optional packed Q6_K panel tile.
	VulkanBuildReceiptSchemaV36 = "fak.vulkan-build-receipt.v36"

	// DefaultComputeBuildReceiptPath is the generic fallback receipt path.
	DefaultComputeBuildReceiptPath = ".fak/compute-build-receipt.json"

	// DefaultVulkanBuildReceiptPath is the default receipt path for Vulkan builds.
	DefaultVulkanBuildReceiptPath = ".fak/vulkan-build-receipt.json"

	// DefaultCUDABuildReceiptPath is the default receipt path for CUDA builds.
	DefaultCUDABuildReceiptPath = ".fak/cuda-build-receipt.json"
)

// SmokeResult captures the outcome of shift-left binary execution.
type SmokeResult struct {
	Command  []string `json:"command"`
	Outcome  string   `json:"outcome"`
	Output   string   `json:"output,omitempty"`
	ExitCode int      `json:"exit_code"`
	Error    string   `json:"error,omitempty"`
}

// ComputeBuildPhase captures the timing and status of an individual build phase.
type ComputeBuildPhase struct {
	Name      string `json:"name"`
	Outcome   string `json:"outcome"`
	ElapsedMS int64  `json:"elapsed_ms"`
	ExitCode  *int   `json:"exit_code,omitempty"`
	Error     string `json:"error,omitempty"`
}

// BuildSourceProvenance binds a successful build to one clean committed Git tree.
type BuildSourceProvenance struct {
	GitCommit           string `json:"git_commit"`
	GitTree             string `json:"git_tree"`
	Clean               bool   `json:"clean"`
	SourceArchiveSHA256 string `json:"source_archive_sha256"`
}

// BuildToolIdentity records the content identity of one effective build tool without
// retaining host-specific absolute paths.
type BuildToolIdentity struct {
	Role       string `json:"role"`
	Executable string `json:"executable"`
	SHA256     string `json:"sha256"`
}

// VulkanBuildProvenance contains only success evidence. Failed receipts omit it.
type VulkanBuildProvenance struct {
	Source                 BuildSourceProvenance `json:"source"`
	SPIRVBundleSHA256      string                `json:"spirv_bundle_sha256"`
	SPIRVModuleCount       int                   `json:"spirv_module_count"`
	Toolchain              []BuildToolIdentity   `json:"toolchain"`
	ToolchainSHA256        string                `json:"toolchain_sha256"`
	NormalizedBuildCommand []string              `json:"normalized_build_command"`
	BuildCommandSHA256     string                `json:"build_command_sha256"`
	StableIdentitySHA256   string                `json:"stable_identity_sha256"`
}

// BuildReproducibility records comparison with a prior successful receipt. The
// baseline status is explicit so it cannot be mistaken for a witnessed match.
type BuildReproducibility struct {
	Status                string   `json:"status"`
	ComparedReceiptSHA256 string   `json:"compared_receipt_sha256,omitempty"`
	MismatchedFields      []string `json:"mismatched_fields,omitempty"`
}

// ToolchainIdentity is the legacy v1 toolchain surface retained for source
// compatibility. Vulkan v2 receipts use content-hashed BuildToolIdentity values.
type ToolchainIdentity struct {
	CC         string   `json:"cc,omitempty"`
	CXX        string   `json:"cxx,omitempty"`
	AR         string   `json:"ar,omitempty"`
	GLSLC      string   `json:"glslc,omitempty"`
	VulkanSDK  string   `json:"vulkan_sdk,omitempty"`
	CxxRuntime string   `json:"cxx_runtime,omitempty"`
	BuildArgs  []string `json:"build_args,omitempty"`
}

// ComputeBuildReceipt records full provenance, phases, artifact metadata, and smoke test results.
type ComputeBuildReceipt struct {
	Schema      string `json:"schema"`
	Backend     string `json:"backend"`
	Command     string `json:"command"`
	Outcome     string `json:"outcome"`
	ExitCode    int    `json:"exit_code"`
	Error       string `json:"error,omitempty"`
	StartedAt   string `json:"started_at"`
	FinishedAt  string `json:"finished_at"`
	ElapsedMS   int64  `json:"elapsed_ms"`
	ReceiptPath string `json:"receipt_path"`
	// Legacy v1 fields remain empty in v2 receipts; authoritative Vulkan
	// provenance lives under Vulkan and never records host-specific paths.
	GitCommit           string                 `json:"git_commit,omitempty"`
	GitRef              string                 `json:"git_ref,omitempty"`
	Clean               *bool                  `json:"clean,omitempty"`
	SourceArchiveSHA256 string                 `json:"source_archive_sha256,omitempty"`
	ShaderBundleSHA256  string                 `json:"shader_bundle_sha256,omitempty"`
	BuildArgs           []string               `json:"build_args,omitempty"`
	Toolchain           *ToolchainIdentity     `json:"toolchain,omitempty"`
	Phases              []ComputeBuildPhase    `json:"phases"`
	Artifact            *BuildArtifact         `json:"artifact,omitempty"`
	Smoke               *SmokeResult           `json:"smoke,omitempty"`
	Vulkan              *VulkanBuildProvenance `json:"vulkan,omitempty"`
	Reproducibility     *BuildReproducibility  `json:"reproducibility,omitempty"`
	// In-memory only: WriteReceiptAtomic uses the explicit successor wire envelope.
	// Keeping this out of the common JSON type preserves V2 unknown-field rejection.
	VulkanRegistry      *VulkanShaderRegistryIdentity `json:"-"`
	VulkanNativeArchive *BuildArtifact                `json:"-"`
	VulkanIndexerScore  *VulkanIndexerScoreContract   `json:"-"`
}
