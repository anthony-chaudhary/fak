package computebuild

import (
	"io"
)

// Toolchain contains paths to discovered build tools and compiler flags.
type Toolchain struct {
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
	Command   string     // "shaders", "lib", "build", "test", "binary"
	RepoRoot  string     // Root of repository containing go.mod
	PkgDir    string     // Directory of compute package (defaults to <RepoRoot>/internal/compute)
	OutPkg    string     // Target Go package for "binary" command
	OutBin    string     // Output binary path for "binary" command
	Toolchain *Toolchain // Pre-discovered or custom toolchain
	Stdout    io.Writer  // Standard output stream
	Stderr    io.Writer  // Standard error stream
}

// CUDAConfig defines parameters for CUDA build/test/bench operations.
type CUDAConfig struct {
	Command            string     // "check", "build", "test", "bench", "binary"
	RepoRoot           string     // Root of repository containing go.mod
	PkgDir             string     // Directory of compute package (defaults to <RepoRoot>/internal/compute)
	Arch               string     // Target arch (e.g. "sm_89", "89", or empty for all arches in cuda_arch.txt)
	EnableNCCL         bool       // Compile NCCL collectives and use -tags cuda,nccl
	NCCLHome           string     // Optional user-space NCCL root directory
	OutPkg             string     // Target Go package for "binary" command
	OutBin             string     // Output binary path for "binary" or "build" command
	BenchModelDir      string     // Model directory for "bench" command
	BenchDecodeSteps   int        // Decode steps for "bench" command
	SkipSign           bool       // Skip Authenticode code-signing on Windows
	SignCertThumbprint string     // SHA-1 thumbprint for Windows certificate store
	SignPFX            string     // Path to .pfx certificate file
	SignPFXPassword    string     // Password for .pfx certificate
	TimestampURL       string     // RFC-3161 timestamp URL (default: http://timestamp.digicert.com)
	Toolchain          *Toolchain // Pre-discovered or custom toolchain
	Stdout             io.Writer  // Standard output stream
	Stderr             io.Writer  // Standard error stream
}

// BuildArtifact records metadata about a generated binary or static library.
type BuildArtifact struct {
	Path      string
	SizeBytes int64
	Signed    bool
}
