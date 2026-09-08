package devcmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// VulkanBuildResult represents the structured JSON output for RunBuildVulkan.
type VulkanBuildResult struct {
	Schema     string `json:"schema"`
	Command    string `json:"command"`
	Success    bool   `json:"success"`
	OutBin     string `json:"out_bin,omitempty"`
	Error      string `json:"error,omitempty"`
	DurationMS int64  `json:"duration_ms"`
}

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

	vulkanSDK := fs.String("vulkan-sdk", "", "path to Vulkan SDK root directory")
	cxx := fs.String("cxx", "", "C++ compiler executable")
	ar := fs.String("ar", "", "archiver executable")
	glslc := fs.String("glslc", "", "GLSL shader compiler executable")
	repoRoot := fs.String("repo-root", "", "repository root directory")
	pkgDir := fs.String("pkg-dir", "", "compute package directory")
	outPkg := fs.String("out-pkg", "", "target Go package for binary build")
	outBin := fs.String("out-bin", "", "output binary path")
	fs.StringVar(outBin, "o", "", "output binary path (shorthand)")
	fs.StringVar(outBin, "out", "", "output binary path (alias)")
	receiptFlag := fs.String("receipt", ".fak/vulkan-build-receipt.json", "path to write durable JSON build receipt")
	smokeFlag := fs.Bool("smoke", true, "run shift-left smoke verification on compiled binary")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")
	cmdFlag := fs.String("cmd", "", "subcommand mode (shaders, lib, build, binary, test)")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev build-vulkan [subcommand] [flags] [pkg] [out]")
		fmt.Fprintln(stderr, "subcommands:")
		fmt.Fprintln(stderr, "  shaders    compile GLSL compute shaders to SPIR-V")
		fmt.Fprintln(stderr, "  lib        compile vulkan_shim.cpp to libfakvulkan.a")
		fmt.Fprintln(stderr, "  build      compile shaders, shim, and build compute package")
		fmt.Fprintln(stderr, "  binary     build full binary with Vulkan backend enabled")
		fmt.Fprintln(stderr, "  test       run Vulkan HAL device integration tests")
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
	if sub == "" || sub == "help" {
		fs.Usage()
		if sub == "help" {
			return 0
		}
		return 2
	}

	switch sub {
	case "shaders", "lib", "build", "binary", "test":
	default:
		fmt.Fprintf(stderr, "build-vulkan: unknown subcommand %q (expected shaders, lib, build, binary, test)\n", sub)
		return 2
	}

	cfg := &computebuild.VulkanConfig{
		Command:     sub,
		RepoRoot:    *repoRoot,
		PkgDir:      *pkgDir,
		OutPkg:      *outPkg,
		OutBin:      *outBin,
		ReceiptPath: *receiptFlag,
		Smoke:       *smokeFlag,
		Stdout:      stdout,
		Stderr:      stderr,
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
		tc, _ := computebuild.DiscoverToolchain()
		if tc == nil {
			tc = &computebuild.Toolchain{IsWindows: runtime.GOOS == "windows"}
		}
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
		cfg.Toolchain = tc
	}

	if *jsonOut {
		cfg.Stdout = io.Discard
	}

	start := time.Now()
	err := computebuild.RunVulkan(context.Background(), cfg)
	dur := time.Since(start).Milliseconds()

	if *jsonOut {
		res := VulkanBuildResult{
			Schema:     "fak.vulkan-build.v1",
			Command:    sub,
			Success:    err == nil,
			OutBin:     cfg.OutBin,
			DurationMS: dur,
		}
		if err != nil {
			res.Error = err.Error()
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
