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

// CUDABuildResult represents the structured JSON output for RunBuildCUDA.
type CUDABuildResult struct {
	Schema     string `json:"schema"`
	Command    string `json:"command"`
	Arch       string `json:"arch,omitempty"`
	Success    bool   `json:"success"`
	OutBin     string `json:"out_bin,omitempty"`
	Error      string `json:"error,omitempty"`
	DurationMS int64  `json:"duration_ms"`
}

// RunBuildCUDA parses CLI arguments and runs CUDA build/test/check tasks via computebuild.RunCUDA.
func RunBuildCUDA(stdout, stderr io.Writer, argv []string) int {
	var sub string
	var flagArgs []string
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		sub = argv[0]
		flagArgs = argv[1:]
	} else {
		flagArgs = argv
	}

	fs := flag.NewFlagSet("build-cuda", flag.ContinueOnError)
	fs.SetOutput(stderr)

	arch := fs.String("arch", "", "target CUDA architecture (e.g. sm_89, 89, or empty for all)")
	cudaHome := fs.String("cuda-home", "", "path to CUDA Toolkit root directory")
	nccl := fs.Bool("nccl", false, "compile with NCCL collectives enabled (-tags cuda,nccl)")
	ncclHome := fs.String("nccl-home", "", "optional path to NCCL installation directory")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")
	receiptFlag := fs.String("receipt", ".fak/cuda-build-receipt.json", "path to write durable JSON build receipt")
	smokeFlag := fs.Bool("smoke", true, "run shift-left smoke verification on compiled binary")
	repoRoot := fs.String("repo-root", "", "repository root directory")
	pkgDir := fs.String("pkg-dir", "", "compute package directory")
	outPkg := fs.String("out-pkg", "", "target Go package for binary build")
	outBin := fs.String("out-bin", "", "output binary path")
	fs.StringVar(outBin, "o", "", "output binary path (shorthand)")
	fs.StringVar(outBin, "out", "", "output binary path (alias)")
	benchModel := fs.String("bench-model", "", "model directory for bench command")
	benchSteps := fs.Int("bench-steps", 0, "decode steps for bench command")
	skipSign := fs.Bool("skip-sign", false, "skip Authenticode code signing on Windows")
	signCert := fs.String("sign-cert", "", "SHA-1 thumbprint for Windows certificate store")
	signPFX := fs.String("sign-pfx", "", "path to .pfx certificate file")
	signPassword := fs.String("sign-password", "", "password for .pfx certificate")
	timestampURL := fs.String("timestamp-url", "", "RFC-3161 timestamp server URL")
	nvcc := fs.String("nvcc", "", "path to nvcc compiler")
	cxx := fs.String("cxx", "", "C++ compiler executable")
	cc := fs.String("cc", "", "C compiler executable")
	cmdFlag := fs.String("cmd", "", "subcommand mode (check, build, test, bench, binary)")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev build-cuda [subcommand] [flags] [pkg] [out]")
		fmt.Fprintln(stderr, "subcommands:")
		fmt.Fprintln(stderr, "  check      run GPU-free ABI parity, arch matrix, and header syntax checks")
		fmt.Fprintln(stderr, "  build      compile CUDA kernels to libfakcuda.a and build compute package")
		fmt.Fprintln(stderr, "  test       run CUDA HAL device integration tests")
		fmt.Fprintln(stderr, "  bench      run decode microbenchmarks against model weights")
		fmt.Fprintln(stderr, "  binary     build full binary with CUDA backend enabled")
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
	case "check", "build", "test", "bench", "binary":
	default:
		fmt.Fprintf(stderr, "build-cuda: unknown subcommand %q (expected check, build, test, bench, binary)\n", sub)
		return 2
	}

	cfg := &computebuild.CUDAConfig{
		Command:            sub,
		RepoRoot:           *repoRoot,
		PkgDir:             *pkgDir,
		Arch:               *arch,
		EnableNCCL:         *nccl,
		NCCLHome:           *ncclHome,
		OutPkg:             *outPkg,
		OutBin:             *outBin,
		ReceiptPath:        *receiptFlag,
		Smoke:              *smokeFlag,
		BenchModelDir:      *benchModel,
		BenchDecodeSteps:   *benchSteps,
		SkipSign:           *skipSign,
		SignCertThumbprint: *signCert,
		SignPFX:            *signPFX,
		SignPFXPassword:    *signPassword,
		TimestampURL:       *timestampURL,
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
	if *cudaHome != "" || *nvcc != "" || *cxx != "" || *cc != "" {
		tc, _ := computebuild.DiscoverToolchain()
		if tc == nil {
			tc = &computebuild.Toolchain{IsWindows: runtime.GOOS == "windows"}
		}
		if *cudaHome != "" {
			tc.CUDAHome = *cudaHome
			tc.CUDAInc = []string{filepath.Join(*cudaHome, "include")}
			if tc.IsWindows {
				tc.CUDALib = []string{filepath.Join(*cudaHome, "lib", "x64")}
			} else {
				tc.CUDALib = []string{filepath.Join(*cudaHome, "lib64")}
			}
		}
		if *nvcc != "" {
			tc.NVCC = *nvcc
		}
		if *cxx != "" {
			tc.CXX = *cxx
		}
		if *cc != "" {
			tc.CC = *cc
		}
		cfg.Toolchain = tc
	}

	if *jsonOut {
		cfg.Stdout = io.Discard
	}

	start := time.Now()
	err := computebuild.RunCUDA(context.Background(), cfg)
	dur := time.Since(start).Milliseconds()

	if *jsonOut {
		res := CUDABuildResult{
			Schema:     "fak.cuda-build.v1",
			Command:    sub,
			Arch:       cfg.Arch,
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
		fmt.Fprintf(stderr, "build-cuda: %v\n", err)
		return 1
	}

	return 0
}
