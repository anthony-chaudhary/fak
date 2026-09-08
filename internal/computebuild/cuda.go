package computebuild

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// ParseCUDAArchMatrix parses architecture declarations and generates nvcc -gencode flags.
func ParseCUDAArchMatrix(archText string, targetArch string) ([]string, string, error) {
	var supported []string
	for _, line := range strings.Split(archText, "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimSuffix(line, "\r")
		if line != "" {
			supported = append(supported, line)
		}
	}
	if len(supported) == 0 {
		return nil, "", fmt.Errorf("cuda_arch.txt is empty or contains no valid architectures")
	}

	var gencode []string
	var buildArchs string

	if targetArch != "" {
		arch := targetArch
		if !strings.HasPrefix(arch, "sm_") {
			arch = "sm_" + arch
		}
		found := false
		for _, s := range supported {
			if s == arch {
				found = true
				break
			}
		}
		if !found {
			return nil, "", fmt.Errorf("unsupported CUDA arch %q; choose one from: %s", targetArch, strings.Join(supported, ", "))
		}
		cc := strings.TrimPrefix(arch, "sm_")
		gencode = []string{"-gencode", fmt.Sprintf("arch=compute_%s,code=%s", cc, arch)}
		buildArchs = fmt.Sprintf("%s (single-arch dev build)", arch)
	} else {
		var lastCC string
		for _, arch := range supported {
			cc := strings.TrimPrefix(arch, "sm_")
			gencode = append(gencode, "-gencode", fmt.Sprintf("arch=compute_%s,code=%s", cc, arch))
			lastCC = cc
		}
		gencode = append(gencode, "-gencode", fmt.Sprintf("arch=compute_%s,code=compute_%s", lastCC, lastCC))
		buildArchs = fmt.Sprintf("%s + compute_%s PTX", strings.Join(supported, ", "), lastCC)
	}

	return gencode, buildArchs, nil
}

// LoadCUDAArchMatrix loads cuda_arch.txt from repoRoot/internal/compute/ and parses gencode flags.
func LoadCUDAArchMatrix(repoRoot string, targetArch string) ([]string, string, error) {
	archFile := filepath.Join(repoRoot, "internal", "compute", "cuda_arch.txt")
	content, err := os.ReadFile(archFile)
	if err != nil {
		return nil, "", fmt.Errorf("reading %s: %w", archFile, err)
	}
	return ParseCUDAArchMatrix(string(content), targetArch)
}

// CompileCUDAKernels compiles cuda_kernels.cu (and optional NCCL files) into object files.
func CompileCUDAKernels(ctx context.Context, tc *Toolchain, repoRoot string, arch string, enableNCCL bool, stdout io.Writer) ([]string, error) {
	if tc == nil || tc.NVCC == "" {
		return nil, fmt.Errorf("nvcc not found — install CUDA Toolkit and set CUDA_PATH or CUDA_HOME")
	}

	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	gencodeArgs, buildArchs, err := LoadCUDAArchMatrix(repoRoot, arch)
	if err != nil {
		return nil, err
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[cuda] nvcc compile kernels (%s) ...\n", buildArchs)
	}

	var objs []string
	env := os.Environ()
	if len(tc.VsDevEnv) > 0 {
		env = MergeEnviron(env, tc.VsDevEnv)
	}

	if tc.IsWindows {
		obj := filepath.Join(pkgDir, "cuda_kernels.obj")
		nvccArgs := []string{"-O3", "-std=c++14"}
		nvccArgs = append(nvccArgs, gencodeArgs...)
		nvccArgs = append(nvccArgs, "-Xcompiler", "/MD", "-c", "cuda_kernels.cu", "-o", obj)
		if err := RunCmd(ctx, tc.NVCC, nvccArgs, pkgDir, env, stdout, os.Stderr); err != nil {
			return nil, fmt.Errorf("nvcc compile cuda_kernels.cu: %w", err)
		}
		objs = append(objs, obj)
	} else {
		obj := filepath.Join(pkgDir, "cuda_kernels.o")
		nvccArgs := []string{"-O3", "-std=c++14"}
		nvccArgs = append(nvccArgs, gencodeArgs...)
		if tc.CXX != "" {
			nvccArgs = append(nvccArgs, "-ccbin", tc.CXX)
		}
		for _, inc := range tc.CUDAInc {
			nvccArgs = append(nvccArgs, "-I"+inc)
		}
		nvccArgs = append(nvccArgs, "-Xcompiler", "-fPIC", "-c", "cuda_kernels.cu", "-o", obj)
		if err := RunCmd(ctx, tc.NVCC, nvccArgs, pkgDir, env, stdout, os.Stderr); err != nil {
			return nil, fmt.Errorf("nvcc compile cuda_kernels.cu: %w", err)
		}
		objs = append(objs, obj)

		if enableNCCL {
			for _, ncclFile := range []string{"cuda_nccl.cu", "cuda_nccl_pg.cu"} {
				ncclObj := filepath.Join(pkgDir, strings.TrimSuffix(ncclFile, ".cu")+".o")
				nArgs := []string{"-O3", "-std=c++14"}
				nArgs = append(nArgs, gencodeArgs...)
				if tc.CXX != "" {
					nArgs = append(nArgs, "-ccbin", tc.CXX)
				}
				for _, inc := range tc.CUDAInc {
					nArgs = append(nArgs, "-I"+inc)
				}
				nArgs = append(nArgs, "-Xcompiler", "-fPIC", "-c", ncclFile, "-o", ncclObj)
				if err := RunCmd(ctx, tc.NVCC, nArgs, pkgDir, env, stdout, os.Stderr); err != nil {
					return nil, fmt.Errorf("nvcc compile %s: %w", ncclFile, err)
				}
				objs = append(objs, ncclObj)
			}
		}
	}
	return objs, nil
}

// ArchiveCUDALib packages the compiled object files into libfakcuda.a.
func ArchiveCUDALib(ctx context.Context, tc *Toolchain, pkgDir string, objs []string, stdout io.Writer) (*BuildArtifact, error) {
	libOut := filepath.Join(pkgDir, "libfakcuda.a")
	env := os.Environ()
	if tc != nil && len(tc.VsDevEnv) > 0 {
		env = MergeEnviron(env, tc.VsDevEnv)
	}

	if tc != nil && tc.AR != "" {
		arArgs := append([]string{"rcs", libOut}, objs...)
		if err := RunCmd(ctx, tc.AR, arArgs, pkgDir, env, stdout, os.Stderr); err != nil {
			return nil, fmt.Errorf("archiving libfakcuda.a via ar: %w", err)
		}
	} else if tc != nil && tc.IsWindows {
		// Fallback to nvcc --lib if ar is absent on Windows
		nvccLibArgs := append([]string{"--lib", "--output-file", libOut}, objs...)
		if err := RunCmd(ctx, tc.NVCC, nvccLibArgs, pkgDir, env, stdout, os.Stderr); err != nil {
			return nil, fmt.Errorf("archiving libfakcuda.a via nvcc --lib: %w", err)
		}
	} else {
		return nil, fmt.Errorf("no archiver (ar/llvm-ar) available to build libfakcuda.a")
	}

	fi, err := os.Stat(libOut)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", libOut, err)
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[cuda] built %s (%d bytes)\n", libOut, fi.Size())
	}
	return &BuildArtifact{
		Path:      libOut,
		SizeBytes: fi.Size(),
	}, nil
}

// BuildCUDALib compiles cuda_kernels.cu (and optional NCCL files) and archives them into libfakcuda.a.
func BuildCUDALib(ctx context.Context, tc *Toolchain, repoRoot string, arch string, enableNCCL bool, stdout io.Writer) (*BuildArtifact, error) {
	objs, err := CompileCUDAKernels(ctx, tc, repoRoot, arch, enableNCCL, stdout)
	if err != nil {
		return nil, err
	}
	return ArchiveCUDALib(ctx, tc, filepath.Join(repoRoot, "internal", "compute"), objs, stdout)
}

// SignArtifact signs a binary with Authenticode on Windows using signtool.exe.
func SignArtifact(ctx context.Context, tc *Toolchain, cfg *CUDAConfig, artifactPath string) error {
	if cfg == nil || cfg.SkipSign || tc == nil || !tc.IsWindows {
		return nil
	}
	if tc.SignTool == "" {
		return fmt.Errorf("signtool.exe not found — install Windows SDK or pass SkipSign")
	}

	tsURL := cfg.TimestampURL
	if tsURL == "" {
		tsURL = "http://timestamp.digicert.com"
	}

	var signArgs []string
	if cfg.SignCertThumbprint != "" {
		signArgs = []string{"sign", "/sha1", cfg.SignCertThumbprint, "/fd", "SHA256", "/tr", tsURL, "/td", "SHA256", artifactPath}
	} else if cfg.SignPFX != "" {
		signArgs = []string{"sign", "/f", cfg.SignPFX, "/fd", "SHA256", "/tr", tsURL, "/td", "SHA256"}
		if cfg.SignPFXPassword != "" {
			signArgs = append(signArgs, "/p", cfg.SignPFXPassword)
		}
		signArgs = append(signArgs, artifactPath)
	} else {
		return fmt.Errorf("no signing certificate specified; set SignCertThumbprint or SignPFX, or pass SkipSign")
	}

	if cfg.Stdout != nil {
		fmt.Fprintf(cfg.Stdout, "[cuda-win] signtool %s\n", strings.Join(signArgs, " "))
	}
	return RunCmd(ctx, tc.SignTool, signArgs, "", nil, cfg.Stdout, cfg.Stderr)
}

// RunCUDA orchestrates CUDA build, test, and check actions.
func RunCUDA(ctx context.Context, cfg *CUDAConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil CUDAConfig")
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
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, DefaultCUDABuildReceiptPath)
	} else if !filepath.IsAbs(cfg.ReceiptPath) {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, cfg.ReceiptPath)
	}
	if cfg.Command == "binary" && !cfg.SkipSmoke {
		cfg.Smoke = true
	}

	tracker := newReceiptTracker("cuda", cfg.Command, cfg.ReceiptPath)
	cfg.Receipt = tracker.receipt
	defer func() {
		_ = tracker.finish(cfg.ReceiptPath)
	}()

	goTags := "cuda"
	if cfg.EnableNCCL {
		goTags = "cuda,nccl"
	}

	switch cfg.Command {
	case "check":
		return tracker.recordPhase("preflight_check", func() error {
			// 1. Python ABI parity check
			py, err := exec.LookPath("python3")
			if err != nil {
				py, err = exec.LookPath("python")
			}
			if err == nil {
				if cfg.Stdout != nil {
					fmt.Fprintln(cfg.Stdout, "[cuda] GPU-free ABI/header portability check ...")
				}
				parityScript := filepath.Join(cfg.RepoRoot, "tools", "cuda_abi_parity.py")
				if FileExists(parityScript) {
					_ = RunCmd(ctx, py, []string{parityScript, "--check"}, cfg.RepoRoot, nil, cfg.Stdout, cfg.Stderr)
				}
			}

			// 2. Architecture matrix test
			if cfg.Stdout != nil {
				fmt.Fprintln(cfg.Stdout, "[cuda] GPU-free architecture matrix check ...")
			}
			if err := RunCmd(ctx, "go", []string{"test", "./internal/cudaarch"}, cfg.RepoRoot, nil, cfg.Stdout, cfg.Stderr); err != nil {
				return err
			}

			// 3. Strict standalone header parse
			if cfg.Toolchain.CC != "" {
				headerPath := filepath.Join(cfg.PkgDir, "cuda_backend.h")
				if FileExists(headerPath) {
					if cfg.Stdout != nil {
						fmt.Fprintf(cfg.Stdout, "[cuda] strict standalone parse cuda_backend.h (%s) ...\n", cfg.Toolchain.CC)
					}
					headerArgs := []string{"-x", "c", "-std=c11", "-fsyntax-only", "-Wall", "-Werror", headerPath}
					_ = RunCmd(ctx, cfg.Toolchain.CC, headerArgs, cfg.PkgDir, nil, cfg.Stdout, cfg.Stderr)
				}
			}
			return nil
		})

	case "build":
		var objs []string
		if err := tracker.recordPhase("compile_kernels", func() error {
			var err error
			objs, err = CompileCUDAKernels(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Arch, cfg.EnableNCCL, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		var artifact *BuildArtifact
		if err := tracker.recordPhase("archive", func() error {
			var err error
			artifact, err = ArchiveCUDALib(ctx, cfg.Toolchain, cfg.PkgDir, objs, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		tracker.receipt.Artifact = artifact

		cgoEnv := SynthesizeCUDACgoEnv(cfg.Toolchain, cfg.PkgDir, cfg.EnableNCCL, nil)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}

		outBin := cfg.OutBin
		if cfg.Toolchain.IsWindows && outBin == "" {
			outBin = "fak.exe"
		}
		if outBin != "" {
			outPkg := cfg.OutPkg
			if outPkg == "" {
				outPkg = "./cmd/fak"
			}
			args := []string{"build", "-tags", goTags, "-o", outBin, outPkg}
			if err := tracker.recordPhase("build_or_test", func() error {
				if err := RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr); err != nil {
					return err
				}
				if cfg.Toolchain.IsWindows {
					artifactPath := filepath.Join(cfg.RepoRoot, outBin)
					if err := SignArtifact(ctx, cfg.Toolchain, cfg, artifactPath); err != nil {
						return err
					}
				}
				return nil
			}); err != nil {
				return err
			}
			outBinPath := outBin
			if !filepath.IsAbs(outBinPath) {
				outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
			}
			if FileExists(outBinPath) {
				if art, err := InspectArtifact(outBinPath); err == nil {
					if cfg.Toolchain.IsWindows && !cfg.SkipSign {
						art.Signed = true
					}
					tracker.receipt.Artifact = art
				}
			}
			if cfg.Smoke && FileExists(outBinPath) {
				if err := tracker.recordSmoke(ctx, outBinPath); err != nil {
					return err
				}
			}
		} else {
			args := []string{"build", "-tags", goTags, "./internal/compute/"}
			if err := tracker.recordPhase("build_or_test", func() error {
				return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
			}); err != nil {
				return err
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
		var objs []string
		if err := tracker.recordPhase("compile_kernels", func() error {
			var err error
			objs, err = CompileCUDAKernels(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Arch, cfg.EnableNCCL, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("archive", func() error {
			_, err := ArchiveCUDALib(ctx, cfg.Toolchain, cfg.PkgDir, objs, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeCUDACgoEnv(cfg.Toolchain, cfg.PkgDir, cfg.EnableNCCL, nil)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", goTags, "-o", cfg.OutBin, cfg.OutPkg}
		if err := tracker.recordPhase("build_or_test", func() error {
			if err := RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr); err != nil {
				return err
			}
			if cfg.Toolchain.IsWindows {
				artifact := cfg.OutBin
				if !filepath.IsAbs(artifact) {
					artifact = filepath.Join(cfg.RepoRoot, artifact)
				}
				if err := SignArtifact(ctx, cfg.Toolchain, cfg, artifact); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return err
		}
		outBinPath := cfg.OutBin
		if !filepath.IsAbs(outBinPath) {
			outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
		}
		if FileExists(outBinPath) {
			if art, err := InspectArtifact(outBinPath); err == nil {
				if cfg.Toolchain.IsWindows && !cfg.SkipSign {
					art.Signed = true
				}
				tracker.receipt.Artifact = art
			}
		}
		if cfg.Smoke && FileExists(outBinPath) {
			if err := tracker.recordSmoke(ctx, outBinPath); err != nil {
				return err
			}
		}
		return nil

	case "test":
		var objs []string
		if err := tracker.recordPhase("compile_kernels", func() error {
			var err error
			objs, err = CompileCUDAKernels(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Arch, cfg.EnableNCCL, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("archive", func() error {
			_, err := ArchiveCUDALib(ctx, cfg.Toolchain, cfg.PkgDir, objs, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeCUDACgoEnv(cfg.Toolchain, cfg.PkgDir, cfg.EnableNCCL, nil)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"test", "-tags", goTags, "-count=1", "-run", "CUDA|HALDevice", "./internal/compute/", "./internal/model/"}
		return tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		})

	case "bench":
		var objs []string
		if err := tracker.recordPhase("compile_kernels", func() error {
			var err error
			objs, err = CompileCUDAKernels(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Arch, cfg.EnableNCCL, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("archive", func() error {
			_, err := ArchiveCUDALib(ctx, cfg.Toolchain, cfg.PkgDir, objs, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeCUDACgoEnv(cfg.Toolchain, cfg.PkgDir, cfg.EnableNCCL, nil)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		dir := cfg.BenchModelDir
		if dir == "" {
			dir = "internal/model/.cache/smollm2-135m"
		}
		steps := cfg.BenchDecodeSteps
		if steps <= 0 {
			steps = 128
		}
		args := []string{
			"run", "-tags", goTags, "./cmd/modelbench",
			"-dir", dir,
			"-backend", "cuda",
			"-decode-steps", strconv.Itoa(steps),
			"-decode-reps", "5",
			"-decode-prompt", "16",
		}
		return tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		})

	default:
		err := fmt.Errorf("unknown subcommand: %s (use check|build|test|bench|binary)", cfg.Command)
		tracker.receipt.Outcome = "failed"
		tracker.receipt.ExitCode = 1
		tracker.receipt.Error = err.Error()
		return err
	}
}
