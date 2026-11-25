package devcmd

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// cudaLegacyGoBin is a var so tests can point the PATH prepend at a directory that does not exist.
var cudaLegacyGoBin = "/usr/local/go/bin"

const cudaLegacyUsage = "usage: fak-dev build-cuda legacy [check|build|test|bench [model] [steps]]\n" +
	"       fak-dev build-cuda legacy binary <pkg> <out-abs-path>"

// runCUDALegacy is the native driver behind `fak-dev build-cuda legacy`. It is self-sufficient:
// it validates FAK_CUDA_ARCH, prepares the process environment, refuses an incomplete Go toolchain
// and runs the GPU-free check verb, so it can be invoked directly without a shell wrapper.
//
//	usage: fak-dev build-cuda legacy [check|build|test|bench]
//	       fak-dev build-cuda legacy binary <pkg> <out-abs-path>
//	env:   FAK_CUDA_ARCH=sm_89|sm_90|sm_100|sm_120  unset = fat build of every cuda_arch.txt line plus PTX;
//	                                                 "89" is accepted; "a" needs sm_90+, "f" needs sm_100+
//	       FAK_CUDA_V41=0                            sm_120 only: skip the default V4.1 tags
//	       FAK_CUDA_NCCL=1                           also compile cuda_nccl.cu (Go tags cuda,nccl)
//	       CUDA_HOME=/usr/local/cuda                 default ~/cudaenv, else the nvcc on PATH
//	       NCCL_HOME=/tmp/nccl-root                  optional user-space nccl.h/libnccl.so root
//	       PYTHON, CC                                check verb: interpreter and C compiler overrides
//
// The check verb needs no Go toolchain preflight, CUDA_HOME or nvcc.
func runCUDALegacy(stdout, stderr io.Writer, argv []string, build func(context.Context, *computebuild.CUDAConfig) error, command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error) int {
	sub := "test"
	if len(argv) > 0 {
		sub, argv = argv[0], argv[1:]
	}
	if arch := os.Getenv("FAK_CUDA_ARCH"); arch != "" {
		if err := checkCUDALegacyArch(arch); err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
	}
	args := []string{sub, "--smoke=false"}
	switch sub {
	case "check", "build", "test":
		if len(argv) != 0 {
			fmt.Fprintln(stderr, "legacy CUDA command accepts no trailing arguments")
			fmt.Fprintln(stderr, cudaLegacyUsage)
			return 2
		}
	case "binary":
		if len(argv) != 2 {
			fmt.Fprintln(stderr, "usage: fak-dev build-cuda legacy binary <pkg> <out-abs-path>")
			return 2
		}
		args = append(args, "--out-pkg", argv[0], "--out-bin", argv[1])
	case "bench":
		if len(argv) > 2 {
			fmt.Fprintln(stderr, cudaLegacyUsage)
			return 2
		}
		if len(argv) > 0 {
			args = append(args, "--bench-model", argv[0])
		}
		if len(argv) > 1 {
			args = append(args, "--bench-steps", argv[1])
		}
	default:
		fmt.Fprintf(stderr, "unknown legacy CUDA command %q\n%s\n", sub, cudaLegacyUsage)
		return 2
	}
	if sub != "check" {
		// The check verb runs on the caller's PATH and GOTOOLCHAIN, as the old shim did.
		applyCUDALegacyEnvironment()
		if _, err := exec.LookPath("go"); err != nil {
			fmt.Fprintln(stderr, "[cuda] Go 1.26+ is required (no go executable on PATH)")
			return 2
		}
		goRoot := cudaLegacyGoRoot(command)
		for _, name := range []string{"src/encoding/base32/base32.go", "src/go/types/api.go"} {
			if goRoot == "" || !computebuild.FileExists(filepath.Join(goRoot, filepath.FromSlash(name))) {
				where := goRoot
				if where == "" {
					where = "<unknown>"
				}
				fmt.Fprintf(stderr, "[cuda] incomplete Go toolchain at %s: missing %s; install a complete Go 1.26+ distribution before the CUDA build\n", where, name)
				return 2
			}
		}
		home := os.Getenv("CUDA_HOME")
		if home == "" {
			userHome := cudaLegacyDefault("HOME", "/opt")
			home = filepath.Join(userHome, "cudaenv")
		}
		if info, err := os.Stat(filepath.Join(home, "bin", "nvcc")); err != nil || !info.Mode().IsRegular() || runtime.GOOS != "windows" && info.Mode()&0111 == 0 {
			if nvcc, err := exec.LookPath("nvcc"); err == nil {
				home = filepath.Dir(filepath.Dir(nvcc))
			}
		}
		args = append(args, "--cuda-home", home, "--cxx", cudaLegacyDefault("FAK_NVCC_CCBIN", "/usr/bin/g++"), "--cc", cudaLegacyDefault("CC", "/usr/bin/gcc"))
	}
	return runBuildCUDAWith(stdout, stderr, args, func(ctx context.Context, selected *computebuild.CUDAConfig) error {
		if sub == "check" {
			selected.Toolchain = &computebuild.Toolchain{IsWindows: runtime.GOOS == "windows", CC: cudaLegacyCheckCC()}
			return build(computebuild.WithStrictCUDACheck(ctx), selected)
		}
		if selected.Toolchain != nil {
			cloned := *selected.Toolchain
			cloned.VsDevEnv = make(map[string]string, len(selected.Toolchain.VsDevEnv)+1)
			for key, value := range selected.Toolchain.VsDevEnv {
				cloned.VsDevEnv[key] = value
			}
			cloned.VsDevEnv["CXX"] = cudaLegacyDefault("CXX", "/usr/bin/g++")
			selected.Toolchain = &cloned
		}
		return build(ctx, selected)
	}, command)
}

// checkCUDALegacyArch applies the same supported-arch rule as the native build, with the
// legacy error wording, so a bad FAK_CUDA_ARCH fails before any toolchain work.
func checkCUDALegacyArch(arch string) error {
	if !strings.HasPrefix(arch, "sm_") {
		arch = "sm_" + arch
	}
	root, err := computebuild.ResolveRepoRoot("")
	if err != nil {
		return fmt.Errorf("ERROR: cannot validate CUDA arch '%s': %w", arch, err)
	}
	manifest := computebuild.CUDAArchManifestPath(root)
	content, err := os.ReadFile(manifest)
	if err != nil {
		return fmt.Errorf("ERROR: cannot read CUDA arch manifest %s: %w", manifest, err)
	}
	supported := computebuild.ParseCUDAArchList(string(content))
	base, _, ok := computebuild.SplitCUDAArchSuffix(arch)
	if !ok || !slices.Contains(supported, base) {
		return fmt.Errorf("ERROR: unsupported CUDA arch '%s'; choose one from %s: %s (a/f suffix on sm_90+/sm_100+)", arch, manifest, strings.Join(supported, " "))
	}
	return nil
}

// applyCUDALegacyEnvironment is applied to the process itself so every child go command inherits
// it; the computebuild go invocations pass a nil environment.
func applyCUDALegacyEnvironment() {
	if runtime.GOOS != "windows" && computebuild.DirExists(cudaLegacyGoBin) {
		_ = os.Setenv("PATH", prependPathDir(os.Getenv("PATH"), cudaLegacyGoBin))
	}
	if os.Getenv("GOTOOLCHAIN") == "" {
		_ = os.Setenv("GOTOOLCHAIN", "auto")
	}
}

func prependPathDir(pathList, dir string) string {
	if parts := filepath.SplitList(pathList); len(parts) > 0 && parts[0] == dir {
		return pathList
	}
	if pathList == "" {
		return dir
	}
	return dir + string(filepath.ListSeparator) + pathList
}

// cudaLegacyGoRoot returns the trimmed GOROOT, or "" when go cannot report one.
func cudaLegacyGoRoot(command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error) string {
	var root bytes.Buffer
	if err := command(context.Background(), "go", []string{"env", "GOROOT"}, "", nil, &root, io.Discard); err != nil {
		return ""
	}
	return strings.TrimSpace(root.String())
}

// cudaLegacyCheckCC resolves the C compiler the header parse uses: CC, else the first of gcc, clang, cc on PATH.
func cudaLegacyCheckCC() string {
	if cc := os.Getenv("CC"); cc != "" {
		return cc
	}
	for _, name := range []string{"gcc", "clang", "cc"} {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}
	return ""
}

func cudaLegacyDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
