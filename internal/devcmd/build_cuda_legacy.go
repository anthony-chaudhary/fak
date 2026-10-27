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
	"strings"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

func runCUDALegacy(stdout, stderr io.Writer, argv []string, build func(context.Context, *computebuild.CUDAConfig) error, command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error) int {
	sub := "test"
	if len(argv) > 0 {
		sub, argv = argv[0], argv[1:]
	}
	args := []string{sub, "--smoke=false"}
	switch sub {
	case "check", "build", "test":
		if len(argv) != 0 {
			fmt.Fprintln(stderr, "legacy CUDA command accepts no trailing arguments")
			return 2
		}
	case "binary":
		if len(argv) != 2 {
			fmt.Fprintln(stderr, "usage: build_cuda.sh binary <pkg> <out>")
			return 2
		}
		args = append(args, "--out-pkg", argv[0], "--out-bin", argv[1])
	case "bench":
		if len(argv) > 2 {
			return 2
		}
		if len(argv) > 0 {
			args = append(args, "--bench-model", argv[0])
		}
		if len(argv) > 1 {
			args = append(args, "--bench-steps", argv[1])
		}
	default:
		fmt.Fprintf(stderr, "unknown legacy CUDA command %q\n", sub)
		return 2
	}
	if sub != "check" {
		goEnv := computebuild.MergeEnviron(os.Environ(), map[string]string{"GOTOOLCHAIN": cudaLegacyDefault("GOTOOLCHAIN", "auto")})
		var root bytes.Buffer
		if err := command(context.Background(), "go", []string{"env", "GOROOT"}, "", goEnv, &root, stderr); err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		for _, name := range []string{"src/encoding/base32/base32.go", "src/go/types/api.go"} {
			if !computebuild.FileExists(filepath.Join(strings.TrimSpace(root.String()), filepath.FromSlash(name))) {
				fmt.Fprintln(stderr, "incomplete Go toolchain: missing", name)
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
		if selected.Toolchain != nil && sub != "check" {
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

func cudaLegacyDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
